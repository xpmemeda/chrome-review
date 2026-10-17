#!/usr/bin/env python3
"""Reverse-connected development agent.

Run ``python3 reverse_agent.py <device-name>`` on the target machine. The
agent dials the remote_fleet rendezvous on the work machine with HTTP long
polling, so the target machine does not need to accept inbound connections.
For targets the work machine can reach directly, use direct_agent.py instead.
Each request received over the poll channel runs in its own thread.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from urllib.parse import parse_qs, urlparse


COPY_CHUNK = 1024 * 1024
MAX_DOWNLOAD_CHUNK = 16 * 1024 * 1024
POLL_WAIT = 25
RESULT_RETRIES = 5
DEFAULT_SERVER = "http://74.211.111.53:18765"
# Names of [envs.<name>] tables in the MCP server's hosts.toml; "null" = none.
ENV_CHOICES = ("merlin", "volcano", "null")
# HTTP proxy used to reach the rendezvous from each env's network.
ENV_PROXIES = {
    "merlin": "http://sys-proxy-rd-relay.byted.org:8118",
    "volcano": "http://100.66.18.103:3128",
}
MAX_BACKOFF = 30


class AgentState:
    def __init__(
        self,
        roots: list[Path],
        state_dir: Path,
        max_upload: int | None,
    ):
        self.roots = [path.expanduser().resolve() for path in roots]
        self.state_dir = state_dir.expanduser().resolve()
        self.logs_dir = self.state_dir / "jobs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.state_dir / "audit.jsonl"
        self.max_upload = max_upload
        self.jobs: dict[str, dict] = {}
        self.lock = threading.RLock()
        self.path_locks: dict[str, threading.Lock] = {}

    def resolve_path(self, raw: str, *, must_exist: bool = False) -> Path:
        if not raw:
            raise ValueError("path is required")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = self.roots[0] / path
        path = path.resolve(strict=must_exist)
        if not any(path == root or root in path.parents for root in self.roots):
            raise PermissionError(f"path is outside allowed roots: {path}")
        return path

    def audit(self, action: str, **fields) -> None:
        record = {"time": time.time(), "action": action, **fields}
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self.lock:
            with self.audit_path.open("a", encoding="utf-8") as stream:
                stream.write(line)

    def path_lock(self, path: Path) -> threading.Lock:
        key = str(path)
        with self.lock:
            return self.path_locks.setdefault(key, threading.Lock())


@dataclass
class Request:
    method: str
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]
    body: bytes = b""

    def json(self) -> dict:
        value = json.loads(self.body or b"null")
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def arg(self, name: str, default: str | None = None) -> str | None:
        return self.query.get(name, [default])[0]


@dataclass
class Response:
    status: int
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)


def json_response(status: int, payload: dict) -> Response:
    return Response(
        status,
        json.dumps(payload, ensure_ascii=False).encode(),
        {"Content-Type": "application/json; charset=utf-8"},
    )


class Dispatcher:
    """Executes one agent API request; transport independent."""

    def __init__(self, state: AgentState):
        self.state = state

    def handle(self, request: Request) -> Response:
        try:
            return self._route(request)
        except (ValueError, PermissionError, FileNotFoundError, FileExistsError) as exc:
            log(f"rejected {request.method} {request.path}: {exc}")
            return json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:
            log(f"error in {request.method} {request.path}: {exc!r}")
            self.state.audit(
                "internal_error",
                method=request.method,
                path=request.path,
                error=repr(exc),
            )
            return json_response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _route(self, request: Request) -> Response:
        method, path = request.method, request.path
        if method == "GET":
            if path == "/v1/health":
                return json_response(HTTPStatus.OK, {
                    "ok": True,
                    "hostname": os.uname().nodename,
                    "pid": os.getpid(),
                    "roots": [str(p) for p in self.state.roots],
                })
            if path == "/v1/files":
                return self._download(request)
            if path == "/v1/files/info":
                return self._file_info(request)
            parts = path.strip("/").split("/")
            if len(parts) in (3, 4) and parts[:2] == ["v1", "jobs"]:
                if len(parts) == 4 and parts[3] != "logs":
                    raise ValueError("unknown job endpoint")
                return self._job_info(parts[2], logs=len(parts) == 4, request=request)
        elif method == "POST":
            if path == "/v1/exec":
                return self._exec(request.json())
            if path == "/v1/jobs":
                return self._start_job(request.json())
            if path == "/v1/files/finalize":
                return self._finalize_upload(request.json())
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["v1", "jobs"] and parts[3] == "stop":
                return self._stop_job(parts[2])
        elif method == "PUT":
            if path == "/v1/files/chunk":
                return self._upload_chunk(request)
        return json_response(HTTPStatus.NOT_FOUND, {"error": "endpoint not found"})

    def _prepare_command(self, body: dict) -> tuple[str, Path, dict[str, str]]:
        command = body.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be a non-empty string")
        cwd = self.state.resolve_path(body.get("cwd") or str(self.state.roots[0]), must_exist=True)
        if not cwd.is_dir():
            raise ValueError("cwd must be a directory")
        extra_env = body.get("env") or {}
        if not isinstance(extra_env, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in extra_env.items()
        ):
            raise ValueError("env must be an object containing string values")
        env = os.environ.copy()
        env.update(extra_env)
        return command, cwd, env

    def _exec(self, body: dict) -> Response:
        command, cwd, env = self._prepare_command(body)
        timeout = float(body.get("timeout", 300))
        if not 0 < timeout <= 86400:
            raise ValueError("timeout must be between 0 and 86400 seconds")
        started = time.monotonic()
        tag = f"exec[{uuid.uuid4().hex[:6]}]"
        log(f"{tag} $ {brief(command)}  (cwd={cwd})")
        process = subprocess.Popen(
            ["/bin/bash", "-lc", command], cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace",
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            log(f"{tag} killed after {timeout:g}s timeout")
            self.state.audit("exec_timeout", command=command, cwd=str(cwd), timeout=timeout)
            return json_response(HTTPStatus.REQUEST_TIMEOUT, {
                "error": "command timed out",
                "stdout": stdout,
                "stderr": stderr,
            })
        log(f"{tag} exit={process.returncode} in {time.monotonic() - started:.1f}s")
        self.state.audit("exec", command=command, cwd=str(cwd), exit_code=process.returncode)
        return json_response(HTTPStatus.OK, {
            "exit_code": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "duration_seconds": time.monotonic() - started,
        })

    def _start_job(self, body: dict) -> Response:
        command, cwd, env = self._prepare_command(body)
        job_id = uuid.uuid4().hex
        log_path = self.state.logs_dir / f"{job_id}.log"
        log_stream = log_path.open("ab", buffering=0)
        try:
            process = subprocess.Popen(
                ["/bin/bash", "-lc", command], cwd=cwd, env=env,
                stdout=log_stream, stderr=subprocess.STDOUT, start_new_session=True,
            )
        finally:
            log_stream.close()
        job = {
            "id": job_id, "command": command, "cwd": str(cwd), "pid": process.pid,
            "process": process, "log_path": str(log_path), "started_at": time.time(),
        }
        with self.state.lock:
            self.state.jobs[job_id] = job
        log(f"job[{job_id[:8]}] started pid={process.pid} $ {brief(command)}  (cwd={cwd})")
        threading.Thread(
            target=lambda: log(f"job[{job_id[:8]}] exited code={process.wait()}"),
            name=f"job-{job_id[:8]}",
            daemon=True,
        ).start()
        self.state.audit("job_start", job_id=job_id, command=command, cwd=str(cwd), pid=process.pid)
        return json_response(HTTPStatus.CREATED, self._public_job(job))

    @staticmethod
    def _public_job(job: dict) -> dict:
        process = job["process"]
        code = process.poll()
        return {
            "id": job["id"], "command": job["command"], "cwd": job["cwd"],
            "pid": job["pid"], "started_at": job["started_at"],
            "running": code is None, "exit_code": code,
        }

    def _get_job(self, job_id: str) -> dict:
        with self.state.lock:
            job = self.state.jobs.get(job_id)
        if job is None:
            raise ValueError("unknown job id")
        return job

    def _job_info(self, job_id: str, *, logs: bool, request: Request) -> Response:
        job = self._get_job(job_id)
        if not logs:
            return json_response(HTTPStatus.OK, self._public_job(job))
        offset = int(request.arg("offset", "0"))
        limit = min(int(request.arg("limit", str(1024 * 1024))), 4 * 1024 * 1024)
        if offset < 0 or limit < 1:
            raise ValueError("offset and limit must be positive")
        with open(job["log_path"], "rb") as stream:
            stream.seek(offset)
            data = stream.read(limit)
            next_offset = stream.tell()
        return json_response(HTTPStatus.OK, {
            **self._public_job(job), "offset": offset, "next_offset": next_offset,
            "size": Path(job["log_path"]).stat().st_size,
            "log": data.decode("utf-8", errors="replace"),
        })

    def _stop_job(self, job_id: str) -> Response:
        job = self._get_job(job_id)
        process = job["process"]
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        log(f"job[{job_id[:8]}] stop requested")
        self.state.audit("job_stop", job_id=job_id, exit_code=process.returncode)
        return json_response(HTTPStatus.OK, self._public_job(job))

    def _download(self, request: Request) -> Response:
        path = self.state.resolve_path(request.arg("path", ""), must_exist=True)
        if not path.is_file():
            raise ValueError("path is not a file")
        size = path.stat().st_size
        offset = int(request.arg("offset", "0"))
        limit = int(request.arg("limit", str(MAX_DOWNLOAD_CHUNK)))
        if offset < 0 or offset > size or not 0 <= limit <= MAX_DOWNLOAD_CHUNK:
            raise ValueError(
                f"invalid download offset or limit (limit <= {MAX_DOWNLOAD_CHUNK})"
            )
        with path.open("rb") as stream:
            stream.seek(offset)
            data = stream.read(min(limit, size - offset))
        if offset == 0:
            log(f"download {path} ({size} bytes)")
        self.state.audit("download", path=str(path), offset=offset, size=len(data))
        return Response(
            HTTPStatus.PARTIAL_CONTENT,
            data,
            {"Content-Type": "application/octet-stream", "X-File-Size": str(size)},
        )

    def _file_info(self, request: Request) -> Response:
        path = self.state.resolve_path(request.arg("path", ""))
        if not path.exists():
            return json_response(HTTPStatus.OK, {"exists": False, "size": 0, "sha256": None})
        if not path.is_file():
            raise ValueError("path is not a file")
        size = path.stat().st_size
        raw_length = request.arg("hash_length")
        hash_length = None if raw_length is None else int(raw_length)
        if hash_length is not None and not 0 <= hash_length <= size:
            raise ValueError("invalid hash_length")
        return json_response(HTTPStatus.OK, {
            "exists": True,
            "size": size,
            "sha256": sha256_file(path, hash_length),
        })

    def _upload_chunk(self, request: Request) -> Response:
        path = self.state.resolve_path(request.arg("path", ""))
        offset = int(request.arg("offset", "-1"))
        data = request.body
        if offset < 0 or (
            self.state.max_upload is not None
            and offset + len(data) > self.state.max_upload
        ):
            raise ValueError("invalid upload offset or chunk size")
        expected = request.headers.get("x-content-sha256", "").lower()
        actual = hashlib.sha256(data).hexdigest()
        if expected and not hmac.compare_digest(expected, actual):
            raise ValueError("chunk SHA-256 mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.state.path_lock(path):
            current = path.stat().st_size if path.exists() else 0
            if current != offset:
                raise ValueError(f"partial size mismatch: {current} != {offset}")
            try:
                with path.open("ab") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception:
                with path.open("r+b") as stream:
                    stream.truncate(offset)
                raise
        size = path.stat().st_size
        self.state.audit("upload_chunk", path=str(path), offset=offset, size=len(data))
        return json_response(HTTPStatus.CREATED, {"path": str(path), "size": size, "sha256": actual})

    def _finalize_upload(self, body: dict) -> Response:
        partial = self.state.resolve_path(str(body.get("partial_path", "")), must_exist=True)
        destination = self.state.resolve_path(str(body.get("path", "")))
        size = int(body.get("size", -1))
        expected = str(body.get("sha256", "")).lower()
        overwrite = bool(body.get("overwrite", False))
        if partial != Path(str(destination) + ".partial"):
            raise ValueError("partial_path must be '<path>.partial'")
        if size < 0 or (
            self.state.max_upload is not None
            and size > self.state.max_upload
        ):
            raise ValueError("invalid or excessive upload size")
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("invalid SHA-256")
        with self.state.path_lock(partial):
            if not partial.is_file() or partial.stat().st_size != size:
                raise ValueError("upload size mismatch")
            actual = sha256_file(partial)
            if not hmac.compare_digest(expected, actual):
                raise ValueError("upload SHA-256 mismatch")
            with self.state.path_lock(destination):
                if destination.exists() and not overwrite:
                    raise FileExistsError(f"destination already exists: {destination}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(partial, destination)
        log(f"upload {destination} ({size} bytes)")
        self.state.audit("upload_finalize", path=str(destination), size=size, sha256=actual)
        return json_response(
            HTTPStatus.CREATED,
            {"path": str(destination), "size": size, "sha256": actual},
        )


def sha256_file(path: Path, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = limit
    with path.open("rb") as stream:
        while remaining is None or remaining > 0:
            read_size = COPY_CHUNK if remaining is None else min(COPY_CHUNK, remaining)
            chunk = stream.read(read_size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    if remaining not in (None, 0):
        raise ValueError("file ended before requested hash length")
    return digest.hexdigest()


class FatalError(Exception):
    """The rendezvous rejected this agent; retrying will not help."""


class Connector:
    """Long-polls the rendezvous and runs each received request concurrently."""

    def __init__(
        self,
        server: str,
        name: str,
        dispatcher: Dispatcher,
        env: str | None,
        proxy: str | None,
    ):
        self.server = server.rstrip("/")
        self.name = name
        self.dispatcher = dispatcher
        self.instance = uuid.uuid4().hex
        self.registration = {
            "name": name,
            "instance": self.instance,
            "hostname": os.uname().nodename,
            "pid": os.getpid(),
            "roots": [str(root) for root in dispatcher.state.roots],
            "env": env,
            "wait": POLL_WAIT,
        }
        # Only the rendezvous connection uses this proxy; commands keep their
        # own environment. Without one, urllib follows HTTP(S)_PROXY/NO_PROXY.
        if proxy is None:
            self.opener = urllib.request.build_opener()
        else:
            self.opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": proxy, "https": proxy})
            )

    def _post(self, path: str, payload: dict, timeout: float) -> dict:
        request = urllib.request.Request(
            self.server + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.opener.open(request, timeout=timeout) as response:
            return json.load(response)

    def run_forever(self) -> None:
        backoff = 1
        connected = False
        while True:
            try:
                value = self._post(
                    "/v1/agents/poll",
                    {**self.registration, "wait": POLL_WAIT if connected else 0},
                    timeout=POLL_WAIT + 30,
                )
            except urllib.error.HTTPError as exc:
                message = error_message(exc)
                connected = False
                if exc.code == HTTPStatus.CONFLICT:
                    log(f"registration rejected: {message}; retrying in 10s")
                    time.sleep(10)
                    continue
                if 400 <= exc.code < 500:
                    raise FatalError(f"HTTP {exc.code}: {message}") from None
                log(f"rendezvous error HTTP {exc.code}: {message}; retrying in {backoff}s")
            except (OSError, ValueError) as exc:
                if connected:
                    log(f"disconnected: {exc}")
                else:
                    log(f"cannot reach {self.server}: {exc}; retrying in {backoff}s")
                connected = False
            else:
                if not connected:
                    log(f"connected to {self.server} as {self.name!r}")
                    connected = True
                backoff = 1
                for task in value.get("tasks", []):
                    threading.Thread(
                        target=self._run_task,
                        args=(task,),
                        name=f"task-{str(task.get('id'))[:8]}",
                        daemon=True,
                    ).start()
                continue
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)

    def _run_task(self, task: dict) -> None:
        url = urlparse(str(task.get("path", "")))
        headers = task.get("headers") or {}
        request = Request(
            method=str(task.get("method", "GET")).upper(),
            path=url.path,
            query=parse_qs(url.query),
            headers={str(k).lower(): str(v) for k, v in headers.items()},
            body=base64.b64decode(str(task.get("body", ""))),
        )
        response = self.dispatcher.handle(request)
        payload = {
            "name": self.name,
            "instance": self.instance,
            "id": task.get("id"),
            "status": int(response.status),
            "headers": response.headers,
            "body": base64.b64encode(response.body).decode(),
        }
        delay = 1
        for attempt in range(RESULT_RETRIES):
            try:
                self._post("/v1/agents/result", payload, timeout=120)
                return
            except urllib.error.HTTPError as exc:
                log(f"result for {request.method} {request.path} rejected: HTTP {exc.code}")
                return
            except (OSError, ValueError) as exc:
                if attempt == RESULT_RETRIES - 1:
                    log(f"dropping result for {request.method} {request.path}: {exc}")
                    return
                time.sleep(delay)
                delay = min(delay * 2, MAX_BACKOFF)


def error_message(exc: urllib.error.HTTPError) -> str:
    try:
        return str(json.loads(exc.read()).get("error", ""))
    except (OSError, ValueError, AttributeError):
        return exc.reason or ""


def brief(command: str, limit: int = 300) -> str:
    """Single-line command preview for the console log."""
    text = " ".join(command.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="device name shown by remote_fleet list_hosts")
    parser.add_argument(
        "--server",
        default=os.environ.get("REMOTE_FLEET_SERVER", DEFAULT_SERVER),
        help=f"rendezvous URL (env REMOTE_FLEET_SERVER; default {DEFAULT_SERVER})",
    )
    parser.add_argument(
        "--env",
        choices=ENV_CHOICES,
        required=True,
        help="network env: picks the proxy for reaching the server (null: "
        "HTTP(S)_PROXY from the environment, else direct) and the env "
        "injected for network=true commands",
    )
    parser.add_argument("--root", action="append", default=[], help="allowed filesystem root; repeatable; default /")
    parser.add_argument("--state-dir", default="~/.local/state/dev-agent")
    parser.add_argument(
        "--max-upload-gib",
        type=float,
        default=0,
        help="maximum finalized upload size; 0 means unlimited",
    )
    args = parser.parse_args()
    if args.max_upload_gib < 0:
        parser.error("--max-upload-gib must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    roots = [Path(value) for value in args.root] or [Path("/")]
    max_upload = int(args.max_upload_gib * 1024**3) if args.max_upload_gib else None
    state = AgentState(roots, Path(args.state_dir), max_upload)
    proxy = ENV_PROXIES.get(args.env)
    env = None if args.env == "null" else args.env
    connector = Connector(args.server, args.name, Dispatcher(state), env, proxy)
    route = f"via {proxy}" if proxy else "proxy from environment, if any"
    log(f"agent {args.name!r} (pid {os.getpid()}) dialing {connector.server} ({route})")
    log("allowed roots: " + ", ".join(str(root) for root in state.roots))
    try:
        connector.run_forever()
    except FatalError as exc:
        raise SystemExit(f"rendezvous rejected agent: {exc}")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
