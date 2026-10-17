#!/usr/bin/env python3
"""Directly reachable development agent.

Run ``python3 direct_agent.py`` on a target machine that the work machine can
reach; it listens on port 18765 and the work machine calls it using the URL
registered in remote_fleet's hosts.toml (backend = "agent"). For targets that
cannot accept inbound connections use reverse_agent.py instead.
Each request is served in its own thread.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import signal
import socket
import ssl
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


COPY_CHUNK = 1024 * 1024
MAX_DOWNLOAD_CHUNK = 64 * 1024 * 1024
MAX_BODY_BYTES = 64 * 1024 * 1024
DEFAULT_PORT = 18765


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


class AgentServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128

    def __init__(self, address, handler, dispatcher: Dispatcher):
        self.address_family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET
        super().__init__(address, handler)
        self.dispatcher = dispatcher


class Handler(BaseHTTPRequestHandler):
    """Adapts HTTP requests to the transport-independent Dispatcher."""

    server: AgentServer

    def log_message(self, fmt: str, *args) -> None:
        pass

    def _serve(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 <= length <= MAX_BODY_BYTES:
            response = json_response(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "body too large"})
        else:
            url = urlparse(self.path)
            request = Request(
                method=self.command,
                path=url.path,
                query=parse_qs(url.query),
                headers={key.lower(): value for key, value in self.headers.items()},
                body=self.rfile.read(length) if length else b"",
            )
            response = self.server.dispatcher.handle(request)
        self.send_response(response.status)
        for key, value in response.headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(response.body)))
        self.end_headers()
        self.wfile.write(response.body)

    do_GET = do_POST = do_PUT = _serve


def brief(command: str, limit: int = 300) -> str:
    """Single-line command preview for the console log."""
    text = " ".join(command.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="::", help="listen address; :: means all IPv6 interfaces")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--root", action="append", default=[], help="allowed filesystem root; repeatable; default /")
    parser.add_argument("--state-dir", default="~/.local/state/dev-agent")
    parser.add_argument(
        "--max-upload-gib",
        type=float,
        default=0,
        help="maximum finalized upload size; 0 means unlimited",
    )
    parser.add_argument("--tls-cert")
    parser.add_argument("--tls-key")
    args = parser.parse_args()
    if bool(args.tls_cert) != bool(args.tls_key):
        parser.error("--tls-cert and --tls-key must be provided together")
    if args.max_upload_gib < 0:
        parser.error("--max-upload-gib must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    roots = [Path(value) for value in args.root] or [Path("/")]
    max_upload = int(args.max_upload_gib * 1024**3) if args.max_upload_gib else None
    state = AgentState(roots, Path(args.state_dir), max_upload)
    server = AgentServer((args.host, args.port), Handler, Dispatcher(state))
    scheme = "http"
    if args.tls_cert:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.tls_cert, args.tls_key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    display_host = f"[{args.host}]" if ":" in args.host else args.host
    log(f"direct agent (pid {os.getpid()}) listening on {scheme}://{display_host}:{args.port}")
    log("allowed roots: " + ", ".join(str(root) for root in state.roots))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
