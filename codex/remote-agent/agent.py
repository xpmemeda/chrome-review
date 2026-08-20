#!/usr/bin/env python3
"""Small HTTP agent for development machines that cannot be reached by SSH."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import signal
import socket
import ssl
import subprocess
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


MAX_JSON_BYTES = 2 * 1024 * 1024
COPY_CHUNK = 1024 * 1024
AGENT_PORT = 18765


class AgentState:
    def __init__(self, roots: list[Path], state_dir: Path, max_upload: int):
        self.roots = [path.expanduser().resolve() for path in roots]
        self.state_dir = state_dir.expanduser().resolve()
        self.logs_dir = self.state_dir / "jobs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.state_dir / "audit.jsonl"
        self.max_upload = max_upload
        self.jobs: dict[str, dict] = {}
        self.lock = threading.RLock()

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


class AgentServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, state: AgentState):
        self.address_family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET
        super().__init__(address, handler)
        self.state = state


class Handler(BaseHTTPRequestHandler):
    server: AgentServer

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_JSON_BYTES:
            raise ValueError("invalid JSON body size")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def _prepare_command(self, body: dict) -> tuple[str, Path, dict[str, str]]:
        command = body.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be a non-empty string")
        cwd = self.server.state.resolve_path(body.get("cwd") or str(self.server.state.roots[0]), must_exist=True)
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

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/v1/health":
                self._json(HTTPStatus.OK, {
                    "ok": True,
                    "hostname": os.uname().nodename,
                    "pid": os.getpid(),
                    "roots": [str(p) for p in self.server.state.roots],
                })
                return
            if parsed.path == "/v1/files":
                self._download(parse_qs(parsed.query))
                return
            parts = parsed.path.strip("/").split("/")
            if len(parts) in (3, 4) and parts[:2] == ["v1", "jobs"]:
                if len(parts) == 4 and parts[3] != "logs":
                    raise ValueError("unknown job endpoint")
                self._job_info(parts[2], logs=len(parts) == 4, query=parse_qs(parsed.query))
                return
            self._error(HTTPStatus.NOT_FOUND, "endpoint not found")
        except (ValueError, PermissionError, FileNotFoundError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self.server.state.audit("internal_error", method="GET", path=self.path, error=repr(exc))
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/v1/exec":
                self._exec(self._read_json())
                return
            if parsed.path == "/v1/jobs":
                self._start_job(self._read_json())
                return
            parts = parsed.path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["v1", "jobs"] and parts[3] == "stop":
                self._stop_job(parts[2])
                return
            self._error(HTTPStatus.NOT_FOUND, "endpoint not found")
        except (ValueError, PermissionError, FileNotFoundError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self.server.state.audit("internal_error", method="POST", path=self.path, error=repr(exc))
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_PUT(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path != "/v1/files":
                self._error(HTTPStatus.NOT_FOUND, "endpoint not found")
                return
            self._upload(parse_qs(parsed.query))
        except (ValueError, PermissionError, FileNotFoundError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self.server.state.audit("internal_error", method="PUT", path=self.path, error=repr(exc))
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _exec(self, body: dict) -> None:
        command, cwd, env = self._prepare_command(body)
        timeout = float(body.get("timeout", 300))
        if not 0 < timeout <= 86400:
            raise ValueError("timeout must be between 0 and 86400 seconds")
        started = time.monotonic()
        process = subprocess.Popen(
                ["/bin/bash", "-lc", command], cwd=cwd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace",
                start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            payload = {
                "exit_code": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "duration_seconds": time.monotonic() - started,
            }
            self.server.state.audit("exec", command=command, cwd=str(cwd), exit_code=process.returncode)
            self._json(HTTPStatus.OK, payload)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            self.server.state.audit("exec_timeout", command=command, cwd=str(cwd), timeout=timeout)
            self._json(HTTPStatus.REQUEST_TIMEOUT, {
                "error": "command timed out",
                "stdout": stdout,
                "stderr": stderr,
            })

    def _start_job(self, body: dict) -> None:
        command, cwd, env = self._prepare_command(body)
        job_id = uuid.uuid4().hex
        log_path = self.server.state.logs_dir / f"{job_id}.log"
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
        with self.server.state.lock:
            self.server.state.jobs[job_id] = job
        self.server.state.audit("job_start", job_id=job_id, command=command, cwd=str(cwd), pid=process.pid)
        self._json(HTTPStatus.CREATED, self._public_job(job))

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
        with self.server.state.lock:
            job = self.server.state.jobs.get(job_id)
        if job is None:
            raise ValueError("unknown job id")
        return job

    def _job_info(self, job_id: str, *, logs: bool, query: dict) -> None:
        job = self._get_job(job_id)
        if not logs:
            self._json(HTTPStatus.OK, self._public_job(job))
            return
        offset = int(query.get("offset", ["0"])[0])
        limit = min(int(query.get("limit", [str(1024 * 1024)])[0]), 4 * 1024 * 1024)
        if offset < 0 or limit < 1:
            raise ValueError("offset and limit must be positive")
        with open(job["log_path"], "rb") as stream:
            stream.seek(offset)
            data = stream.read(limit)
            next_offset = stream.tell()
        self._json(HTTPStatus.OK, {
            **self._public_job(job), "offset": offset, "next_offset": next_offset,
            "log": data.decode("utf-8", errors="replace"),
        })

    def _stop_job(self, job_id: str) -> None:
        job = self._get_job(job_id)
        process = job["process"]
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        self.server.state.audit("job_stop", job_id=job_id, exit_code=process.returncode)
        self._json(HTTPStatus.OK, self._public_job(job))

    def _upload(self, query: dict) -> None:
        raw_path = query.get("path", [""])[0]
        path = self.server.state.resolve_path(raw_path)
        length = int(self.headers.get("Content-Length", "0"))
        if length < 0 or length > self.server.state.max_upload:
            raise ValueError("invalid or excessive upload size")
        expected = self.headers.get("X-Content-SHA256", "")
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.upload")
        digest = hashlib.sha256()
        remaining = length
        try:
            with temp.open("wb") as stream:
                while remaining:
                    chunk = self.rfile.read(min(COPY_CHUNK, remaining))
                    if not chunk:
                        raise ValueError("upload ended early")
                    stream.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
            actual = digest.hexdigest()
            if expected and not hmac.compare_digest(expected.lower(), actual):
                raise ValueError("SHA-256 mismatch")
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        self.server.state.audit("upload", path=str(path), size=length, sha256=actual)
        self._json(HTTPStatus.CREATED, {"path": str(path), "size": length, "sha256": actual})

    def _download(self, query: dict) -> None:
        path = self.server.state.resolve_path(query.get("path", [""])[0], must_exist=True)
        if not path.is_file():
            raise ValueError("path is not a file")
        size = path.stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("X-Content-SHA256", sha256_file(path))
        self.end_headers()
        with path.open("rb") as stream:
            while chunk := stream.read(COPY_CHUNK):
                self.wfile.write(chunk)
        self.server.state.audit("download", path=str(path), size=size)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(COPY_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="::", help="listen address; defaults to all IPv6 interfaces")
    parser.add_argument("--root", action="append", default=[], help="allowed filesystem root; repeatable")
    parser.add_argument("--state-dir", default="~/.local/state/dev-agent")
    parser.add_argument("--max-upload-gib", type=float, default=20)
    parser.add_argument("--tls-cert")
    parser.add_argument("--tls-key")
    args = parser.parse_args()
    if bool(args.tls_cert) != bool(args.tls_key):
        parser.error("--tls-cert and --tls-key must be provided together")
    return args


def main() -> None:
    args = parse_args()
    roots = [Path(value) for value in args.root] or [Path.cwd()]
    state = AgentState(roots, Path(args.state_dir), int(args.max_upload_gib * 1024**3))
    server = AgentServer((args.host, AGENT_PORT), Handler, state)
    scheme = "http"
    if args.tls_cert:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.tls_cert, args.tls_key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    display_host = f"[{args.host}]" if ":" in args.host else args.host
    print(f"dev-agent listening on {scheme}://{display_host}:{AGENT_PORT}")
    print("allowed roots:", ", ".join(str(root) for root in state.roots))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
