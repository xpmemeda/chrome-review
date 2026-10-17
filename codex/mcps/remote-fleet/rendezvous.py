#!/usr/bin/env python3
"""Rendezvous point for reverse-connected agents.

Agents dial in with HTTP long polling: ``POST /v1/agents/poll`` registers the
agent (or refreshes it) and returns queued requests, ``POST /v1/agents/result``
returns the response of one request. Requests are opaque HTTP-like messages,
so the MCP side keeps talking to the agent's ``/v1/...`` API unchanged.
"""

from __future__ import annotations

import base64
import json
import re
import select
import socket
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable


NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
MAX_BODY_BYTES = 64 * 1024 * 1024
MAX_POLL_WAIT = 30.0
MAX_TASKS_PER_POLL = 64
# An agent is offline when it has no poll in flight and has not polled for
# this long; its queued and dispatched requests then fail immediately.
OFFLINE_AFTER = 15.0
# How often a waiting poll checks whether the agent closed its connection.
ALIVE_CHECK_INTERVAL = 1.0
# Offline devices stay listed this long so list_hosts can explain them.
FORGET_AFTER = 24 * 3600.0
REAP_INTERVAL = 5.0


class ConflictError(Exception):
    """Another live agent instance already holds the requested name."""


@dataclass
class Pending:
    id: str
    message: dict[str, Any]
    event: threading.Event = field(default_factory=threading.Event)
    dispatched: bool = False
    response: tuple[int, dict[str, str], bytes] | None = None
    error: str | None = None


@dataclass
class Device:
    name: str
    instance: str
    hostname: str
    pid: int | None
    roots: list[str]
    env: str | None
    connected_at: float
    last_seen: float
    last_seen_wall: float
    active_polls: int = 0
    queue: deque[Pending] = field(default_factory=deque)
    pending: dict[str, Pending] = field(default_factory=dict)

    def online(self, now: float) -> bool:
        return self.active_polls > 0 or now - self.last_seen < OFFLINE_AFTER

    def public(self, now: float) -> dict[str, Any]:
        return {
            "name": self.name,
            "online": self.online(now),
            "hostname": self.hostname,
            "pid": self.pid,
            "roots": list(self.roots),
            "env": self.env,
            "connected_at": self.connected_at,
            "last_seen": self.last_seen_wall,
            "pending_requests": len(self.pending),
        }


class Rendezvous:
    def __init__(self, name_check: Callable[[str], str | None] | None = None):
        self.name_check = name_check
        self.cond = threading.Condition()
        self.devices: dict[str, Device] = {}
        self.httpd: ThreadingHTTPServer | None = None
        self.stopping = False

    # ----- MCP side -----

    def list_devices(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        with self.cond:
            return [device.public(now) for device in self.devices.values()]

    def device(self, name: str) -> dict[str, Any] | None:
        now = time.monotonic()
        with self.cond:
            device = self.devices.get(name)
            return None if device is None else device.public(now)

    def request(
        self,
        name: str,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
        timeout: float = 300,
    ) -> tuple[int, dict[str, str], bytes]:
        pending = Pending(
            id=uuid.uuid4().hex,
            message={
                "method": method,
                "path": path,
                "headers": dict(headers or {}),
                "body": base64.b64encode(body).decode(),
            },
        )
        with self.cond:
            device = self.devices.get(name)
            if device is None or not device.online(time.monotonic()):
                raise ConnectionError(f"device {name!r} is not connected")
            device.queue.append(pending)
            device.pending[pending.id] = pending
            self.cond.notify_all()
        if not pending.event.wait(timeout):
            with self.cond:
                if device.pending.pop(pending.id, None) is not None:
                    try:
                        device.queue.remove(pending)
                    except ValueError:
                        pass
            if pending.response is None and pending.error is None:
                raise TimeoutError(
                    f"device {name!r} did not answer {method} {path} within {timeout}s"
                )
        if pending.error is not None:
            raise ConnectionError(pending.error)
        assert pending.response is not None
        return pending.response

    # ----- agent side -----

    def poll(
        self,
        payload: dict[str, Any],
        alive: Callable[[], bool] | None = None,
    ) -> list[dict[str, Any]]:
        name = payload.get("name")
        instance = payload.get("instance")
        if not isinstance(name, str) or not NAME_RE.fullmatch(name):
            raise ValueError(
                "name must match [A-Za-z0-9][A-Za-z0-9_.-]{0,63}"
            )
        if not isinstance(instance, str) or not instance:
            raise ValueError("instance is required")
        roots = payload.get("roots") or ["/"]
        if not isinstance(roots, list) or not all(
            isinstance(root, str) and root.startswith("/") for root in roots
        ):
            raise ValueError("roots must be a list of absolute paths")
        env_name = payload.get("env")
        if env_name is not None and not isinstance(env_name, str):
            raise ValueError("env must be a string")
        wait = min(max(float(payload.get("wait", 25)), 0.0), MAX_POLL_WAIT)
        if self.name_check is not None:
            problem = self.name_check(name)
            if problem:
                raise ConflictError(problem)

        now = time.monotonic()
        with self.cond:
            device = self.devices.get(name)
            if device is not None and device.instance != instance:
                if device.online(now):
                    raise ConflictError(
                        f"name {name!r} is in use by {device.hostname} "
                        f"(pid {device.pid})"
                    )
                self._fail(device, "agent was replaced by a new instance")
                device = None
            if device is None:
                device = Device(
                    name=name,
                    instance=instance,
                    hostname="",
                    pid=None,
                    roots=[],
                    env=None,
                    connected_at=time.time(),
                    last_seen=now,
                    last_seen_wall=time.time(),
                )
                self.devices[name] = device
            device.hostname = str(payload.get("hostname", ""))
            pid = payload.get("pid")
            device.pid = pid if isinstance(pid, int) else None
            device.roots = list(roots)
            device.env = env_name
            device.active_polls += 1
            self._touch(device)
            try:
                deadline = time.monotonic() + wait
                while (
                    not device.queue
                    and not self.stopping
                    and self.devices.get(name) is device
                ):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self.cond.wait(min(remaining, ALIVE_CHECK_INTERVAL))
                    if alive is not None and not device.queue and not alive():
                        return []
                tasks = []
                while device.queue and len(tasks) < MAX_TASKS_PER_POLL:
                    pending = device.queue.popleft()
                    pending.dispatched = True
                    tasks.append({"id": pending.id, **pending.message})
                return tasks
            finally:
                device.active_polls -= 1
                self._touch(device)

    def undeliverable(self, name: str, task_ids: list[str], error: str) -> None:
        with self.cond:
            device = self.devices.get(name)
            if device is None:
                return
            for task_id in task_ids:
                pending = device.pending.pop(task_id, None)
                if pending is not None:
                    pending.error = error
                    pending.event.set()

    def result(self, payload: dict[str, Any]) -> bool:
        name = payload.get("name")
        instance = payload.get("instance")
        task_id = payload.get("id")
        status = int(payload.get("status", 500))
        headers = payload.get("headers") or {}
        if not isinstance(headers, dict):
            raise ValueError("headers must be an object")
        body = base64.b64decode(str(payload.get("body", "")))
        with self.cond:
            device = self.devices.get(str(name))
            if device is None or device.instance != instance:
                return False
            self._touch(device)
            pending = device.pending.pop(str(task_id), None)
            if pending is None:
                return False
            pending.response = (
                status,
                {str(k): str(v) for k, v in headers.items()},
                body,
            )
            pending.event.set()
            return True

    # ----- internals -----

    @staticmethod
    def _touch(device: Device) -> None:
        device.last_seen = time.monotonic()
        device.last_seen_wall = time.time()

    @staticmethod
    def _fail(device: Device, error: str) -> None:
        for pending in device.pending.values():
            pending.error = f"device {device.name!r}: {error}"
            pending.event.set()
        device.pending.clear()
        device.queue.clear()

    def _reap_forever(self) -> None:
        while not self.stopping:
            time.sleep(REAP_INTERVAL)
            now = time.monotonic()
            with self.cond:
                for name, device in list(self.devices.items()):
                    if device.online(now):
                        continue
                    if device.pending:
                        self._fail(device, "agent disconnected")
                    if now - device.last_seen > FORGET_AFTER:
                        del self.devices[name]

    def serve(self, host: str, port: int) -> None:
        """Start the agent-facing HTTP listener in background threads."""
        self.httpd = RendezvousServer((host, port), RendezvousHandler, self)
        threading.Thread(
            target=self.httpd.serve_forever,
            name="rendezvous-http",
            daemon=True,
        ).start()
        threading.Thread(
            target=self._reap_forever,
            name="rendezvous-reaper",
            daemon=True,
        ).start()

    def shutdown(self) -> None:
        with self.cond:
            self.stopping = True
            self.cond.notify_all()
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()


class RendezvousServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128

    def __init__(self, address, handler, rendezvous: Rendezvous):
        self.address_family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET
        super().__init__(address, handler)
        self.rendezvous = rendezvous

    def handle_error(self, request, client_address) -> None:
        # Agents that vanish mid-response are expected; stay quiet.
        pass

    def server_bind(self) -> None:
        if self.address_family == socket.AF_INET6:
            # Accept IPv4 clients too when listening on "::".
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


class RendezvousHandler(BaseHTTPRequestHandler):
    server: RendezvousServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        pass

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ValueError("invalid JSON body size")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def _client_alive(self) -> bool:
        """False once the agent has closed its end of the poll connection."""
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            if not readable:
                return True
            return bool(self.connection.recv(1, socket.MSG_PEEK))
        except OSError:
            return False

    def do_GET(self) -> None:
        if self.path == "/v1/health":
            self._json(HTTPStatus.OK, {"ok": True})
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "endpoint not found"})

    def do_POST(self) -> None:
        rendezvous = self.server.rendezvous
        try:
            if self.path == "/v1/agents/poll":
                payload = self._read_json()
                tasks = rendezvous.poll(payload, alive=self._client_alive)
                try:
                    self._json(HTTPStatus.OK, {"tasks": tasks})
                except OSError:
                    rendezvous.undeliverable(
                        str(payload.get("name")),
                        [task["id"] for task in tasks],
                        "request could not be delivered to the agent",
                    )
                    raise
                return
            if self.path == "/v1/agents/result":
                accepted = rendezvous.result(self._read_json())
                self._json(HTTPStatus.OK, {"accepted": accepted})
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "endpoint not found"})
        except ConflictError as exc:
            self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
        except (ValueError, TypeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
