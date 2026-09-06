#!/usr/bin/env python3
"""Expose heterogeneous remote machines through one local MCP server."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import shlex
import ssl
import subprocess
import tempfile
import threading
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from mcp.server.fastmcp import FastMCP


DEFAULT_CONFIG = Path(
    os.environ.get("REMOTE_FLEET_CONFIG", Path(__file__).with_name("hosts.toml"))
).expanduser()
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
TRANSFER_CHUNK_BYTES = 8 * 1024 * 1024
TRANSFER_REQUEST_TIMEOUT = 120
TRANSFER_METADATA_TIMEOUT = 3600


@dataclass(frozen=True)
class HostConfig:
    name: str
    backend: str
    ssh_alias: str
    proxy: str | None
    root: str
    ssh_archive: Path | None = None
    hostname: str | None = None
    user: str | None = None
    port: int | None = None
    identity_file: Path | None = None
    known_hosts_file: Path | None = None
    url: str | None = None
    insecure: bool = False
    network_env: dict[str, str] = field(default_factory=dict)


def load_hosts(path: Path) -> dict[str, HostConfig]:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    hosts: dict[str, HostConfig] = {}
    for name, item in raw.get("hosts", {}).items():
        backend = item.get("backend", "ssh_archive")
        if backend not in {"ssh", "ssh_archive", "agent"}:
            raise ValueError(f"host {name!r}: unsupported backend {backend!r}")
        if backend == "ssh_archive" and not item.get("ssh_archive"):
            raise ValueError(f"host {name!r}: ssh_archive is required")
        if backend == "ssh" and not item.get("hostname"):
            raise ValueError(f"host {name!r}: hostname is required")
        if backend == "ssh" and not item.get("identity_file"):
            raise ValueError(f"host {name!r}: identity_file is required")
        if backend == "agent" and not item.get("url"):
            raise ValueError(f"host {name!r}: url is required")
        if backend == "agent":
            parsed_url = urllib.parse.urlparse(item["url"])
            if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
                raise ValueError(
                    f"host {name!r}: url must be an absolute HTTP(S) URL"
                )
        root = item.get("root", "/")
        if not root.startswith("/"):
            raise ValueError(f"host {name!r}: root must be absolute")
        hosts[name] = HostConfig(
            name=name,
            backend=backend,
            ssh_alias=item.get("ssh_alias", name),
            proxy=item.get("proxy"),
            root=root,
            ssh_archive=(
                Path(item["ssh_archive"]).expanduser()
                if item.get("ssh_archive")
                else None
            ),
            hostname=item.get("hostname"),
            user=item.get("user"),
            port=int(item["port"]) if item.get("port") is not None else None,
            identity_file=(
                Path(item["identity_file"]).expanduser()
                if item.get("identity_file")
                else None
            ),
            known_hosts_file=(
                Path(item["known_hosts_file"]).expanduser()
                if item.get("known_hosts_file")
                else None
            ),
            url=item.get("url"),
            insecure=bool(item.get("insecure", False)),
            network_env={str(k): str(v) for k, v in item.get("network_env", {}).items()},
        )
    if not hosts:
        raise ValueError(f"no hosts configured in {path}")
    return hosts


class SSHArchiveSession:
    def __init__(self, host: HostConfig):
        self.host = host
        if host.ssh_archive is None:
            raise ValueError(f"host {host.name!r}: ssh_archive is required")
        self._temp = tempfile.TemporaryDirectory(prefix=f"remote-fleet-{host.name}-")
        root = Path(self._temp.name)
        with zipfile.ZipFile(host.ssh_archive) as archive:
            archive.extractall(root)
        configs = sorted(root.rglob("config"), key=lambda p: (p.parent.name != ".ssh", len(p.parts)))
        if not configs:
            raise ValueError(f"{host.ssh_archive}: SSH config not found")
        self.config = configs[0]
        self.config.chmod(0o600)
        self.keys = self._resolve_keys(root)
        if not self.keys:
            raise ValueError(f"{host.ssh_archive}: no private key found")
        known_hosts = list(root.rglob("known_hosts"))
        self.known_hosts = known_hosts[0] if known_hosts else root / "known_hosts"
        if not self.known_hosts.exists():
            self.known_hosts.touch(mode=0o600)

    def _resolve_keys(self, root: Path) -> list[Path]:
        candidates: list[Path] = []
        for line in self.config.read_text(errors="replace").splitlines():
            parts = shlex.split(line, comments=True)
            if len(parts) >= 2 and parts[0].lower() == "identityfile":
                configured = Path(parts[1])
                if not configured.is_absolute() and not parts[1].startswith("~"):
                    candidates.append(self.config.parent / configured)
                else:
                    candidates.append(Path(os.path.expanduser(parts[1])))
                candidates.extend(root.rglob(configured.name))
        candidates.extend(
            path
            for path in root.rglob("id_*")
            if not path.name.endswith(".pub")
        )
        result: list[Path] = []
        for path in candidates:
            if path.is_file() and not path.name.endswith(".pub") and path not in result:
                path.chmod(0o600)
                result.append(path)
        return result

    def _argv(self) -> list[str]:
        argv = [
            "ssh",
            "-F",
            str(self.config),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            f"UserKnownHostsFile={self.known_hosts}",
            "-o",
            "StrictHostKeyChecking=accept-new",
        ]
        if self.host.proxy:
            argv += [
                "-o",
                f"ProxyCommand=nc -X connect -x {self.host.proxy} %h %p",
            ]
        for key in self.keys:
            argv += ["-i", str(key)]
        return argv + [self.host.ssh_alias]

    def run(
        self,
        command: str,
        *,
        timeout: float | None = 300,
        input_data: bytes | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            self._argv() + [command],
            input=input_data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )


class SSHSession:
    def __init__(self, host: HostConfig):
        self.host = host
        if not host.hostname or not host.identity_file:
            raise ValueError(
                f"host {host.name!r}: hostname and identity_file are required"
            )
        if not host.identity_file.is_file():
            raise ValueError(f"identity file not found: {host.identity_file}")
        if host.known_hosts_file and not host.known_hosts_file.is_file():
            raise ValueError(
                f"known hosts file not found: {host.known_hosts_file}"
            )

    def _argv(self) -> list[str]:
        argv = [
            "ssh",
            "-F",
            "/dev/null",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            "-o",
            "IdentitiesOnly=yes",
            "-i",
            str(self.host.identity_file),
        ]
        if self.host.known_hosts_file:
            argv += [
                "-o",
                f"UserKnownHostsFile={self.host.known_hosts_file}",
                "-o",
                "StrictHostKeyChecking=yes",
            ]
        else:
            argv += ["-o", "StrictHostKeyChecking=accept-new"]
        if self.host.proxy:
            argv += [
                "-o",
                f"ProxyCommand=nc -X connect -x {self.host.proxy} %h %p",
            ]
        if self.host.port:
            argv += ["-p", str(self.host.port)]
        destination = self.host.hostname
        if self.host.user:
            destination = f"{self.host.user}@{destination}"
        return argv + [destination]

    def run(
        self,
        command: str,
        *,
        timeout: float | None = 300,
        input_data: bytes | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            self._argv() + [command],
            input=input_data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )


class AgentSession:
    def __init__(self, host: HostConfig):
        self.host = host
        if not host.url:
            raise ValueError(f"host {host.name!r}: url is required")
        self.url = host.url.rstrip("/")
        self.context = ssl._create_unverified_context() if host.insecure else None
        proxy_url = host.proxy
        if proxy_url and "://" not in proxy_url:
            proxy_url = "http://" + proxy_url
        handlers: list[Any] = [
            urllib.request.ProxyHandler(
                {"http": proxy_url, "https": proxy_url} if proxy_url else {}
            )
        ]
        if self.context:
            handlers.append(urllib.request.HTTPSHandler(context=self.context))
        self.opener = urllib.request.build_opener(*handlers)

    def request(
        self,
        method: str,
        path: str,
        *,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 300,
    ):
        request = urllib.request.Request(
            self.url + path,
            data=data,
            headers=headers or {},
            method=method,
        )
        return self.opener.open(request, timeout=timeout)

    def json_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float = 300,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"} if data is not None else None
        with self.request(method, path, data=data, headers=headers, timeout=timeout) as response:
            value = json.load(response)
        if not isinstance(value, dict):
            raise ValueError(f"agent {self.host.name!r}: response must be a JSON object")
        return value

    @staticmethod
    def _failed_process(exc: Exception) -> subprocess.CompletedProcess[bytes]:
        if isinstance(exc, urllib.error.HTTPError):
            stderr = f"HTTP {exc.code}: ".encode() + exc.read()
            return subprocess.CompletedProcess([], 1, b"", stderr)
        return subprocess.CompletedProcess([], 255, b"", str(exc).encode())

    def health(self) -> dict[str, Any]:
        return self.json_request("GET", "/v1/health", timeout=30)

    def run(
        self,
        command: str,
        *,
        cwd: str,
        timeout: float | None = 300,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        payload = {
            "command": command,
            "cwd": cwd,
            "timeout": timeout,
            "env": env or {},
        }
        try:
            value = self.json_request(
                "POST", "/v1/exec", payload, timeout=timeout + 5
            )
        except urllib.error.HTTPError as exc:
            if exc.code == 408:
                try:
                    value = json.loads(exc.read())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    value = {}
                raise subprocess.TimeoutExpired(
                    command,
                    timeout,
                    output=str(value.get("stdout", "")).encode(),
                    stderr=str(value.get("stderr", "")).encode(),
                ) from None
            return self._failed_process(exc)
        except (OSError, ValueError) as exc:
            return self._failed_process(exc)
        return subprocess.CompletedProcess(
            [],
            int(value.get("exit_code", 1)),
            str(value.get("stdout", "")).encode(),
            str(value.get("stderr", "")).encode(),
        )

    def start_job(
        self,
        command: str,
        *,
        cwd: str,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return self.json_request(
            "POST",
            "/v1/jobs",
            {"command": command, "cwd": cwd, "env": env or {}},
            timeout=30,
        )

    def job_info(self, job_id: str) -> dict[str, Any]:
        return self.json_request("GET", f"/v1/jobs/{job_id}", timeout=30)

    def job_logs(self, job_id: str, offset: int, limit: int) -> dict[str, Any]:
        query = urllib.parse.urlencode({"offset": offset, "limit": limit})
        return self.json_request(
            "GET", f"/v1/jobs/{job_id}/logs?{query}", timeout=30
        )

    def stop_job(self, job_id: str) -> dict[str, Any]:
        return self.json_request(
            "POST", f"/v1/jobs/{job_id}/stop", {}, timeout=30
        )

    def download(
        self, remote_path: str, *, timeout: float | None = None
    ) -> tuple[bytes, str | None]:
        query = urllib.parse.urlencode({"path": remote_path})
        with self.request("GET", f"/v1/files?{query}", timeout=timeout) as response:
            return response.read(), response.headers.get("X-Content-SHA256")

    def upload(
        self, remote_path: str, data: bytes, *, timeout: float | None = None
    ) -> dict[str, Any]:
        query = urllib.parse.urlencode({"path": remote_path})
        digest = hashlib.sha256(data).hexdigest()
        with self.request(
            "PUT",
            f"/v1/files?{query}",
            data=data,
            headers={"X-Content-SHA256": digest},
            timeout=timeout,
        ) as response:
            value = json.load(response)
        if not isinstance(value, dict):
            raise ValueError(f"agent {self.host.name!r}: response must be a JSON object")
        return value

    def file_info(self, remote_path: str) -> dict[str, Any]:
        query = urllib.parse.urlencode({"path": remote_path})
        return self.json_request("GET", f"/v1/files/info?{query}", timeout=30)

    def download_chunk(self, remote_path: str, offset: int, limit: int) -> bytes:
        query = urllib.parse.urlencode(
            {"path": remote_path, "offset": offset, "limit": limit}
        )
        with self.request(
            "GET",
            f"/v1/files?{query}",
            timeout=TRANSFER_REQUEST_TIMEOUT,
        ) as response:
            return response.read()

    def upload_chunk(self, remote_path: str, offset: int, data: bytes) -> dict[str, Any]:
        query = urllib.parse.urlencode({"path": remote_path, "offset": offset})
        digest = hashlib.sha256(data).hexdigest()
        with self.request(
            "PUT",
            f"/v1/files/chunk?{query}",
            data=data,
            headers={"X-Content-SHA256": digest},
            timeout=TRANSFER_REQUEST_TIMEOUT,
        ) as response:
            value = json.load(response)
        if not isinstance(value, dict):
            raise ValueError(f"agent {self.host.name!r}: response must be a JSON object")
        return value

    def finalize_upload(
        self,
        partial_path: str,
        remote_path: str,
        size: int,
        sha256: str,
        overwrite: bool,
    ) -> dict[str, Any]:
        return self.json_request(
            "POST",
            "/v1/files/finalize",
            {
                "partial_path": partial_path,
                "path": remote_path,
                "size": size,
                "sha256": sha256,
                "overwrite": overwrite,
            },
            timeout=TRANSFER_METADATA_TIMEOUT,
        )


class RemoteFleet:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._config_lock = threading.RLock()
        self.hosts = load_hosts(config_path)
        self.sessions: dict[str, SSHArchiveSession | SSHSession | AgentSession] = {}
        self._config_signature = self._read_config_signature()

    def _read_config_signature(self) -> tuple[int, int, int, int]:
        stat = self.config_path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def reload_if_changed(self) -> bool:
        signature = self._read_config_signature()
        if signature == self._config_signature:
            return False
        with self._config_lock:
            signature = self._read_config_signature()
            if signature == self._config_signature:
                return False
            hosts = load_hosts(self.config_path)
            self.hosts = hosts
            self.sessions.clear()
            self._config_signature = signature
            return True

    def configured_hosts(self) -> tuple[HostConfig, ...]:
        self.reload_if_changed()
        with self._config_lock:
            return tuple(self.hosts.values())

    def host(self, name: str) -> HostConfig:
        self.reload_if_changed()
        with self._config_lock:
            if name not in self.hosts:
                raise ValueError(
                    f"unknown host {name!r}; configured: {', '.join(self.hosts)}"
                )
            return self.hosts[name]

    def session(self, name: str) -> SSHArchiveSession | SSHSession | AgentSession:
        self.reload_if_changed()
        with self._config_lock:
            if name not in self.hosts:
                raise ValueError(
                    f"unknown host {name!r}; configured: {', '.join(self.hosts)}"
                )
            host = self.hosts[name]
            if name not in self.sessions:
                if host.backend == "ssh_archive":
                    self.sessions[name] = SSHArchiveSession(host)
                elif host.backend == "agent":
                    self.sessions[name] = AgentSession(host)
                else:
                    self.sessions[name] = SSHSession(host)
            return self.sessions[name]

    def shell_command(
        self,
        host: HostConfig,
        command: str,
        cwd: str | None,
        network: bool,
    ) -> str:
        workdir = cwd or host.root
        prefix = ""
        if network:
            prefix = "".join(
                f"export {key}={shlex.quote(value)}; "
                for key, value in host.network_env.items()
            )
        script = f"cd -- {shlex.quote(workdir)} && {prefix}{command}"
        return "bash -lc " + shlex.quote(script)

    def run(
        self,
        name: str,
        command: str,
        cwd: str | None = None,
        timeout: float | None = 300,
        network: bool = False,
        input_data: bytes | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        session = self.session(name)
        if isinstance(session, AgentSession):
            if input_data is not None:
                raise ValueError("agent exec does not support stdin")
            return session.run(
                command,
                cwd=cwd or session.host.root,
                timeout=timeout,
                env=session.host.network_env if network else None,
            )
        return session.run(
            self.shell_command(session.host, command, cwd, network),
            timeout=timeout,
            input_data=input_data,
        )

    def validate_transfer_path(self, host: HostConfig, value: str) -> PurePosixPath:
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("remote path must be absolute and cannot contain '..'")
        root = PurePosixPath(host.root)
        if path != root and root not in path.parents:
            raise ValueError(f"remote path is outside root: {path}")
        return path


def decoded(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def result_of(proc: subprocess.CompletedProcess[bytes], elapsed: float | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exit_code": proc.returncode,
        "stdout": decoded(proc.stdout),
        "stderr": decoded(proc.stderr),
    }
    if elapsed is not None:
        result["duration_seconds"] = round(elapsed, 3)
    return result


def parse_job_ref(fleet: RemoteFleet, job_ref: str) -> tuple[HostConfig, str]:
    try:
        host_name, job_id = job_ref.split(":", 1)
    except ValueError as exc:
        raise ValueError("job_ref must have the form host:job_id") from exc
    host = fleet.host(host_name)
    if not JOB_ID_RE.fullmatch(job_id):
        raise ValueError("invalid job_ref")
    return host, job_id


def job_paths(host: HostConfig, job_id: str) -> tuple[str, str, str]:
    base = f"$HOME/.local/state/remote-fleet/jobs/{host.name}/{job_id}"
    return f"{base}.pid", f"{base}.log", f"{base}.exit"


def sha256_path(path: Path, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = limit
    with path.open("rb") as stream:
        while remaining is None or remaining > 0:
            read_size = TRANSFER_CHUNK_BYTES
            if remaining is not None:
                read_size = min(read_size, remaining)
            chunk = stream.read(read_size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    if remaining not in (None, 0):
        raise ValueError(f"file ended before {limit} bytes: {path}")
    return digest.hexdigest()


class TransferManager:
    def __init__(self, fleet: RemoteFleet):
        self.fleet = fleet
        self.state_dir = Path(
            os.environ.get(
                "REMOTE_FLEET_TRANSFER_STATE_DIR",
                "~/.local/state/remote-fleet/transfers",
            )
        ).expanduser()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.states: dict[str, dict[str, Any]] = {}
        self.cancel_events: dict[str, threading.Event] = {}
        self.active_targets: dict[str, str] = {}

    def _state_path(self, transfer_id: str) -> Path:
        return self.state_dir / f"{transfer_id}.json"

    def _save(self, transfer_id: str) -> None:
        with self.lock:
            payload = dict(self.states[transfer_id])
        path = self._state_path(transfer_id)
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, path)

    def _update(self, transfer_id: str, **fields: Any) -> None:
        with self.lock:
            self.states[transfer_id].update(fields)
            self.states[transfer_id]["updated_at"] = time.time()
        self._save(transfer_id)

    def _new(
        self,
        host: str,
        direction: str,
        source: str,
        destination: str,
        partial_path: str,
        expected_sha256: str | None,
        overwrite: bool,
        target_key: str,
    ) -> tuple[str, threading.Event]:
        transfer_id = uuid.uuid4().hex
        event = threading.Event()
        now = time.time()
        state = {
            "transfer_ref": f"{host}:{transfer_id}",
            "host": host,
            "direction": direction,
            "state": "queued",
            "source": source,
            "destination": destination,
            "partial_path": partial_path,
            "bytes_transferred": 0,
            "total_bytes": None,
            "sha256": None,
            "expected_sha256": expected_sha256,
            "overwrite": overwrite,
            "started_at": now,
            "updated_at": now,
        }
        with self.lock:
            active = self.active_targets.get(target_key)
            if active is not None:
                raise ValueError(
                    "another transfer is already writing this destination: "
                    f"{host}:{active}"
                )
            self.states[transfer_id] = state
            self.cancel_events[transfer_id] = event
            self.active_targets[target_key] = transfer_id
        try:
            self._save(transfer_id)
        except Exception:
            with self.lock:
                self.states.pop(transfer_id, None)
                self.cancel_events.pop(transfer_id, None)
                if self.active_targets.get(target_key) == transfer_id:
                    del self.active_targets[target_key]
            raise
        return transfer_id, event

    def _start(self, transfer_id: str, target_key: str, worker) -> None:
        def run() -> None:
            try:
                self._update(transfer_id, state="running")
                worker()
            except InterruptedError as exc:
                self._update(transfer_id, state="cancelled", error=str(exc))
            except Exception as exc:
                self._update(
                    transfer_id,
                    state="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            finally:
                with self.lock:
                    if self.active_targets.get(target_key) == transfer_id:
                        del self.active_targets[target_key]
                    self.cancel_events.pop(transfer_id, None)

        threading.Thread(
            target=run,
            name=f"remote-fleet-transfer-{transfer_id[:8]}",
            daemon=True,
        ).start()

    @staticmethod
    def _validate_sha256(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", normalized):
            raise ValueError("expected_sha256 must contain 64 hexadecimal characters")
        return normalized

    @staticmethod
    def _remote_python(
        session: SSHArchiveSession | SSHSession,
        source: str,
        data: bytes | None = None,
        timeout: float = TRANSFER_REQUEST_TIMEOUT,
    ) -> bytes:
        proc = session.run(
            "python3 -c " + shlex.quote(source),
            timeout=timeout,
            input_data=data,
        )
        if proc.returncode != 0:
            raise RuntimeError(decoded(proc.stderr) or decoded(proc.stdout))
        return proc.stdout

    def _remote_info(
        self,
        session: SSHArchiveSession | SSHSession | AgentSession,
        path: str,
        hash_length: int | None = None,
    ) -> dict[str, Any]:
        if isinstance(session, AgentSession):
            query = urllib.parse.urlencode(
                {
                    "path": path,
                    **({"hash_length": hash_length} if hash_length is not None else {}),
                }
            )
            return session.json_request(
                "GET",
                f"/v1/files/info?{query}",
                timeout=TRANSFER_METADATA_TIMEOUT,
            )
        code = f"""import hashlib
import json
import os
p = {path!r}
n = {hash_length!r}
exists = os.path.isfile(p)
size = os.path.getsize(p) if exists else 0
digest = hashlib.sha256()
remaining = n
if exists:
    with open(p, "rb") as stream:
        while remaining is None or remaining > 0:
            chunk = stream.read(8388608 if remaining is None else min(8388608, remaining))
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
if exists and remaining not in (None, 0):
    raise ValueError("file ended before requested hash_length")
print(json.dumps({{"exists": exists, "size": size, "sha256": digest.hexdigest() if exists else None}}))
"""
        return json.loads(
            decoded(
                self._remote_python(
                    session,
                    code,
                    timeout=TRANSFER_METADATA_TIMEOUT,
                )
            )
        )

    def _download_chunk(
        self,
        session: SSHArchiveSession | SSHSession | AgentSession,
        path: str,
        offset: int,
        limit: int,
    ) -> bytes:
        if isinstance(session, AgentSession):
            return session.download_chunk(path, offset, limit)
        code = (
            "import os,sys; "
            f"p={path!r}; o={offset}; n={limit}; "
            "f=open(p,'rb'); f.seek(o); d=f.read(n); f.close(); sys.stdout.buffer.write(d)"
        )
        return self._remote_python(session, code)

    def _append_remote(
        self,
        session: SSHArchiveSession | SSHSession | AgentSession,
        path: str,
        offset: int,
        data: bytes,
    ) -> None:
        if isinstance(session, AgentSession):
            value = session.upload_chunk(path, offset, data)
            if int(value.get("size", -1)) != offset + len(data):
                raise RuntimeError("agent reported an unexpected partial upload size")
            return
        code = (
            "import fcntl,os,sys; "
            f"p={path!r}; o={offset}; "
            "os.makedirs(os.path.dirname(p),exist_ok=True); "
            "f=open(p,'a+b'); fcntl.flock(f.fileno(),fcntl.LOCK_EX); "
            "f.seek(0,os.SEEK_END); size=f.tell(); "
            "assert size==o, f'partial size mismatch: {size} != {o}'; "
            "d=sys.stdin.buffer.read(); f.write(d); f.flush(); os.fsync(f.fileno()); f.close(); "
            "print(os.path.getsize(p))"
        )
        result = self._remote_python(session, code, data)
        if int(decoded(result).strip()) != offset + len(data):
            raise RuntimeError("remote host reported an unexpected partial upload size")

    def start_download(
        self,
        host_name: str,
        remote_path: str,
        local_path: str,
        expected_sha256: str | None,
        overwrite: bool,
    ) -> dict[str, Any]:
        host = self.fleet.host(host_name)
        remote = str(self.fleet.validate_transfer_path(host, remote_path))
        destination = Path(local_path).expanduser().resolve()
        partial = Path(str(destination) + ".partial")
        expected = self._validate_sha256(expected_sha256)
        target_key = f"download:{destination}"
        transfer_id, event = self._new(
            host_name,
            "download",
            remote,
            str(destination),
            str(partial),
            expected,
            overwrite,
            target_key,
        )

        def worker() -> None:
            session = self.fleet.session(host_name)
            info = self._remote_info(session, remote)
            if not info.get("exists"):
                raise FileNotFoundError(remote)
            total = int(info["size"])
            remote_sha = str(info["sha256"]).lower()
            if expected and not hmac.compare_digest(expected, remote_sha):
                raise ValueError("remote SHA-256 does not match expected_sha256")
            self._update(transfer_id, total_bytes=total, sha256=remote_sha)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.is_file() and destination.stat().st_size == total and hmac.compare_digest(
                    sha256_path(destination), remote_sha
                ):
                    self._update(
                        transfer_id,
                        state="completed",
                        bytes_transferred=total,
                        reused=True,
                    )
                    return
                if not overwrite:
                    raise FileExistsError(f"destination already exists: {destination}")
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > total:
                raise ValueError("partial download is larger than the remote file")
            if offset:
                prefix = self._remote_info(session, remote, offset)
                if not hmac.compare_digest(sha256_path(partial), str(prefix["sha256"])):
                    raise ValueError("existing .partial content does not match remote prefix")
            self._update(transfer_id, bytes_transferred=offset, resumed=bool(offset))
            with partial.open("ab") as stream:
                while offset < total:
                    if event.is_set():
                        raise InterruptedError("transfer stopped; .partial file was preserved")
                    chunk = self._download_chunk(
                        session,
                        remote,
                        offset,
                        min(TRANSFER_CHUNK_BYTES, total - offset),
                    )
                    if not chunk:
                        raise EOFError("remote file ended before the advertised size")
                    stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
                    offset += len(chunk)
                    self._update(transfer_id, bytes_transferred=offset)
            actual = sha256_path(partial)
            if partial.stat().st_size != total:
                raise ValueError("download size mismatch")
            if not hmac.compare_digest(actual, remote_sha):
                raise ValueError("download SHA-256 mismatch")
            if destination.exists() and not overwrite:
                raise FileExistsError(f"destination appeared during transfer: {destination}")
            os.replace(partial, destination)
            self._update(
                transfer_id,
                state="completed",
                bytes_transferred=total,
                verified_size=True,
                verified_sha256=True,
            )

        with self.lock:
            result = dict(self.states[transfer_id])
        self._start(transfer_id, target_key, worker)
        return result

    def start_upload(
        self,
        host_name: str,
        local_path: str,
        remote_path: str,
        expected_sha256: str | None,
        overwrite: bool,
    ) -> dict[str, Any]:
        host = self.fleet.host(host_name)
        destination = str(self.fleet.validate_transfer_path(host, remote_path))
        partial = destination + ".partial"
        source = Path(local_path).expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"local source is not a file: {source}")
        expected = self._validate_sha256(expected_sha256)
        target_key = f"upload:{host_name}:{destination}"
        transfer_id, event = self._new(
            host_name,
            "upload",
            str(source),
            destination,
            partial,
            expected,
            overwrite,
            target_key,
        )

        def worker() -> None:
            total = source.stat().st_size
            local_sha = sha256_path(source)
            if expected and not hmac.compare_digest(expected, local_sha):
                raise ValueError("local SHA-256 does not match expected_sha256")
            self._update(transfer_id, total_bytes=total, sha256=local_sha)
            session = self.fleet.session(host_name)
            final_info = self._remote_info(session, destination)
            if final_info.get("exists"):
                if int(final_info["size"]) == total and hmac.compare_digest(
                    str(final_info["sha256"]), local_sha
                ):
                    self._update(
                        transfer_id,
                        state="completed",
                        bytes_transferred=total,
                        reused=True,
                    )
                    return
                if not overwrite:
                    raise FileExistsError(f"remote destination already exists: {destination}")
            partial_info = self._remote_info(session, partial)
            offset = int(partial_info["size"]) if partial_info.get("exists") else 0
            if offset > total:
                raise ValueError("remote .partial is larger than the local file")
            if offset and not hmac.compare_digest(
                sha256_path(source, offset), str(partial_info["sha256"])
            ):
                raise ValueError("remote .partial content does not match local prefix")
            self._update(transfer_id, bytes_transferred=offset, resumed=bool(offset))
            with source.open("rb") as stream:
                stream.seek(offset)
                while offset < total:
                    if event.is_set():
                        raise InterruptedError("transfer stopped; remote .partial was preserved")
                    chunk = stream.read(min(TRANSFER_CHUNK_BYTES, total - offset))
                    if not chunk:
                        raise EOFError("local source ended before its initial size")
                    self._append_remote(session, partial, offset, chunk)
                    offset += len(chunk)
                    self._update(transfer_id, bytes_transferred=offset)
            if source.stat().st_size != total or not hmac.compare_digest(
                sha256_path(source), local_sha
            ):
                raise ValueError("local source changed during upload")
            if isinstance(session, AgentSession):
                session.finalize_upload(
                    partial,
                    destination,
                    total,
                    local_sha,
                    overwrite,
                )
            else:
                code = f"""import hashlib
import os
p = {partial!r}
d = {destination!r}
size = {total}
expected = {local_sha!r}
overwrite = {overwrite!r}
assert os.path.isfile(p), "partial upload is missing"
assert os.path.getsize(p) == size, "upload size mismatch"
digest = hashlib.sha256()
with open(p, "rb") as stream:
    while chunk := stream.read(8388608):
        digest.update(chunk)
assert digest.hexdigest() == expected, "upload SHA-256 mismatch"
assert overwrite or not os.path.exists(d), f"destination already exists: {{d}}"
os.replace(p, d)
"""
                self._remote_python(
                    session,
                    code,
                    timeout=TRANSFER_METADATA_TIMEOUT,
                )
            self._update(
                transfer_id,
                state="completed",
                bytes_transferred=total,
                verified_size=True,
                verified_sha256=True,
            )

        with self.lock:
            result = dict(self.states[transfer_id])
        self._start(transfer_id, target_key, worker)
        return result

    def status(self, transfer_ref: str) -> dict[str, Any]:
        host_name, transfer_id = parse_job_ref(self.fleet, transfer_ref)
        with self.lock:
            state = self.states.get(transfer_id)
        restored = state is None
        if state is None:
            path = self._state_path(transfer_id)
            if not path.is_file():
                raise ValueError("unknown transfer_ref")
            state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("host") != host_name.name:
            raise ValueError("transfer_ref host does not match persisted transfer")
        result = dict(state)
        if restored and result.get("state") in {"queued", "running"}:
            result["state"] = "interrupted"
            result["error"] = (
                "the MCP process restarted; call resume_transfer with this "
                "transfer_ref to resume the preserved .partial file"
            )
        total = result.get("total_bytes")
        if isinstance(total, int) and total > 0:
            result["progress"] = round(result["bytes_transferred"] / total, 6)
        return result

    def stop(self, transfer_ref: str) -> dict[str, Any]:
        host, transfer_id = parse_job_ref(self.fleet, transfer_ref)
        with self.lock:
            event = self.cancel_events.get(transfer_id)
            state = self.states.get(transfer_id)
        if event is None or state is None or state.get("host") != host.name:
            raise ValueError("transfer is not active in this MCP process")
        if state.get("state") not in {"queued", "running"}:
            raise ValueError(f"transfer is already {state.get('state')}")
        event.set()
        return {"transfer_ref": transfer_ref, "stop_requested": True}

    def resume(self, transfer_ref: str) -> dict[str, Any]:
        previous = self.status(transfer_ref)
        if previous.get("state") in {"queued", "running"}:
            raise ValueError("transfer is still active")
        direction = previous.get("direction")
        expected = previous.get("expected_sha256")
        overwrite = bool(previous.get("overwrite", False))
        if direction == "download":
            return self.start_download(
                str(previous["host"]),
                str(previous["source"]),
                str(previous["destination"]),
                expected,
                overwrite,
            )
        if direction == "upload":
            return self.start_upload(
                str(previous["host"]),
                str(previous["source"]),
                str(previous["destination"]),
                expected,
                overwrite,
            )
        raise ValueError(f"unsupported persisted transfer direction: {direction!r}")


def create_server(
    config_path: Path,
    host: str = "127.0.0.1",
    port: int = 18766,
) -> tuple[FastMCP, RemoteFleet]:
    fleet = RemoteFleet(config_path)
    transfers = TransferManager(fleet)
    server = FastMCP(
        "remote_fleet",
        host=host,
        port=port,
        instructions=(
            "统一管理配置中的远程机器。先用 list_hosts/health 确认目标；短命令用 exec，"
            "长任务用 start_job 并保留完整 job_ref。文件传输是异步、可续传任务，"
            "使用 download_file/upload_file 启动后，以 transfer_status 查询。"
            "失败、中止或 MCP 重启后使用 resume_transfer 续传。"
            "文件传输限制在主机配置的 root 下。"
            "hosts.toml 变更会在下一次工具调用时自动加载。"
            "network=true 会注入该机器配置的网络代理环境。"
        ),
    )

    @server.tool()
    def list_hosts() -> list[dict[str, Any]]:
        """列出配置的远程机器、后端和根目录，不建立连接。"""
        return [
            {
                "host": host.name,
                "backend": host.backend,
                "root": host.root,
            }
            for host in fleet.configured_hosts()
        ]

    @server.tool()
    def health(host: str) -> dict[str, Any]:
        """验证远程机器是否可登录，并返回主机名。"""
        session = fleet.session(host)
        if isinstance(session, AgentSession):
            try:
                value = session.health()
            except (OSError, ValueError) as exc:
                proc = session._failed_process(exc)
                return {
                    "host": host,
                    "reachable": False,
                    **result_of(proc),
                }
            hostname = str(value.get("hostname", ""))
            return {
                "host": host,
                "reachable": bool(value.get("ok")),
                "exit_code": 0 if value.get("ok") else 1,
                "stdout": hostname + ("\n" if hostname else ""),
                "stderr": "",
            }
        proc = fleet.run(host, "hostname", cwd="/", timeout=30)
        return {"host": host, "reachable": proc.returncode == 0, **result_of(proc)}

    @server.tool()
    def inspect_host(host: str) -> dict[str, Any]:
        """只读检查主机、GPU、根目录和启动脚本是否存在。"""
        cfg = fleet.host(host)
        start_script = str(PurePosixPath(cfg.root) / "start.sh")
        script = (
            "printf 'hostname='; hostname; "
            "printf 'cwd='; pwd; "
            "printf 'start_script='; test -f " + shlex.quote(start_script) +
            " && echo present || echo missing; "
            "command -v nvidia-smi >/dev/null && "
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true"
        )
        proc = fleet.run(host, script, timeout=45)
        return {"host": host, **result_of(proc)}

    @server.tool(name="exec")
    def exec_command(
        host: str,
        command: str,
        cwd: str | None = None,
        timeout: int = 300,
        network: bool = False,
    ) -> dict[str, Any]:
        """执行有边界的短命令；timeout 单位为秒，network 控制是否注入代理。"""
        timeout = max(1, min(timeout, 3600))
        started = time.monotonic()
        try:
            proc = fleet.run(host, command, cwd, timeout, network)
            return result_of(proc, time.monotonic() - started)
        except subprocess.TimeoutExpired as exc:
            return {
                "timed_out": True,
                "timeout_seconds": timeout,
                "stdout": decoded(exc.stdout or b""),
                "stderr": decoded(exc.stderr or b""),
            }

    @server.tool()
    def start_job(
        host: str,
        command: str,
        cwd: str | None = None,
        network: bool = False,
    ) -> dict[str, Any]:
        """在远端启动长任务，返回供 status/logs/stop 使用的完整 job_ref。"""
        cfg = fleet.host(host)
        session = fleet.session(host)
        workdir = cwd or cfg.root
        if isinstance(session, AgentSession):
            value = session.start_job(
                command,
                cwd=workdir,
                env=cfg.network_env if network else None,
            )
            job_id = str(value["id"])
            if not JOB_ID_RE.fullmatch(job_id):
                raise ValueError(f"agent returned invalid job id: {job_id!r}")
            return {
                "started": True,
                "job_ref": f"{host}:{job_id}",
                "pid": value.get("pid"),
                "stderr": "",
            }
        job_id = uuid.uuid4().hex
        pid_path, log_path, exit_path = job_paths(cfg, job_id)
        exports = ""
        if network:
            exports = "".join(
                f"export {key}={shlex.quote(value)}; "
                for key, value in cfg.network_env.items()
            )
        inner = (
            f"cd -- {shlex.quote(workdir)} && {exports}"
            f"bash -lc {shlex.quote(command)}; rc=$?; "
            f"printf '%s\\n' \"$rc\" > {exit_path}; exit \"$rc\""
        )
        script = (
            f"mkdir -p $(dirname {pid_path}); "
            f"rm -f {exit_path}; "
            f"setsid bash -c {shlex.quote(inner)} > {log_path} 2>&1 < /dev/null & "
            f"pid=$!; printf '%s\\n' \"$pid\" > {pid_path}; printf '%s\\n' \"$pid\""
        )
        proc = session.run("bash -lc " + shlex.quote(script), timeout=30)
        if proc.returncode != 0:
            return {"started": False, **result_of(proc)}
        return {
            "started": True,
            "job_ref": f"{host}:{job_id}",
            "pid": decoded(proc.stdout).strip(),
            "log_path": log_path,
            "stderr": decoded(proc.stderr),
        }

    @server.tool()
    def job_status(job_ref: str) -> dict[str, Any]:
        """查询 start_job 返回的任务状态和退出码。"""
        host, job_id = parse_job_ref(fleet, job_ref)
        session = fleet.session(host.name)
        if isinstance(session, AgentSession):
            value = session.job_info(job_id)
            running = bool(value.get("running"))
            return {
                "job_ref": job_ref,
                "state": "running" if running else "completed",
                "exit_code": None if running else value.get("exit_code"),
                "stderr": "",
            }
        pid_path, _, exit_path = job_paths(host, job_id)
        script = (
            f"if test -f {exit_path}; then printf 'completed:'; cat {exit_path}; "
            f"elif test -f {pid_path} && kill -0 $(cat {pid_path}) 2>/dev/null; "
            "then echo running; else echo unknown; fi"
        )
        proc = session.run("bash -lc " + shlex.quote(script), timeout=30)
        value = decoded(proc.stdout).strip()
        state, _, code = value.partition(":")
        return {
            "job_ref": job_ref,
            "state": state or "unknown",
            "exit_code": int(code) if code.lstrip("-").isdigit() else None,
            "stderr": decoded(proc.stderr),
        }

    @server.tool()
    def job_logs(job_ref: str, offset: int = 0, limit: int = 262144) -> dict[str, Any]:
        """按字节偏移读取任务日志，并返回下一次应使用的 offset。"""
        host, job_id = parse_job_ref(fleet, job_ref)
        _, log_path, _ = job_paths(host, job_id)
        offset = max(0, offset)
        limit = max(1, min(limit, 1024 * 1024))
        session = fleet.session(host.name)
        if isinstance(session, AgentSession):
            value = session.job_logs(job_id, offset, limit)
            return {
                "job_ref": job_ref,
                "log": str(value.get("log", "")),
                "size": value.get("size", value.get("next_offset", offset)),
                "next_offset": value.get("next_offset", offset),
                "running": value.get("running"),
                "exit_code": value.get("exit_code"),
                "stderr": "",
            }
        py = (
            "import base64,json,os; p=os.path.expandvars(" + repr(log_path) + "); "
            f"o={offset}; n={limit}; "
            "d=b''; s=os.path.getsize(p) if os.path.exists(p) else 0; "
            "f=open(p,'rb') if os.path.exists(p) else None; "
            "f.seek(o) if f else None; d=f.read(n) if f else b''; f.close() if f else None; "
            "print(json.dumps({'data':base64.b64encode(d).decode(),'size':s,'next_offset':o+len(d)}))"
        )
        proc = session.run("python3 -c " + shlex.quote(py), timeout=30)
        if proc.returncode != 0:
            return {"job_ref": job_ref, **result_of(proc)}
        payload = json.loads(decoded(proc.stdout))
        return {
            "job_ref": job_ref,
            "log": base64.b64decode(payload.pop("data")).decode("utf-8", errors="replace"),
            **payload,
            "stderr": decoded(proc.stderr),
        }

    @server.tool()
    def stop_job(job_ref: str) -> dict[str, Any]:
        """终止远端任务的整个进程组。"""
        host, job_id = parse_job_ref(fleet, job_ref)
        session = fleet.session(host.name)
        if isinstance(session, AgentSession):
            value = session.stop_job(job_id)
            return {
                "job_ref": job_ref,
                "stopped": not bool(value.get("running")),
                "exit_code": value.get("exit_code"),
                "stdout": "",
                "stderr": "",
            }
        pid_path, _, exit_path = job_paths(host, job_id)
        script = (
            f"test -f {pid_path} || exit 3; pid=$(cat {pid_path}); "
            "kill -TERM -- -$pid 2>/dev/null || exit 4; "
            f"test -f {exit_path} || printf '143\\n' > {exit_path}"
        )
        proc = session.run("bash -lc " + shlex.quote(script), timeout=30)
        return {"job_ref": job_ref, "stopped": proc.returncode == 0, **result_of(proc)}

    @server.tool()
    def download_file(
        host: str,
        remote_path: str,
        local_path: str,
        expected_sha256: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """启动无总超时的异步下载；写入 .partial，可续传，校验后原子改名。"""
        return transfers.start_download(
            host,
            remote_path,
            local_path,
            expected_sha256,
            overwrite,
        )

    @server.tool()
    def upload_file(
        host: str,
        local_path: str,
        remote_path: str,
        expected_sha256: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """启动无总超时的异步上传；写入 .partial，可续传，校验后原子改名。"""
        return transfers.start_upload(
            host,
            local_path,
            remote_path,
            expected_sha256,
            overwrite,
        )

    @server.tool()
    def transfer_status(transfer_ref: str) -> dict[str, Any]:
        """查询异步上传或下载的进度、校验结果与最终状态。"""
        return transfers.status(transfer_ref)

    @server.tool()
    def stop_transfer(transfer_ref: str) -> dict[str, Any]:
        """请求停止传输；已写入的 .partial 文件会保留供下次续传。"""
        return transfers.stop(transfer_ref)

    @server.tool()
    def resume_transfer(transfer_ref: str) -> dict[str, Any]:
        """从原任务保留的 .partial 文件创建续传任务，并返回新的 transfer_ref。"""
        return transfers.resume(transfer_ref)

    return server, fleet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18766)
    parser.add_argument("--check-host")
    args = parser.parse_args()
    server, fleet = create_server(
        args.config.expanduser(),
        host=args.host,
        port=args.port,
    )
    if args.check_host:
        proc = fleet.run(
            args.check_host,
            "hostname",
            cwd=fleet.host(args.check_host).root,
            timeout=30,
        )
        print(json.dumps(result_of(proc), ensure_ascii=False, indent=2))
        raise SystemExit(proc.returncode)
    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
