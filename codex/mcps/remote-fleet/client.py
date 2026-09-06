#!/usr/bin/env python3
"""CLI client for agent.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class Client:
    def __init__(self, url: str, insecure: bool = False):
        self.url = url.rstrip("/")
        self.context = ssl._create_unverified_context() if insecure else None

    def request(
        self,
        method: str,
        path: str,
        data=None,
        headers=None,
        timeout: float | None = None,
    ):
        request_headers = dict(headers or {})
        req = urllib.request.Request(self.url + path, data=data, headers=request_headers, method=method)
        if timeout is None:
            timeout = 3 if method == "GET" and path == "/v1/health" else 86400
        try:
            return urllib.request.urlopen(req, timeout=timeout, context=self.context)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise SystemExit(f"HTTP {exc.code}: {body}") from None

    def json(self, method: str, path: str, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        with self.request(method, path, data, {"Content-Type": "application/json"}) as response:
            return json.load(response)


def add_command_args(parser: argparse.ArgumentParser, *, background: bool = False) -> None:
    parser.add_argument("command", nargs=argparse.REMAINDER)
    parser.add_argument("--cwd")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    if not background:
        parser.add_argument("--timeout", type=float, default=300)


def command_payload(args, *, timeout: bool) -> dict:
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("command is required; put command options after --")
    env = {}
    for item in args.env:
        if "=" not in item:
            raise SystemExit(f"invalid --env value: {item}")
        key, value = item.split("=", 1)
        env[key] = value
    command_text = command[0] if len(command) == 1 else shlex.join(command)
    payload = {"command": command_text, "env": env}
    if args.cwd:
        payload["cwd"] = args.cwd
    if timeout:
        payload["timeout"] = args.timeout
    return payload


def print_json(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def sha256_file(path: Path, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = limit
    with path.open("rb") as stream:
        while remaining is None or remaining > 0:
            size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            chunk = stream.read(size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    if remaining not in (None, 0):
        raise SystemExit(f"file ended before {limit} bytes: {path}")
    return digest.hexdigest()


def remote_info(client: Client, path: str, hash_length: int | None = None) -> dict:
    params = {"path": path}
    if hash_length is not None:
        params["hash_length"] = hash_length
    return client.json("GET", "/v1/files/info?" + urllib.parse.urlencode(params))


def upload_file(client: Client, source: Path, remote_path: str, chunk_size: int, overwrite: bool) -> dict:
    source = source.expanduser().resolve(strict=True)
    if not source.is_file():
        raise SystemExit(f"local source is not a file: {source}")
    total = source.stat().st_size
    expected = sha256_file(source)
    final = remote_info(client, remote_path)
    if final.get("exists"):
        if int(final["size"]) == total and final.get("sha256") == expected:
            return {"path": remote_path, "size": total, "sha256": expected, "reused": True}
        if not overwrite:
            raise SystemExit(f"remote destination already exists: {remote_path}")
    partial = remote_path + ".partial"
    info = remote_info(client, partial)
    offset = int(info["size"]) if info.get("exists") else 0
    if offset > total:
        raise SystemExit("remote .partial is larger than the local file")
    if offset and sha256_file(source, offset) != info.get("sha256"):
        raise SystemExit("remote .partial content does not match local source")
    with source.open("rb") as stream:
        stream.seek(offset)
        while offset < total:
            data = stream.read(min(chunk_size, total - offset))
            if not data:
                raise SystemExit("local source ended early")
            query = urllib.parse.urlencode({"path": partial, "offset": offset})
            digest = hashlib.sha256(data).hexdigest()
            with client.request(
                "PUT",
                f"/v1/files/chunk?{query}",
                data,
                {"X-Content-SHA256": digest},
                timeout=120,
            ) as response:
                value = json.load(response)
            offset = int(value["size"])
    if source.stat().st_size != total or sha256_file(source) != expected:
        raise SystemExit("local source changed during upload")
    return client.json(
        "POST",
        "/v1/files/finalize",
        {
            "partial_path": partial,
            "path": remote_path,
            "size": total,
            "sha256": expected,
            "overwrite": overwrite,
        },
    )


def download_file(client: Client, remote_path: str, destination: Path, chunk_size: int, overwrite: bool) -> dict:
    info = remote_info(client, remote_path)
    if not info.get("exists"):
        raise SystemExit(f"remote file does not exist: {remote_path}")
    total = int(info["size"])
    expected = str(info["sha256"])
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_file() and destination.stat().st_size == total and sha256_file(destination) == expected:
            return {"path": str(destination), "size": total, "sha256": expected, "reused": True}
        if not overwrite:
            raise SystemExit(f"local destination already exists: {destination}")
    partial = Path(str(destination) + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > total:
        raise SystemExit("local .partial is larger than the remote file")
    if offset:
        prefix = remote_info(client, remote_path, offset)
        if sha256_file(partial) != prefix.get("sha256"):
            raise SystemExit("local .partial content does not match remote source")
    with partial.open("ab") as stream:
        while offset < total:
            limit = min(chunk_size, total - offset)
            query = urllib.parse.urlencode(
                {"path": remote_path, "offset": offset, "limit": limit}
            )
            with client.request("GET", f"/v1/files?{query}", timeout=120) as response:
                data = response.read()
            if not data:
                raise SystemExit("remote source ended early")
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            offset += len(data)
    actual = sha256_file(partial)
    if partial.stat().st_size != total or actual != expected:
        raise SystemExit("download size or SHA-256 mismatch")
    if destination.exists() and not overwrite:
        raise SystemExit(f"local destination appeared during download: {destination}")
    os.replace(partial, destination)
    return {"path": str(destination), "size": total, "sha256": actual}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("DEV_AGENT_URL", "http://127.0.0.1:18765"))
    parser.add_argument("--insecure", action="store_true", help="disable TLS certificate verification")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("health")
    execute = sub.add_parser("exec")
    add_command_args(execute)
    start = sub.add_parser("start")
    add_command_args(start, background=True)
    for name in ("status", "stop"):
        item = sub.add_parser(name)
        item.add_argument("job_id")
    logs = sub.add_parser("logs")
    logs.add_argument("job_id")
    logs.add_argument("--offset", type=int, default=0)
    logs.add_argument("--follow", action="store_true")
    upload = sub.add_parser("upload")
    upload.add_argument("local_path")
    upload.add_argument("remote_path")
    upload.add_argument("--overwrite", action="store_true")
    upload.add_argument("--chunk-mib", type=int, default=8)
    download = sub.add_parser("download")
    download.add_argument("remote_path")
    download.add_argument("local_path")
    download.add_argument("--overwrite", action="store_true")
    download.add_argument("--chunk-mib", type=int, default=8)
    args = parser.parse_args()
    client = Client(args.url, args.insecure)

    if args.action == "health":
        print_json(client.json("GET", "/v1/health"))
    elif args.action == "exec":
        result = client.json("POST", "/v1/exec", command_payload(args, timeout=True))
        sys.stdout.write(result.get("stdout", ""))
        sys.stderr.write(result.get("stderr", ""))
        raise SystemExit(result.get("exit_code", 1))
    elif args.action == "start":
        print_json(client.json("POST", "/v1/jobs", command_payload(args, timeout=False)))
    elif args.action == "status":
        print_json(client.json("GET", f"/v1/jobs/{args.job_id}"))
    elif args.action == "stop":
        print_json(client.json("POST", f"/v1/jobs/{args.job_id}/stop", {}))
    elif args.action == "logs":
        offset = args.offset
        while True:
            value = client.json("GET", f"/v1/jobs/{args.job_id}/logs?offset={offset}")
            sys.stdout.write(value["log"])
            sys.stdout.flush()
            offset = value["next_offset"]
            if not args.follow or not value["running"]:
                break
            time.sleep(1)
    elif args.action == "upload":
        if args.chunk_mib < 1:
            raise SystemExit("--chunk-mib must be positive")
        print_json(
            upload_file(
                client,
                Path(args.local_path),
                args.remote_path,
                args.chunk_mib * 1024 * 1024,
                args.overwrite,
            )
        )
    elif args.action == "download":
        if args.chunk_mib < 1:
            raise SystemExit("--chunk-mib must be positive")
        print_json(
            download_file(
                client,
                args.remote_path,
                Path(args.local_path),
                args.chunk_mib * 1024 * 1024,
                args.overwrite,
            )
        )


if __name__ == "__main__":
    main()
