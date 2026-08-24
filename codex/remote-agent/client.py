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

    def request(self, method: str, path: str, data=None, headers=None):
        request_headers = dict(headers or {})
        req = urllib.request.Request(self.url + path, data=data, headers=request_headers, method=method)
        try:
            return urllib.request.urlopen(req, timeout=86400, context=self.context)
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
    download = sub.add_parser("download")
    download.add_argument("remote_path")
    download.add_argument("local_path")
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
        source = Path(args.local_path)
        data = source.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        query = urllib.parse.urlencode({"path": args.remote_path})
        with client.request("PUT", f"/v1/files?{query}", data, {"X-Content-SHA256": digest}) as response:
            print_json(json.load(response))
    elif args.action == "download":
        query = urllib.parse.urlencode({"path": args.remote_path})
        with client.request("GET", f"/v1/files?{query}") as response:
            data = response.read()
            expected = response.headers.get("X-Content-SHA256")
        actual = hashlib.sha256(data).hexdigest()
        if expected and actual != expected:
            raise SystemExit("download SHA-256 mismatch")
        destination = Path(args.local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        print_json({"path": str(destination), "size": len(data), "sha256": actual})


if __name__ == "__main__":
    main()
