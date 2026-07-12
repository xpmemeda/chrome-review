#!/usr/bin/env python3
"""Test common LLM API endpoint compatibility.

The script calls these endpoints in order:
  1. /chat/completions
  2. /responses
  3. /messages

It accepts a base URL, model name, and API key. It uses only Python's
standard library so it can run on most development machines.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

DEFAULT_PROMPT = "Reply with exactly: pong"


@dataclass
class TestCase:
    name: str
    path: str
    payload: dict[str, Any]
    extra_headers: dict[str, str] | None = None


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.strip().rstrip("/")
    if not base_url:
        raise ValueError("base URL cannot be empty")
    return base_url


def join_url(base_url: str, path: str) -> str:
    base_url = normalize_base_url(base_url)
    path = path if path.startswith("/") else f"/{path}"
    return f"{base_url}{path}"


def compact_json(value: Any, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= limit:
        return text
    return text[:limit] + "... <truncated>"


def parse_json_or_text(body: bytes) -> Any:
    text = body.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def post_json(
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
    extra_headers: dict[str, str] | None,
) -> tuple[int | None, Any, float, str | None]:
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "llm-api-connectivity-test/1.0",
    }
    if extra_headers:
        headers.update(extra_headers)

    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            elapsed = time.monotonic() - started
            return response.status, parse_json_or_text(response.read()), elapsed, None
    except urllib.error.HTTPError as exc:
        elapsed = time.monotonic() - started
        return exc.code, parse_json_or_text(exc.read()), elapsed, None
    except Exception as exc:  # Network, TLS, DNS, timeout, proxy errors.
        elapsed = time.monotonic() - started
        return None, None, elapsed, f"{type(exc).__name__}: {exc}"


def build_tests(model: str, prompt: str, anthropic_version: str) -> list[TestCase]:
    return [
        TestCase(
            name="chat",
            path="/chat/completions",
            payload={
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": 32,
            },
        ),
        TestCase(
            name="responses",
            path="/responses",
            payload={
                "model": model,
                "input": prompt,
                "temperature": 0,
                "max_output_tokens": 32,
            },
        ),
        TestCase(
            name="messages",
            path="/messages",
            payload={
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": 32,
            },
            extra_headers={"anthropic-version": anthropic_version},
        ),
    ]


@dataclass
class TestResult:
    name: str
    url: str
    status: int | None
    elapsed: float
    ok: bool
    body: Any
    error: str | None


def print_intro(base_url: str, model: str, tests: list[TestCase], timeout: float) -> None:
    print("LLM API check")
    print(f"base_url : {base_url}")
    print(f"model    : {model}")
    print(f"timeout  : {timeout:g}s")
    print(f"endpoints: {', '.join(test.name for test in tests)}")
    print()


def print_table(results: list[TestResult]) -> None:
    headers = ["endpoint", "result", "http", "time"]
    rows = []
    for result in results:
        rows.append(
            [
                result.name,
                "PASS" if result.ok else "FAIL",
                str(result.status) if result.status is not None else "-",
                f"{result.elapsed:.2f}s",
            ]
        )

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def render(row: list[str]) -> str:
        return "  ".join(row[i].ljust(widths[i]) for i in range(len(row)))

    print(render(headers))
    print(render(["-" * width for width in widths]))
    for row in rows:
        print(render(row))


def print_details(results: list[TestResult], print_limit: int, verbose: bool) -> None:
    details = [result for result in results if verbose or not result.ok]
    if not details:
        return

    print()
    print("Details")
    for result in details:
        print(f"- {result.name}: {result.url}")
        if result.error:
            print(f"  error: {result.error}")
        elif result.body is not None:
            print(f"  body : {compact_json(result.body, print_limit)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test whether chat/responses/messages LLM APIs are reachable."
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Base URL, for example https://api.openai.com/v1 or https://api.anthropic.com/v1",
    )
    parser.add_argument("--model", required=True, help="Model name to request")
    parser.add_argument("--api-key", required=True, help="API key or bearer token")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Prompt used for every test. Default: {DEFAULT_PROMPT!r}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds. Default: 30",
    )
    parser.add_argument(
        "--anthropic-version",
        default="2023-06-01",
        help="Header value for /messages-compatible APIs. Default: 2023-06-01",
    )
    parser.add_argument(
        "--print-limit",
        type=int,
        default=300,
        help="Maximum response characters printed for details. Default: 300",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every response body.",
    )
    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Stop after the first endpoint that does not return 2xx.",
    )
    args = parser.parse_args()

    base_url = normalize_base_url(args.url)
    tests = build_tests(args.model, args.prompt, args.anthropic_version)
    failures = 0
    results: list[TestResult] = []
    print_intro(base_url, args.model, tests, args.timeout)

    for index, test in enumerate(tests, start=1):
        url = join_url(base_url, test.path)
        print(f"[{index}/{len(tests)}] {test.name:<9}", end=" ", flush=True)
        status, body, elapsed, error = post_json(
            url=url,
            api_key=args.api_key,
            payload=test.payload,
            timeout=args.timeout,
            extra_headers=test.extra_headers,
        )

        if error:
            failures += 1
            ok = False
        else:
            ok = status is not None and 200 <= status < 300
            if not ok:
                failures += 1

        print(f"{'PASS' if ok else 'FAIL'} http={status or '-'} time={elapsed:.2f}s")
        results.append(TestResult(test.name, url, status, elapsed, ok, body, error))
        if failures and args.stop_on_fail:
            break

    print()
    print_table(results)
    print_details(results, args.print_limit, args.verbose)
    print()
    if failures:
        print(
            f"Summary: {len(results) - failures}/{len(results)} passed, "
            f"{failures} failed."
        )
        return 1

    print(f"Summary: {len(results)}/{len(results)} passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
