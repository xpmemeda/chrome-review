#!/usr/bin/env python3
"""Ask the user for confirmation over WeChat (OpenClaw ClawBot) and read the reply."""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import secrets
import sqlite3
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse


OPENCLAW_HOME = Path("~/.openclaw").expanduser()
CLAWBOT_HOME = Path("~/.local/state/wechat-clawbot").expanduser()
DEFAULT_STATE_DIR = Path("~/.local/state/wechat-confirm").expanduser()
CHANNEL = "openclaw-weixin"
MAX_WAIT_SECONDS = 1800
TOOL_THREADS = 64
QID_IN_QUOTE_RE = re.compile(r"\[Q(\d+)\]")
QID_PREFIX_RE = re.compile(r"^\s*[Qq](\d+)\s*[:：,，.。、\-\s]\s*", re.S)
BOT_PREFIX_RE = re.compile(r"^\s*(?:/bot|bot[:：])\s*", re.I)


@dataclass(frozen=True)
class Settings:
    account_id: str
    user_id: str
    openclaw_bin: Path
    node_bin_dir: Path
    state_dir: Path


def detect_account_id() -> str:
    accounts = json.loads((OPENCLAW_HOME / "openclaw-weixin" / "accounts.json").read_text())
    if len(accounts) != 1:
        raise SystemExit(f"expected exactly one weixin account, found {accounts}; pass --account-id")
    return accounts[0]


def detect_user_id(account_id: str) -> str:
    path = OPENCLAW_HOME / "openclaw-weixin" / "accounts" / f"{account_id}.context-tokens.json"
    users = list(json.loads(path.read_text())) if path.exists() else []
    if len(users) != 1:
        raise SystemExit(
            f"cannot infer the WeChat user from {path} ({len(users)} users); pass --user-id"
        )
    return users[0]


class QuestionStore:
    """Questions and answers persisted in SQLite; waiters block on a condition."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                options TEXT NOT NULL DEFAULT '[]',
                source TEXT,
                status TEXT NOT NULL,
                answer TEXT,
                chosen_option TEXT,
                send_message_id TEXT,
                send_error TEXT,
                created_at REAL NOT NULL,
                answered_at REAL
            )"""
        )
        self._cond = threading.Condition()

    def create(self, question: str, options: list[str], source: str | None) -> int:
        with self._cond:
            cur = self._db.execute(
                "INSERT INTO questions (question, options, source, status, created_at)"
                " VALUES (?, ?, ?, 'sending', ?)",
                (question, json.dumps(options, ensure_ascii=False), source, time.time()),
            )
            return int(cur.lastrowid)

    def mark_sent(self, qid: int, message_id: str | None) -> None:
        with self._cond:
            self._db.execute(
                "UPDATE questions SET status='pending', send_message_id=? WHERE id=?",
                (message_id, qid),
            )

    def mark_send_failed(self, qid: int, error: str) -> None:
        with self._cond:
            self._db.execute(
                "UPDATE questions SET status='send_failed', send_error=? WHERE id=?",
                (error, qid),
            )
            self._cond.notify_all()

    def get(self, qid: int) -> dict[str, Any] | None:
        with self._cond:
            row = self._db.execute("SELECT * FROM questions WHERE id=?", (qid,)).fetchone()
        return row_to_dict(row) if row else None

    def list(self, status: str | None, limit: int) -> list[dict[str, Any]]:
        with self._cond:
            if status:
                rows = self._db.execute(
                    "SELECT * FROM questions WHERE status=? ORDER BY id DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT * FROM questions ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
        return [row_to_dict(r) for r in rows]

    def pending_ids(self) -> list[int]:
        with self._cond:
            rows = self._db.execute(
                "SELECT id FROM questions WHERE status='pending' ORDER BY id"
            ).fetchall()
        return [int(r["id"]) for r in rows]

    def answer(self, qid: int, answer: str, chosen: str | None) -> bool:
        with self._cond:
            cur = self._db.execute(
                "UPDATE questions SET status='answered', answer=?, chosen_option=?, answered_at=?"
                " WHERE id=? AND status='pending'",
                (answer, chosen, time.time(), qid),
            )
            self._cond.notify_all()
            return cur.rowcount == 1

    def cancel(self, qid: int) -> bool:
        with self._cond:
            cur = self._db.execute(
                "UPDATE questions SET status='cancelled' WHERE id=? AND status IN ('pending','sending')",
                (qid,),
            )
            self._cond.notify_all()
            return cur.rowcount == 1

    def wait(self, qid: int, timeout: float) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                item = self.get(qid)
                if item is None or item["status"] not in ("pending", "sending"):
                    return item
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return item
                self._cond.wait(remaining)


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["question_id"] = f"Q{item.pop('id')}"
    item["options"] = json.loads(item["options"] or "[]")
    for key in ("created_at", "answered_at"):
        if item.get(key):
            item[key] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(item[key]))
    return {k: v for k, v in item.items() if v not in (None, "")}


def normalize_user(user_id: str) -> str:
    return str(user_id).rsplit(":", 1)[-1].split("@", 1)[0]


def parse_qid(question_id: str | int) -> int:
    match = re.fullmatch(r"\s*[Qq]?(\d+)\s*", str(question_id))
    if not match:
        raise ValueError(f"invalid question_id {question_id!r}; expected like 'Q3'")
    return int(match.group(1))


def resolve_option(answer: str, options: list[str]) -> str | None:
    text = answer.strip().rstrip(".。")
    if text.isdigit() and 1 <= int(text) <= len(options):
        return options[int(text) - 1]
    for option in options:
        if text == option.strip():
            return option
    return None


def format_question(qid: int, question: str, options: list[str], source: str | None) -> str:
    lines = [f"[Q{qid}] {question.strip()}"]
    if source:
        lines.append(f"（来自：{source}）")
    if options:
        lines.append("")
        lines.extend(f"{i}. {opt}" for i, opt in enumerate(options, 1))
        lines.append("")
        lines.append("回复序号或直接写答案。")
    lines.append(f"同时有多个问题时，引用本条回复或以 Q{qid} 开头。")
    return "\n".join(lines)


class WeChatSender:
    """Deliver through the running OpenClaw gateway so the channel context token is used."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()

    def send(self, text: str) -> str | None:
        params = {
            "channel": CHANNEL,
            "accountId": self.settings.account_id,
            "to": self.settings.user_id,
            "message": text,
            "idempotencyKey": f"wechat-confirm-{uuid.uuid4().hex}",
        }
        env = dict(os.environ)
        env["PATH"] = f"{self.settings.node_bin_dir}:{env.get('PATH', '')}"
        cmd = [
            str(self.settings.openclaw_bin),
            "gateway",
            "call",
            "send",
            "--json",
            "--timeout",
            "60000",
            "--params",
            json.dumps(params, ensure_ascii=False),
        ]
        with self._lock:
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=90)
        output = (proc.stdout or "").strip()
        try:
            result = json.loads(output[output.index("{") :]) if "{" in output else {}
        except ValueError:
            result = {}
        if proc.returncode != 0 or not result.get("messageId"):
            detail = (result.get("error") or {}).get("message") if result else None
            raise RuntimeError(detail or (proc.stderr or output or "send failed").strip()[-500:])
        return result.get("messageId")


class Bridge:
    def __init__(self, settings: Settings, store: QuestionStore, sender: WeChatSender):
        self.settings = settings
        self.store = store
        self.sender = sender
        secret_path = settings.state_dir / "bridge.secret"
        if not secret_path.exists():
            settings.state_dir.mkdir(parents=True, exist_ok=True)
            secret_path.write_text(secrets.token_hex(24))
            secret_path.chmod(0o600)
        self.secret = secret_path.read_text().strip()

    def ask(self, question: str, options: list[str], source: str | None) -> dict[str, Any]:
        qid = self.store.create(question, options, source)
        try:
            message_id = self.sender.send(format_question(qid, question, options, source))
        except Exception as exc:  # noqa: BLE001 - report delivery failure to the agent
            self.store.mark_send_failed(qid, str(exc))
        else:
            self.store.mark_sent(qid, message_id)
        return self.store.get(qid) or {}

    def handle_inbound(self, event: dict[str, Any]) -> dict[str, Any]:
        """Decide whether an inbound WeChat message answers a pending question."""
        if event.get("channel") != CHANNEL or event.get("isGroup"):
            return {"handled": False}
        if event.get("senderId") and normalize_user(event["senderId"]) != normalize_user(
            self.settings.user_id
        ):
            return {"handled": False}
        text = str(event.get("content") or event.get("body") or "").strip()
        pending = self.store.pending_ids()
        if not text or not pending or BOT_PREFIX_RE.match(text):
            return {"handled": False}

        qid: int | None = None
        quote = QID_IN_QUOTE_RE.search(str(event.get("replyToBody") or ""))
        if quote:
            qid = int(quote.group(1))
        else:
            prefix = QID_PREFIX_RE.match(text)
            if prefix:
                qid = int(prefix.group(1))
                text = text[prefix.end() :].strip() or text
            elif len(pending) == 1:
                qid = pending[0]
        if qid is None:
            listed = "、".join(f"Q{i}" for i in pending)
            return {
                "handled": True,
                "text": f"现在有多个待回答的问题：{listed}。请引用对应消息回复，或以问题编号开头，例如“Q{pending[0]} 好的”。",
            }

        item = self.store.get(qid)
        if item is None:
            return {"handled": True, "text": f"没有找到 Q{qid}。"}
        if qid not in pending:
            return {"handled": True, "text": f"Q{qid} 当前状态是 {item['status']}，不再接收回复。"}
        chosen = resolve_option(text, item.get("options", []))
        self.store.answer(qid, text, chosen)
        suffix = f"（选择：{chosen}）" if chosen else ""
        return {"handled": True, "text": f"已收到 Q{qid} 的回复{suffix}，已转给 agent。"}


def create_server(settings: Settings, host: str, port: int) -> FastMCP:
    store = QuestionStore(settings.state_dir / "questions.sqlite")
    sender = WeChatSender(settings)
    bridge = Bridge(settings, store, sender)
    server = FastMCP(
        "wechat_confirm",
        host=host,
        port=port,
        stateless_http=True,
        instructions=(
            "通过用户的微信 ClawBot 向用户发起确认并读取回复。"
            "需要用户决策或确认、而用户可能不在终端前时，用 ask_user 发问题（可带选项），"
            "再用 wait_reply 等待；超时返回 pending 时可以继续 wait_reply，问题不会丢。"
            "send_failed 表示微信没有送达，应改为在终端里询问。"
            "notify_user 只发通知、不等待回复。不要用它发送密钥等敏感信息。"
        ),
    )
    limiter: anyio.CapacityLimiter | None = None

    def tool(fn):
        @functools.wraps(fn)
        async def run_in_thread(*args, **kwargs):
            nonlocal limiter
            if limiter is None:
                limiter = anyio.CapacityLimiter(TOOL_THREADS)
            return await anyio.to_thread.run_sync(
                functools.partial(fn, *args, **kwargs), limiter=limiter
            )

        server.tool()(run_in_thread)
        return fn

    @tool
    def ask_user(
        question: str,
        options: list[str] | None = None,
        source: str | None = None,
        wait_seconds: int = 0,
    ) -> dict[str, Any]:
        """通过微信向用户提问，返回 question_id（如 Q3）。options 为可选的候选答案；source 标明提问方（如仓库或任务名）；wait_seconds>0 时发送后直接等待回复。"""
        item = bridge.ask(question, options or [], source)
        if wait_seconds > 0 and item.get("status") == "pending":
            item = store.wait(parse_qid(item["question_id"]), min(wait_seconds, MAX_WAIT_SECONDS)) or item
        return item

    @tool
    def wait_reply(question_id: str, timeout_seconds: int = 240) -> dict[str, Any]:
        """等待某个问题的回复；返回 status=answered 时 answer 为用户原文、chosen_option 为匹配到的选项；超时仍为 pending，可再次调用。"""
        qid = parse_qid(question_id)
        item = store.wait(qid, max(0, min(timeout_seconds, MAX_WAIT_SECONDS)))
        if item is None:
            raise ValueError(f"unknown question_id {question_id}")
        return item

    @tool
    def notify_user(message: str) -> dict[str, Any]:
        """通过微信给用户发一条通知，不等待回复。"""
        return {"sent": True, "message_id": sender.send(message)}

    @tool
    def cancel_question(question_id: str, notify: bool = False) -> dict[str, Any]:
        """取消一个待回答的问题；notify=true 时同时在微信里告知用户。"""
        qid = parse_qid(question_id)
        cancelled = store.cancel(qid)
        if cancelled and notify:
            sender.send(f"[Q{qid}] 已取消，无需回复。")
        return {"question_id": f"Q{qid}", "cancelled": cancelled}

    @tool
    def list_questions(status: str | None = "pending", limit: int = 20) -> list[dict[str, Any]]:
        """列出问题；status 可为 pending/answered/cancelled/send_failed，传空列出全部。"""
        return store.list(status or None, max(1, min(limit, 200)))

    @server.custom_route("/bridge/inbound", methods=["POST"])
    async def inbound(request: Request) -> JSONResponse:
        if not secrets.compare_digest(request.headers.get("x-bridge-secret", ""), bridge.secret):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        event = await request.json()
        result = await anyio.to_thread.run_sync(bridge.handle_inbound, event)
        return JSONResponse(result)

    @server.custom_route("/healthz", methods=["GET"])
    async def healthz(_request: Request) -> JSONResponse:
        return JSONResponse({"ok": True, "pending": store.pending_ids()})

    return server


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18767)
    parser.add_argument("--account-id", help="ClawBot weixin account; auto-detected if unique")
    parser.add_argument("--user-id", help="WeChat user to ask; auto-detected if unique")
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument(
        "--openclaw-bin", type=Path, default=CLAWBOT_HOME / "global" / "bin" / "openclaw"
    )
    parser.add_argument("--node-bin-dir", type=Path, default=CLAWBOT_HOME / "node26" / "bin")
    args = parser.parse_args()
    account_id = args.account_id or detect_account_id()
    settings = Settings(
        account_id=account_id,
        user_id=args.user_id or detect_user_id(account_id),
        openclaw_bin=args.openclaw_bin.expanduser(),
        node_bin_dir=args.node_bin_dir.expanduser(),
        state_dir=args.state_dir.expanduser(),
    )
    create_server(settings, args.host, args.port).run(transport="streamable-http")


if __name__ == "__main__":
    main()
