# wechat_confirm

Let an agent ask the user a question over WeChat (OpenClaw ClawBot) and read the reply.

```
agent ── MCP (127.0.0.1:18767/mcp) ──▶ mcp_server.py ── openclaw gateway call send ──▶ WeChat
                                          ▲
WeChat reply ──▶ OpenClaw gateway ── before_dispatch (openclaw-bridge) ── POST /bridge/inbound
```

- `mcp_server.py`: MCP tools `ask_user`, `wait_reply`, `notify_user`, `cancel_question`,
  `list_questions`. Questions live in `~/.local/state/wechat-confirm/questions.sqlite`,
  so a timed-out `wait_reply` can simply be called again.
- `openclaw-bridge/`: OpenClaw plugin. While a question is pending, a reply is claimed
  for that question and the ClawBot agent does not answer it. With several pending
  questions, quote the question or start with `Q3`. Start a message with `bot:` to
  talk to the ClawBot normally.
- Sends go through the running gateway, so the WeChat context token is used. WeChat
  only accepts bot messages for some time after the user last messaged the bot;
  `status: send_failed` means delivery was rejected.

## Run

```bash
./start.sh            # tmux session mcp-wechat-confirm, log ~/.local/state/wechat-confirm/server.log
curl -s 127.0.0.1:18767/healthz
```

Account and user are auto-detected from `~/.openclaw/openclaw-weixin/`; pass
`--account-id` / `--user-id` to override.

Install the bridge once (hot-reloads into the gateway):

```bash
openclaw plugins install --link ./openclaw-bridge --force
openclaw plugins enable wechat-confirm-bridge
```

Register the MCP server:

```json
"wechat_confirm": { "type": "remote", "url": "http://127.0.0.1:18767/mcp", "oauth": false }
```
