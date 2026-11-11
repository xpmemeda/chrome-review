#!/bin/sh
# Start (or restart) the wechat_confirm MCP server in tmux session "mcp-wechat-confirm".
set -eu
cd "$(dirname "$0")"
SESSION=mcp-wechat-confirm
LOG="$HOME/.local/state/wechat-confirm/server.log"
mkdir -p "$(dirname "$LOG")"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -c "$PWD" \
    "while :; do python3 mcp_server.py $* 2>&1 | tee -a '$LOG'; echo 'server exited; restarting in 3s'; sleep 3; done"
echo "started tmux session $SESSION; log: $LOG"
