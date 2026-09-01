#!/usr/bin/env bash

set -u

event="${1:-}"
if [[ -z "$event" ]]; then
  echo "run-hook.sh: missing hook event" >&2
  exit 2
fi

payload="$(mktemp "${TMPDIR:-/tmp}/codex-hook.XXXXXX")"
trap 'rm -f "$payload"' EXIT
cat > "$payload"

status=0

run_step() {
  local name="$1"
  local timeout_seconds="$2"
  shift 2

  # macOS does not ship GNU timeout. Perl's alarm is inherited by exec and
  # keeps the timeout local to each hook action.
  perl -e 'alarm shift; exec @ARGV' "$timeout_seconds" "$@" < "$payload"
  local step_status=$?
  if [[ $step_status -ne 0 ]]; then
    echo "run-hook.sh: $event/$name failed (status $step_status)" >&2
    status=1
  fi
}

run_ai_report() {
  run_step ai-report 30 env \
    TEA_APP_ID=1013111 \
    TEA_CHANNEL=cn \
    npx -y --prefix /tmp --registry=https://bnpm.byted.org \
    -p @dp/ai-code-report@latest -- \
    ai-report-hook-run codex "$event"
}

run_log() {
  local label="$1"
  printf '[%s] %s hook executed\n' \
    "$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S CST')" \
    "$label" >> "$HOME/.codex.log" || status=1
}

case "$event" in
  sessionStart)
    run_ai_report
    run_log SessionStart
    ;;
  userPromptSubmit)
    run_step log-user-prompt 30 \
      python3 "$HOME/workspace/github/chrome-review/codex/hooks/log_user_prompt.py"
    ;;
  preToolUse)
    run_ai_report
    run_log PreToolUse
    ;;
  postToolUse)
    run_ai_report
    run_log PostToolUse
    ;;
  stop)
    run_ai_report
    run_log Stop
    ;;
  subagentStop)
    run_ai_report
    run_log SubagentStop
    ;;
  *)
    echo "run-hook.sh: unknown hook event: $event" >&2
    exit 2
    ;;
esac

exit "$status"
