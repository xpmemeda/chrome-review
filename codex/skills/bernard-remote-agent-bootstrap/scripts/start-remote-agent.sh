#!/usr/bin/env bash
# 在目标 Pod 内运行：bash start-remote-agent.sh <device-name>
# 可选：REMOTE_FLEET_SERVER=http://74.211.111.53:18765 AGENT_PATH=/opt/tiger/remote-agent/reverse_agent.py TMUX_SESSION=remote-agent AGENT_ENV=merlin|volcano|null
set -Eeuo pipefail

DEVICE_NAME=${1:-${DEVICE_NAME:-}}
[[ -n $DEVICE_NAME ]] || { echo '用法：bash start-remote-agent.sh <device-name>' >&2; exit 2; }
REMOTE_FLEET_SERVER=${REMOTE_FLEET_SERVER:-http://74.211.111.53:18765}
AGENT_PATH=${AGENT_PATH:-/opt/tiger/remote-agent/reverse_agent.py}
AGENT_URL=${AGENT_URL:-https://raw.githubusercontent.com/xpmemeda/chrome-review/dev/codex/mcps/remote-fleet/reverse_agent.py}
TMUX_SESSION=${TMUX_SESSION:-remote-agent}
export DEVICE_NAME REMOTE_FLEET_SERVER AGENT_PATH AGENT_URL TMUX_SESSION

configure_proxy() {
    if [[ -n ${BERNARD_CONFIG+x} ]] || env | grep '^[^=]*MERLIN[^=]*=' >/dev/null; then
        export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
        export NO_PROXY='localhost,.byted.org,byted.org,.bytedance.net,bytedance.net,.byteintl.net,.tiktok-row.net,.tiktok-row.org,127.0.0.1,127.0.0.0/8,2605::/16'
        AGENT_ENV=${AGENT_ENV:-merlin}
    elif [[ $(hostname) == di-* ]]; then
        export HTTP_PROXY=http://100.66.18.103:3128
        export NO_PROXY='localhost,127.0.0.1,mirrors.ivolces.com,pypi.org,files.pythonhosted.org,pypi.python.org'
        export PIP_INDEX_URL=https://mirrors.ivolces.com/pypi/simple
        AGENT_ENV=${AGENT_ENV:-volcano}
    else
        printf '未识别集群（%s），保留现有代理配置。\n' "$(hostname)"
        AGENT_ENV=${AGENT_ENV:-null}
        return
    fi
    export http_proxy=$HTTP_PROXY HTTPS_PROXY=$HTTP_PROXY https_proxy=$HTTP_PROXY
    export no_proxy=$NO_PROXY
}

# 仅对失败的网络命令尝试一次直连，不修改当前 shell 的代理。
network() {
    "$@" || env -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy \
        -u ALL_PROXY -u all_proxy "$@"
}

AGENT_ENV=${AGENT_ENV:-}
configure_proxy
export AGENT_ENV
PYTHON=$(command -v python3 || command -v python || true)
[[ -n $PYTHON ]] || { echo '需要 Python 3，请先安装。' >&2; exit 1; }

if [[ ${2:-} == --worker ]]; then
    trap 'rc=$?; echo "启动失败（exit=$rc），请查看上方输出。" >&2; exit "$rc"' ERR
    mkdir -p "$(dirname "$AGENT_PATH")"
    stage=$(mktemp "${AGENT_PATH}.download.XXXXXX")
    # 仅下载 reverse_agent.py，先校验语法，再原子替换目标文件。
    network curl -fsSL "$AGENT_URL" -o "$stage"
    "$PYTHON" -m py_compile "$stage"
    mv -f "$stage" "$AGENT_PATH"
    args=("$DEVICE_NAME" --server "$REMOTE_FLEET_SERVER" --root /)
    [[ -n $AGENT_ENV ]] && args+=(--env "$AGENT_ENV")  # 未设置时 agent 默认 merlin
    exec "$PYTHON" "$AGENT_PATH" "${args[@]}"
fi

# Agent 不监听端口；以 tmux 输出中最后一条连接状态判断是否已注册。
agent_state() {
    tmux capture-pane -pt "=$TMUX_SESSION:agent" -S -200 2>/dev/null \
        | grep -E 'connected to|disconnected|cannot reach|registration rejected|rejected agent|启动失败' \
        | tail -n 1
}

wait_ready() {
    local deadline=$((SECONDS + 30)) line
    while ((SECONDS < deadline)); do
        line=$(agent_state || true)
        if [[ $line == *"connected to"* && $line != *disconnected* ]]; then
            printf 'READY %s\n' "$line"
            return 0
        fi
        sleep 1
    done
    echo '30 秒内未注册成功；保留后台会话，不要重复启动。最近输出：' >&2
    tmux capture-pane -pt "=$TMUX_SESSION:agent" -S -30 || true
    return 1
}

if ! command -v tmux >/dev/null; then
    [[ $EUID == 0 ]] || { echo '安装 tmux 需要 root，请以 root 运行。' >&2; exit 1; }
    command -v apt-get >/dev/null || { echo '此脚本自动安装仅支持 apt-get。' >&2; exit 1; }
    network apt-get update -qq
    network apt-get install -y tmux
fi

if tmux has-session -t "=$TMUX_SESSION" 2>/dev/null; then
    echo "已有 tmux 会话，未重复启动：$TMUX_SESSION"
    wait_ready
    exit $?
fi

script_path=$(realpath "$0")
# 已有 tmux server 时新会话不继承当前环境，显式传入 worker 需要的变量。
printf -v worker_cmd 'exec env REMOTE_FLEET_SERVER=%q AGENT_ENV=%q AGENT_PATH=%q AGENT_URL=%q bash %q %q --worker' \
    "$REMOTE_FLEET_SERVER" "$AGENT_ENV" "$AGENT_PATH" "$AGENT_URL" "$script_path" "$DEVICE_NAME"
# 先创建等待命令的窗口，开启退出后保留输出，再运行 worker。
tmux new-session -d -s "$TMUX_SESSION" -n agent
tmux set-option -w -t "=$TMUX_SESSION:agent" remain-on-exit on
tmux send-keys -t "=$TMUX_SESSION:agent" -l "$worker_cmd"
tmux send-keys -t "=$TMUX_SESSION:agent" Enter
printf '已提交后台启动；下载和 Agent 均在 tmux 窗口内运行。\n查看：tmux attach -t %q\n' "$TMUX_SESSION"
wait_ready
