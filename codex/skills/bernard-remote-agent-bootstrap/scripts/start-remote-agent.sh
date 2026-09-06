#!/usr/bin/env bash
# 在目标 Pod 内运行：bash start-remote-agent.sh
# 可选：AGENT_REPO=/chrome-review TMUX_SESSION=remote-agent
set -Eeuo pipefail

AGENT_REPO=${AGENT_REPO:-/chrome-review}
TMUX_SESSION=${TMUX_SESSION:-remote-agent}
export AGENT_REPO TMUX_SESSION

configure_proxy() {
    if env | grep '^[^=]*MERLIN[^=]*=' >/dev/null; then
        export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
        export NO_PROXY='localhost,.byted.org,byted.org,.bytedance.net,bytedance.net,.byteintl.net,.tiktok-row.net,.tiktok-row.org,127.0.0.1,127.0.0.0/8,2605::/16'
    elif [[ $(hostname) == di-* ]]; then
        export HTTP_PROXY=http://100.66.18.103:3128
        export NO_PROXY='localhost,127.0.0.1,mirrors.ivolces.com,pypi.org,files.pythonhosted.org,pypi.python.org'
        export PIP_INDEX_URL=https://mirrors.ivolces.com/pypi/simple
    else
        printf '未识别集群（%s），保留现有代理配置。\n' "$(hostname)"
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

configure_proxy
PYTHON=$(command -v python3 || command -v python || true)
[[ -n $PYTHON ]] || { echo '需要 Python 3，请先安装。' >&2; exit 1; }

if [[ ${1:-} == --worker ]]; then
    trap 'rc=$?; echo "启动失败（exit=$rc），请查看上方输出。" >&2; exit "$rc"' ERR
    if [[ ! -e $AGENT_REPO ]]; then
        command -v git >/dev/null || { echo '需要 git，请先安装。' >&2; exit 1; }
        # 下载到临时目录，成功后再移到正式位置。
        mkdir -p "$(dirname "$AGENT_REPO")"
        stage=$(mktemp -d "${AGENT_REPO}.download.XXXXXX")
        if ! git clone --depth 1 --branch master https://github.com/xpmemeda/chrome-review.git "$stage"; then
            # 失败目录留作排查，直连重试使用新的空目录。
            stage=$(mktemp -d "${AGENT_REPO}.download.XXXXXX")
            env -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy \
                -u ALL_PROXY -u all_proxy git clone --depth 1 --branch master \
                https://github.com/xpmemeda/chrome-review.git "$stage"
        fi
        [[ -f $stage/codex/remote-agent/agent.py ]]
        [[ ! -e $AGENT_REPO ]] || { echo '目标目录已被其他进程创建，停止。' >&2; exit 1; }
        mv -T "$stage" "$AGENT_REPO"
    fi
    [[ -f $AGENT_REPO/codex/remote-agent/agent.py ]] || {
        echo "已有目录缺少 agent.py，未覆盖：$AGENT_REPO" >&2
        exit 1
    }
    exec "$PYTHON" "$AGENT_REPO/codex/remote-agent/agent.py" --root /
fi

# 直接检查本机 IPv6 回环，绕过代理；成功必须匹配当前机器和预期根目录。
check_health() {
    "$PYTHON" - "$1" <<'PY'
import json
import os
import sys
import time
import urllib.request

deadline = time.monotonic() + float(sys.argv[1])
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
error = "尚未就绪"
while time.monotonic() < deadline:
    try:
        with opener.open("http://[::1]:18765/v1/health",
                         timeout=min(3, max(0.01, deadline - time.monotonic()))) as response:
            value = json.load(response)
        if not (isinstance(value, dict) and value.get("ok") is True
                and value.get("hostname") == os.uname().nodename
                and type(value.get("pid")) is int and value["pid"] > 0
                and value.get("roots") == ["/"]):
            raise ValueError("health 返回的身份或根目录不匹配")
        print(f'READY hostname={value["hostname"]} pid={value["pid"]} port=18765 roots=/')
        sys.exit(0)
    except (OSError, ValueError) as exc:
        error = str(exc)
    time.sleep(min(1, max(0, deadline - time.monotonic())))
print(f"健康检查未通过：{error}", file=sys.stderr)
sys.exit(1)
PY
}

wait_ready() {
    if check_health 20; then
        return 0
    fi
    echo '20 秒内未就绪；保留后台会话，可能仍在下载。不要重复启动。' >&2
    tmux list-panes -t "=$TMUX_SESSION" -F '#S:#W command=#{pane_current_command} dead=#{pane_dead}' || true
    tmux capture-pane -pt "=$TMUX_SESSION:agent" -S -60 || true
    return 1
}

# 复用已健康的 Agent，不要求它必须由当前 tmux 会话托管。
if check_health 1 2>/dev/null; then
    exit 0
fi

if ! command -v tmux >/dev/null; then
    [[ $EUID == 0 ]] || { echo '安装 tmux 需要 root，请以 root 运行。' >&2; exit 1; }
    command -v apt-get >/dev/null || { echo '此脚本自动安装仅支持 apt-get。' >&2; exit 1; }
    network apt-get update -qq
    network apt-get install -y tmux
fi

if tmux has-session -t "=$TMUX_SESSION" 2>/dev/null; then
    echo "已有 tmux 会话，未重复启动：$TMUX_SESSION"
    wait_ready
    exit 0
fi

# 检查监听端口，避免启动第二个服务或干扰占用端口的进程。
"$PYTHON" - <<'PY'
import socket
with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
    try:
        sock.bind(('::', 18765))
    except OSError as exc:
        raise SystemExit(f'18765 端口不可用，未启动：{exc}')
PY

script_path=$(realpath "$0")
printf -v worker_cmd 'exec bash %q --worker' "$script_path"
# 先创建等待命令的窗口，开启退出后保留输出，再运行 worker。
tmux new-session -d -s "$TMUX_SESSION" -n agent
tmux set-option -w -t "=$TMUX_SESSION:agent" remain-on-exit on
tmux send-keys -t "=$TMUX_SESSION:agent" -l "$worker_cmd"
tmux send-keys -t "=$TMUX_SESSION:agent" Enter
printf '已提交后台启动；下载和 Agent 均在 tmux 窗口内运行。\n查看：tmux attach -t %q\n' "$TMUX_SESSION"
wait_ready
