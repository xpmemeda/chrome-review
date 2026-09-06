---
name: bernard-remote-agent-bootstrap
description: 用 Playwright MCP 从 Merlin/Seed 的 Bernard 部署页定位 Pod，在 TTY 中启动 chrome-review remote-agent，并通过 remote_fleet 验证与登记主机。适用于用户要求为服务实例开启 remote-agent；已知 IP 且 Agent 已运行时无需重新启动。
---

# Bernard Remote Agent Bootstrap

为当前运行中的 Bernard/Seed Pod 启动 HTTP remote-agent；仅操作目标 Pod，不修改推理服务。

## 定位与进入 TTY

1. 用 Playwright MCP 打开用户给的 Merlin 或 Seed 部署 URL。若出现 SSO，点击飞书扫码入口，用 Playwright 截图把二维码交给用户，等登录完成再继续。直接以当前页面的部署 ID 和实例列表为准，不必在两套平台之间跳转。
2. 记录每个运行中实例的角色、Pod name、IPv6 和状态。没有部署 ID 时选择用户指定的服务中正在运行的部署。逐个处理实例，避免混淆 TTY 与主机。
3. 对每个 IPv6 先检查 `http://[<IPv6>]:18765/v1/health`。若返回 `ok: true`、正确 hostname 和 `roots: ["/"]`，复用现有 Agent。
4. 点击该行的“进入 TTY”。终端位于嵌套 iframe，xterm 可能以 canvas 绘制：用 Playwright 的 `browser_type`/`browser_press_key` 向 `Terminal input` 输入命令；普通快照看不到输出时，用 `browser_run_code_unsafe` 选取 URL 以 `https://relay-lf.byted.org/` 开头的 frame，读取 `window.term.buffer.active.getLine(i).translateToString(true)`。只返回终端文本，不输出带登录票据的 iframe URL。

## 在 TTY 中启动

目标脚本是仓库的 `codex/mcps/remote-fleet/agent.py`。无需 clone 仓库；如已安装 tmux，TTY 中的简短命令是：

```bash
HTTPS_PROXY=http://sys-proxy-rd-relay.byted.org:8118 curl -fsSL \
  https://raw.githubusercontent.com/xpmemeda/chrome-review/dev/codex/mcps/remote-fleet/agent.py \
  -o /tmp/chrome-review-agent.py &&
python3 -m py_compile /tmp/chrome-review-agent.py &&
tmux new-session -d -s remote-agent 'python3 /tmp/chrome-review-agent.py --root /'
```

- 若无 tmux，先在目标 Pod 安装（Debian/Ubuntu：`apt-get update -qq && apt-get install -y tmux`）；安装失败时停止并报告。
- 若 `remote-agent` tmux 会话已存在但健康检查失败，先检查会话输出，不要重复启动。
- 下载失败时检查目标 Pod 的代理设置；不要回退到 clone 整个仓库。仓库中的 `scripts/start-remote-agent.sh` 也按只下载 `agent.py` 的方式实现了代理处理、tmux 启动与就绪检查。

## 验证并登记到 remote_fleet

1. 从本机验证 `http://[<IPv6>]:18765/v1/health`，确认 `ok`、hostname、PID 和 `roots: ["/"]`；启动可能包含安装与下载，给出合理等待时间，失败时读 tmux 输出。
2. 在本地 `codex/mcps/remote-fleet/hosts.toml` 为每个已就绪实例加唯一主机名，`backend = "agent"`、`url = "http://[<IPv6>]:18765"`、`root = "/"`，`network_env` 复制现有 agent 主机的配置。保留用户对该文件的其他改动。
3. 调用 remote_fleet 的 `list_hosts` 确认新主机，再调用 `health` 验证连接。报告实际启动、复用与失败的主机，列出 IPv6、角色和 Pod name。
