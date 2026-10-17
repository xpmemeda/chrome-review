---
name: bernard-remote-agent-bootstrap
description: 用 Playwright MCP 从 Merlin/Seed 的 Bernard 部署页定位 Pod，在 TTY 中启动 chrome-review remote-agent，使其反向注册到 remote_fleet，并用 list_hosts/health 验证。适用于用户要求为服务实例开启 remote-agent；list_hosts 中已在线的设备无需重新启动。
---

# Bernard Remote Agent Bootstrap

为当前运行中的 Bernard/Seed Pod 启动反向连接的 remote-agent；仅操作目标 Pod，不修改推理服务。Agent 主动连接工作 VPS 上的 rendezvous 端口，不监听端口，也不需要登记 IP 或修改 `hosts.toml`。

## 准备

1. rendezvous 地址默认是工作 VPS `http://74.211.111.53:18765`，`reverse_agent.py` 已内置；用户指定其他地址时用 `--server` 覆盖。
2. 调用 remote_fleet 的 `list_hosts`，记下已在线的 agent 设备，避免重名或重复启动。

## 定位与进入 TTY

1. 用 Playwright MCP 打开用户给的 Merlin 或 Seed 部署 URL。若出现 SSO，点击飞书扫码入口，用 Playwright 截图把二维码交给用户，等登录完成再继续。直接以当前页面的部署 ID 和实例列表为准，不必在两套平台之间跳转。
2. 记录每个运行中实例的角色、Pod name 和状态，为每个实例确定唯一设备名（如 `<服务简称>-<角色>-<序号>`，只用字母、数字、`_`、`.`、`-`）。逐个处理实例，避免混淆 TTY 与设备。
3. 若某个设备名已在 `list_hosts` 中在线，且 `health` 返回的 hostname 与该 Pod 一致，复用它。
4. 点击该行的“进入 TTY”。终端位于嵌套 iframe，xterm 可能以 canvas 绘制：用 Playwright 的 `browser_type`/`browser_press_key` 向 `Terminal input` 输入命令；普通快照看不到输出时，用 `browser_run_code_unsafe` 选取 URL 以 `https://relay-lf.byted.org/` 开头的 frame，读取 `window.term.buffer.active.getLine(i).translateToString(true)`。只返回终端文本，不输出带登录票据的 iframe URL。

## 在 TTY 中启动

目标脚本是仓库的 `codex/mcps/remote-fleet/reverse_agent.py`。无需 clone 仓库；如已安装 tmux，TTY 中的简短命令是：

```bash
HTTPS_PROXY=http://sys-proxy-rd-relay.byted.org:8118 curl -fsSL \
  https://raw.githubusercontent.com/xpmemeda/chrome-review/dev/codex/mcps/remote-fleet/reverse_agent.py \
  -o /tmp/reverse_agent.py &&
python3 -m py_compile /tmp/reverse_agent.py &&
tmux new-session -d -s remote-agent \
  'python3 /tmp/reverse_agent.py <设备名> --env merlin --root /'
```

- 下载 `reverse_agent.py` 需要上面的代理；agent 连 VPS 的代理由 `--env` 内置决定，无需另设环境变量。火山机器（hostname 以 `di-` 开头）下载时用 `http://100.66.18.103:3128`，启动加 `--env volcano`。`--env` 必填，只能是 `merlin`、`volcano` 或 `null`（沿用环境变量中的代理，未设置则直连）。
- 若无 tmux，先在目标 Pod 安装（Debian/Ubuntu：`apt-get update -qq && apt-get install -y tmux`）；安装失败时停止并报告。
- 若 `remote-agent` tmux 会话已存在，先读会话输出（`tmux capture-pane -pt remote-agent -S -30`），不要重复启动。
- 输出 `connected to ... as '<设备名>'` 表示注册成功；`registration rejected` 表示设备名被其他在线 agent 或 ssh 主机占用，换名重启。
- 仓库中的 `scripts/start-remote-agent.sh` 也实现了代理与网络配置识别、tmux 启动和注册检查：`bash start-remote-agent.sh <设备名>`。

## 直连模式（备选）

仅当用户明确要求直连，或工作机器没有公网 IP 时使用：在 Pod 中用 tmux 运行 `python3 direct_agent.py --root /`（同目录下载 `direct_agent.py`，监听 18765），从工作机器验证 `http://[<IPv6>]:18765/v1/health` 后，在工作机器的 `codex/mcps/remote-fleet/hosts.toml` 增加 `backend = "agent"`、`url = "http://[<IPv6>]:18765"`、`root = "/"`、`env = "merlin"` 的主机条目，保留该文件其他改动。

## 验证

1. 调用 remote_fleet 的 `list_hosts`，确认新设备 `online: true`、hostname 与 Pod 一致、`env` 正确且无 `warning`。
2. 调用 `health` 验证往返请求。报告实际启动、复用与失败的设备，列出设备名、角色和 Pod name。
