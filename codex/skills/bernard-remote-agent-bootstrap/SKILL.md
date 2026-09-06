---
name: bernard-remote-agent-bootstrap
description: 通过 Merlin/Seed 服务管理页面找到 Bernard 服务的部署 Pod，进入 TTY 以 tmux 窗口的方式启动和验证 chrome-review remote-agent。适用于用户要求为 Seed/Bernard 推理服务开启 remote-agent；不要用于已经提供 IP 且 Agent 已运行的机器。
---

# Bernard Remote Agent Bootstrap

- 为当前运行中的 Bernard/Seed 服务实例建立可从本机访问的 HTTP remote-agent。
- 浏览器操作使用已登录的服务管理页面。
- 不要影响服务的运行。

## 输入与默认值

- 需要用户提供网址。
   - Merlin 服务管理页 URL 通常形如：`https://ml.bytedance.net/deployment/serviceList/service/detail/<service-name>/deploy/<deploy-id>`
   - Seed 服务管理页 URL 通常形如：`https://seed.bytedance.net/model/serviceList/service/detail/<service-name>/deploy/<deploy-id>`

## 定位当前实例

1. 如果用户提供的是 Seed URL，则从网址中提取 `<service-name>` 和 `<deploy-id>`，并在 Merlin 服务管理页中搜索该服务和部署。
2. 如果用户提供的网址中没有 `deploy-id`，则选择状态为“正在运行”的部署。
3. 有些服务可能有多个角色，分别记录所选 Pod 的角色和 IPv6。
4. 如果有多台实例，请并行处理每台实例。

## 启动脚本

1. 打开 Merlin URL 后，先用 IPv6 访问 `http://[<IPv6>]:18765/v1/health`，确认该实例的 remote-agent 尚未启动，若已启动则无需重复操作。
2. 点击“进入 TTY”，然后直接执行下面的命令：

   ```bash
   curl -fsSL https://raw.githubusercontent.com/xpmemeda/chrome-review/dev/codex/skills/bernard-remote-agent-bootstrap/scripts/start-remote-agent.sh \
   -o /tmp/start-remote-agent.sh &&
   bash /tmp/start-remote-agent.sh
   ```

3. 发出命令后不需要等待，直接关闭 TTY 窗口即可。

## 验证与交接

1. 用 `http://[<IPv6>]:18765/v1/health` 分别验证每个实例的 remote-agent 是否启动成功，最多等待 10 秒，10 秒后仍未能通过则视为失败。
2. 向用户报告成功和失败的实例列表，并提供每个实例的 IPv6 地址和角色。
