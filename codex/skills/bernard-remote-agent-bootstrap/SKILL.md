---
name: bernard-remote-agent-bootstrap
description: 通过 Merlin/Seed 服务管理页面找到 Bernard 服务的部署 Pod，进入 TTY 以 tmux 窗口的方式启动和验证 chrome-review remote-agent。适用于用户要求为 Seed/Bernard 推理服务开启 remote-agent；不要用于已经提供 IP 且 Agent 已运行的机器。
---

# Bernard Remote Agent Bootstrap

为当前运行中的 Bernard/Seed 服务实例建立可从本机访问的 HTTP remote-agent。浏览器操作使用已登录的服务管理页面；Agent 启动后再用 `remote-dev-agent` 的客户端和健康检查规则接管。

## 输入与默认值

- 需要用户提供网址或者服务名。
- 默认在该服务的所有实例上都启动 remote-agent。

## 定位当前实例

1. 使用应用内浏览器打开 Merlin/Seed 服务管理页。服务详情 URL 通常形如：
   `https://ml.bytedance.net/deployment/serviceList/service/detail/<service-name>/deploy`
2. 选择状态为“正在运行”的部署。
3. 有些服务可能有多个角色，则每个角色都要把 remote-agent 拉起来。
4. 分别记录所选 Pod 的角色、状态、完整 Pod 名和 IPv6，再逐个点击“进入 TTY”。实例刷新后必须重新读取，绝不沿用旧值。

页面可能使用折叠表格或弹出式 TTY。点击没有效果时，先检查遮挡弹窗和当前展开状态，不要连续盲点。TTY 输入出现 shell 续行提示时，关闭并重新打开 TTY，以获得干净会话。

## TTY 内启动

先检测机器类型。网络命令必须在同一个远程 shell 中设置对应代理：

- `env | grep -q '^[^=]*MERLIN[^=]*='` 成功时，使用项目 AGENTS.md 规定的 MERLIN 代理。
- 主机名以 `di-` 开头时，使用项目 AGENTS.md 规定的火山引擎代理。
- 两者都不匹配时，不臆造代理配置；先报告机器类型。

对每个选中 Pod 的 TTY 分别执行以下流程；一台完成后再处理下一台，避免把命令或 IP 写入错误的 TTY：

1. 检查 `python` 和 `18765` 端口是否可用。
2. 检查 tmux 是否已经安装，未安装则直接安装，不需要经过用户同意。
3. 检查同名 tmux session 是否存在。若已存在，查看 pane 命令和 Agent health；不要覆盖或重复启动。
4. 检查 `/chrome-review` 是否已经存在，不存在则从 `https://github.com/xpmemeda/chrome-review.git` 下载到该目录。
6. 在独立 tmux session/window 中启动，例如：

   ```bash
   tmux new-session -d -s remote-agent -n agent \
     "python /chrome-review/codex/remote-agent/agent.py --root /"
   ```

## 验证与交接

1. 用 `tmux list-panes` 确认 `remote-agent:agent` 中运行的是 `python`，并检查 tmux pane 输出没有启动异常。
2. 从当前服务页面分别重新读取每个所选实例的 IPv6，维护清晰的角色、状态、Pod 与 IP 映射。
3. 按 `remote-dev-agent` Skill 对每个地址分别执行 `http://[<IPv6>]:18765` 的 `health`。每个地址的 health 都必须是该 Agent 启动后的第一次远程调用。
4. 分别核对各返回结果的 `ok`、`hostname`、`pid` 和 `roots` 与对应 Pod 及预期权限一致。某台 health 失败时只排查该台的 tmux 日志、监听地址和当前实例 IP；不要误用历史 IP。
5. 保留用户需要继续查看的服务详情或 TTY 页面。

最终报告在哪些实例上拉起了 remote-agent。
