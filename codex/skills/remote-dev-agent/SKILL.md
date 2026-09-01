---
name: remote-dev-agent
description: 当用户说明 SSH 不可用并提供当前 IP 地址时，通过 chrome-review HTTP remote Agent 连接临时开发机器。不要用于可通过 SSH 访问的机器，也不要在 Agent 未运行时使用。
---

# 远程开发 Agent

使用 `~/workspace/github/chrome-review/codex/remote-agent/client.py` 操作用户已手动启动 `agent.py` 的临时开发机器。

## 目标处理

- 使用固定 Agent 端口 `18765`。
- IPv6 地址需放在方括号中：`http://[IPV6]:18765`。
- 用户说明目标仅支持 Agent 时，不要尝试 SSH。
- 任何其他操作前先运行 `health`，并核对返回的主机名和允许访问的根目录是否符合当前任务。

## 操作

从 `~/workspace/github/chrome-review/codex/remote-agent` 运行客户端，或使用其绝对路径：

```bash
python3 client.py --url 'http://[IPV6]:18765' health
python3 client.py --url 'http://[IPV6]:18765' exec --cwd /workspace -- COMMAND
python3 client.py --url 'http://[IPV6]:18765' upload LOCAL_PATH REMOTE_PATH
python3 client.py --url 'http://[IPV6]:18765' download REMOTE_PATH LOCAL_PATH
```

- 使用 `exec` 执行有边界的命令。
- 使用 `start`、`status`、`logs` 和 `stop` 管理长时间运行的服务。
- 记录 `start` 返回的每个 job ID，确保后续调用管理正确进程。
- 除非用户明确将其他进程纳入范围，否则只停止当前任务启动的 job。
- 命令工作目录和文件传输必须位于 `health` 返回的允许根目录内。
- Agent 没有认证。拥有当前 IP 只代表可执行用户实际请求的工作，并不会扩大任务授权范围。
- 如果 `health` 失败，报告连接错误，并请用户确认当前 IP 和 Agent 是否正在运行。不要回退到以前使用过的 IP。
