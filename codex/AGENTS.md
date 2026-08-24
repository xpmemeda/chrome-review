# 远程开发环境

在远程开发机器上工作时，先检测机器类型并设置对应的代理环境变量，然后再运行需要访问网络的命令。

大多数网络连接都需要代理，因此默认使用已配置的代理。部分地址可能无法通过代理访问。如果网络命令失败且可能由代理导致，仅针对该命令清除 `HTTP_PROXY`、`http_proxy`、`HTTPS_PROXY`、`https_proxy`、`ALL_PROXY` 和 `all_proxy` 后重试一次。不要在后续远程 shell 中永久取消代理变量。

## MERLIN

如果以下命令执行成功，则将远程机器视为 MERLIN 机器：

```bash
env | grep -q '^[^=]*MERLIN[^=]*='
```

在运行后续命令的同一个远程 shell 中设置以下变量：

```bash
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy="${HTTP_PROXY}"
export HTTPS_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="localhost,.byted.org,byted.org,.bytedance.net,bytedance.net,.byteintl.net,.tiktok-row.net,.tiktok-row.org,127.0.0.1,127.0.0.0/8,2605::/16"
export no_proxy="${NO_PROXY}"
```

## 火山引擎

如果远程机器的主机名以 `di-` 开头，则将其视为火山引擎机器：

```bash
[[ "$(hostname)" == di-* ]]
```

在运行后续命令的同一个远程 shell 中设置以下变量：

```bash
export HTTP_PROXY="http://100.66.18.103:3128"
export http_proxy=$HTTP_PROXY
export HTTPS_PROXY="http://100.66.18.103:3128"
export https_proxy=$HTTPS_PROXY
export NO_PROXY="localhost,127.0.0.1,mirrors.ivolces.com,pypi.org,files.pythonhosted.org,pypi.python.org"
export PIP_INDEX_URL=https://mirrors.ivolces.com/pypi/simple
```

# Codex Skills

创建或更新用户专属的 Codex Skill 时：

- 使用中文来描述 Skill。
- 将 Skill 的规范源文件存放在 `~/workspace/github/chrome-review/codex/skills/<skill-name>` 下。
- 在 `~/.codex/skills/<skill-name>` 创建符号链接，以便 Codex 发现该 Skill。
- 将 `chrome-review` 仓库中的版本视为唯一事实来源；不要在 `~/.codex/skills` 下另存一份副本。
- 创建符号链接前，先检查目标位置是否已存在。未经用户确认，不要覆盖真实目录或指向其他位置的符号链接。
- 使用绝对路径作为符号链接目标，避免 Skill 发现依赖当前工作目录。

# 模型文件

模型文件通常存放在 `~/workspace/models` 下。下载所需模型前，先检查对应的本地目录是否已经存在且完整。未经用户确认，绝不覆盖非空或不完整的模型目录。

当远程推理、服务、基准测试或开发任务需要本地尚不存在的模型时，使用 `model-artifact-fetch` Skill。数据源优先级如下：

1. 复用完整的本地目录。
2. 当 HDFS 可执行文件和模型路径都存在时，使用 `/opt/tiger/yarn_deploy/hadoop/bin/hdfs` 从内部 HDFS 根目录 `hdfs://haruna/home/byte_device_intelligence_model/xiongpeng.123` 下载。
3. 仅当 HDFS 可执行文件不存在或 HDFS 中不存在该模型时，才回退到从 Hugging Face 克隆模型仓库。

先下载到同级临时目录，验证成功后再将其重命名到正式位置。如果 HDFS 显示模型存在，但 HDFS 下载失败，不要静默回退到 Hugging Face。

# 临时开发机器

当用户提供临时开发机器的 IP 地址，并提出任务时，先使用 `remote-dev-agent` Skill 检查 HTTP remote Agent，再对该机器执行任何其他操作。

- 使用 Agent 的固定端口 `18765`，并在任何其他操作之前运行健康检查。
- 如果健康检查失败或 Agent 未运行，停止任务并请用户启动 Agent。不要尝试通过 SSH 连接，也不要将 SSH 作为备用方案。
