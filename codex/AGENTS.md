# 远程机器

## 登陆方式

用户可能以不同的方式提供机器。

### 别称

比如说 H20 / A100 / merlin-cc 之类的，非 IP 地址的名称，直接通过 `ssh <名称>` 登陆，登陆失败则在提示用户后终止对话。

### IPV6 地址

无法登陆，但是可以使用 `~/workspace/github/chrome-review/codex/remote-agent/client.py` 脚本来远程操控。

```bash
python3 client.py --url 'http://[IPV6]:18765' health
python3 client.py --url 'http://[IPV6]:18765' exec --cwd /workspace -- COMMAND
python3 client.py --url 'http://[IPV6]:18765' upload LOCAL_PATH REMOTE_PATH
python3 client.py --url 'http://[IPV6]:18765' download REMOTE_PATH LOCAL_PATH
```

- 先用 `health` 命令检测是否可连接，连接失败则在提示用户后终止对话。
- 使用 `exec` 执行有边界的命令。
- 使用 `start`、`status`、`logs` 和 `stop` 管理长时间运行的服务。
- 记录 `start` 返回的每个 job ID，确保后续调用管理正确进程。

## 环境

在远程机器上工作时，先检测机器所属的网络集群并设置对应的代理环境变量，然后再运行需要访问网络的命令。

大多数网络连接都需要代理，因此默认使用已配置的代理。部分地址可能无法通过代理访问。如果网络命令失败且可能由代理导致，仅针对该命令清除 `HTTP_PROXY`、`http_proxy`、`HTTPS_PROXY`、`https_proxy`、`ALL_PROXY` 和 `all_proxy` 后重试一次。不要在后续远程 shell 中永久取消代理变量。

### 梅林

如果以下命令执行成功，则将远程机器视为梅林的机器：

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

### 火山

如果远程机器的主机名以 `di-` 开头，则将其视为火山的机器：

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

## 资源地址

### 服务日志

根目录是 `/opt/tiger/toutiao/log/run`，有几个不同的日志文件：

- `run/bernard_stdout_log.YYYYMMDD-0000`: 服务的外层启动与管理脚本输出，当容器只有一个服务时，这里也会有服务日志。
- `executor_<N>.log.YYYY-MM-DD`: 当容器内部署了多个服务时，第 N 个服务的日志。

### 模型文件

模型文件通常存放在 `~/workspace/models` 下。下载所需模型前，先检查对应的本地目录是否已经存在且完整。未经用户确认，绝不覆盖非空或不完整的模型目录。

当远程推理、服务、基准测试或开发任务需要本地尚不存在的模型时，使用 `model-artifact-fetch` Skill。数据源优先级如下：

1. 复用完整的本地目录。
2. 当 HDFS 可执行文件和模型路径都存在时，使用 `/opt/tiger/yarn_deploy/hadoop/bin/hdfs` 从内部 HDFS 根目录 `hdfs://haruna/home/byte_device_intelligence_model/xiongpeng.123` 下载。
3. 仅当 HDFS 可执行文件不存在或 HDFS 中不存在该模型时，才回退到从 Hugging Face 克隆模型仓库。

先下载到同级临时目录，验证成功后再将其重命名到正式位置。如果 HDFS 显示模型存在，但 HDFS 下载失败，不要静默回退到 Hugging Face。

# 创建 Skill 规范

创建或更新用户专属的 Codex Skill 时：

- 使用中文来描述 Skill。
- 将 Skill 的规范源文件存放在 `~/workspace/github/chrome-review/codex/skills/<skill-name>` 下。
- 在 `~/.codex/skills/<skill-name>` 创建符号链接，以便 Codex 发现该 Skill。
- 将 `chrome-review` 仓库中的版本视为唯一事实来源；不要在 `~/.codex/skills` 下另存一份副本。
- 创建符号链接前，先检查目标位置是否已存在。未经用户确认，不要覆盖真实目录或指向其他位置的符号链接。
- 使用绝对路径作为符号链接目标，避免 Skill 发现依赖当前工作目录。

# 代码仓库

- 代码仓库放在本地机器 `~/workspace/github` 或者 `~/workspace/byted` 目录下。
- 提交 commit 时，信息里面带上 "Co-authored-by: Codex <noreply@openai.com>"，不要 push。

# 本地执行环境

- 使用交互式 zsh 加载用户环境，比如 `zsh -ic 'python ...'`

# 常见任务

## 下载远程机器的日志

1. 下载后解压保存到 `~/workspace/bernard-logs/YYYYMMDDHHMM-YYYYMMDDHHMM/<ipv6>.log`，目录名为日志的开始和结束时间。
2. 如果一次请求中包含了多台目标机器，请使用同一个目录。
