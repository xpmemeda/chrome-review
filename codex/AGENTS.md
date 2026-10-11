# 回复要求

1. 输出表格的时候给我 markdown 原始字符串。

# 远程机器操作

- 用户提到机器别名、远程登录、执行远程命令或传输文件时，优先使用 `remote_fleet` MCP。
- 先调用 `list_hosts` 确认目标，再调用 `health` 检查连接。

## 临时绕过 Bernard 重启还原配置

- 不要把直接修改 `/etc/llmserver/config.yaml` 当作持久修改。无论执行 `bernard service restart` 还是直接 `systemctl restart lab.bernard.serving.service`，inferobust 启动时都可能重新生成该文件。
- 还原链路位于 `/opt/tiger/inferobust/executor.py` 的 `init_multiple_files()`：它读取进程环境变量 `BERNARD_CONFIG`，将其中的 JSON/base64 内容解码，再按照 `LLMSERVER_CONFIG_PATH` 等环境变量指向的路径写回配置文件。
- 排查时可以只列出 `BERNARD_CONFIG` 包含的键、目标路径和解码内容特征；不要直接打印完整环境变量，避免泄露 token、密码等敏感信息。
- 正式方案是修改 Bernard 平台下发的配置，使 `BERNARD_CONFIG` 中的原始 YAML 包含目标字段。
- 仅做临时服务实验、且暂时无法修改平台配置时，可以先备份 `/opt/tiger/inferobust/executor.py`，然后在 `init_multiple_files()` 中 `decoded_str = base64.b64decode(v).decode("utf-8")` 之后、写文件之前，按配置键注入修改。例如只处理 `k == "LLMSERVER_CONFIG_PATH"`，对 `decoded_str` 做精确替换。不要无条件修改所有下发文件。
- 修改后先运行 `python3 -m py_compile /opt/tiger/inferobust/executor.py`，再使用 `bernard service restart`。重启完成后必须检查 `/etc/llmserver/config.yaml`，确认注入字段仍存在；配置通常在 executor 启动后才写入，过早检查可能看到短暂的旧版本。
- 等 `/opt/tiger/toutiao/log/run/readiness.log.YYYY-MM-DD` 明确显示 pod ready、服务端口开始监听后，再发测试请求。仅看到 systemd `active (running)` 不代表模型已经可用。
- 实验结束后恢复 `executor.py` 备份并再次重启。不要将这种运行时注入当作长期部署方案，也不要修改或回显 `/var/docker_environment` 中完整的 `BERNARD_CONFIG`。

# 资源地址

不论是本地机器还是通过 MCP 操作的远程机器，资源路径均遵守以下规则。

## 日志

根目录是 `/opt/tiger/toutiao/log/run`，有几个不同的日志文件：

- `run/bernard_stdout_log.YYYYMMDD-0000`: 服务的外层启动与管理脚本输出，当容器只有一个服务时，这里也会有服务日志。
- `executor_<N>.log.YYYY-MM-DD`: 当容器内部署了多个服务时，第 N 个服务的日志。

## 模型文件

模型文件通常存放在 `~/workspace/models` 下。下载所需模型前，先检查对应的本地目录是否已经存在且完整。未经用户确认，绝不覆盖非空或不完整的模型目录。

当远程推理、服务、基准测试或开发任务需要本地尚不存在的模型时，使用 `model-artifact-fetch` Skill。数据源优先级如下：

1. 复用完整的本地目录。
2. 当 HDFS 可执行文件和模型路径都存在时，使用 `/opt/tiger/hdfs_client/bin/hdfs` 或者 `/opt/tiger/yarn_deploy/hadoop/bin/hdfs` 从内部 HDFS 根目录 `hdfs://haruna/home/byte_device_intelligence_model/xiongpeng.123` 下载。
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

## 编写原则

- 不用急着写测试和运行测试，先把功能实现了，用户 review 过后没问题会提示你去新增和运行测试。避免在不符合预期的实现上花费太长时间。
- 注意模块化的功能划分。

# 本地执行环境

- 使用交互式 zsh 加载用户环境，比如 `zsh -ic 'python ...'`

# 平台环境相关知识

有些环境变量，你在排查问题的时候可能需要意识到这些。

- BERNARD_ENABLE_CONSUL: 是否注册到 bernard 平台的服务发现。
- IGNORE_LIVENESS_CHECK: 是否绕过 bernard 平台的探活机制，否则启动时间过长会导致被平台杀掉重启。

还有些平台的行为

- 一个容器可能运行多个服务，平台可能分别检查各服务的监听端口；任一被检查端口停止监听都可能触发容器重启。不要只依据当前 shell 的 `PORT` / `PORT0` 判断需要保活的端口。
- 停止或重启服务前，先根据 `executor.py --service <N>` 及其子进程链定位各服务的 launcher、frontend 和 worker；只读取各 launcher 环境中的 `PORT0`、`CUDA_VISIBLE_DEVICES` 等必要字段，再用 `ss -lntp` 或 `psutil.net_connections()` 将实际监听端口与进程对应。多个 frontend 共享同一端口不代表多个服务，worker 的内部通信端口也不要当作服务入口；共享网络命名空间中无法归属当前容器进程的端口应单独标明。
- 调试切换服务时，应为所有受影响的平台探活端口安排轻量保活进程接管，并确认原服务退出不会使 executor 或容器入口进程结束、也不会被自动拉起。保活端口监听正常不代表模型已就绪，仍须检查模型自身的 readiness 和实际推理请求。

## 一些可用的工具

1. 通过 `bvc` 命令下载 SCM 发布的代码库。

# 常见任务

## 下载远程机器的日志

1. 下载后解压保存到 `~/workspace/bernard-logs/YYYYMMDDHHMM-YYYYMMDDHHMM/<ipv6>.log`，目录名为日志的开始和结束时间。
2. 如果一次请求中包含了多台目标机器，请使用同一个目录。
