---
name: smoke-test
description: 当用户要求发起冒烟测试时，使用 ocean.ocean-benchmark/smoke 中的脚本，按编号依次测试用户提供的 llmhub IP 和端口，覆盖请求回放、纯文本、图片及同一 session 多轮滚动删图。
---

# 冒烟测试

## 输入与执行位置

- 必须取得用户提供的 llmhub IP 和端口；可复用当前对话中明确的目标。缺少时只询问缺失项，不使用脚本示例地址或猜测端口。支持分别提供 IP、端口或 `[IPv6]:port`，端口须在 1–65535 内。
- 默认在本机执行 `~/workspace/byted/ocean.ocean-benchmark/smoke` 中的客户端脚本，用 `zsh -ic` 加载用户环境。仓库缺失时，在 `~/workspace/github` 和 `~/workspace/byted` 下查找。
- llmhub IP 是 gRPC 服务地址，提供 IP 本身不表示要登录机器，也不要求该服务运行 remote-agent。仅在用户指定远程执行机时，按适用的远程连接和代理规范执行。

## 执行前检查

读取当前仓库的 `smoke/1.py`、`2.py`、`3.py`、`4.py`，核实参数、资源路径和调用顺序；以代码为准，不以可能过时的注释为准。复用现有 Python 环境，检查能导入 `grpc` 和仓库内的 `llmserver.proto.ultraman_pb2`、`ultraman_pb2_grpc`。

默认资源如下，均须在执行客户端的机器上可读：

- `1.py`：`~/workspace/datasets/omni-agent/202609021652286639CBC046C543512923/step-14-request.json`。可使用用户指定的 `--messages PATH`；JSON/JSONL 多样本文件仅发送第一条。
- `3.py`：`~/workspace/resources/pictures/cat.png` 和 `cat.heif`，脚本内部按此顺序发送。
- `4.py`：默认使用上述数据集目录，当前 `STEPS` 为 12、13、14。每个 step 必须恰好匹配一个 `*-step-N-messages.json` 或 `step-NN-request.json`。可使用用户指定的 `--requests-dir PATH`。

资源或依赖缺失时说明具体缺失项，优先查找已有资源或可用环境；无法解决再向用户询问，不伪造样本、不跳过后宣称全部通过。

当前 `1.py`、`2.py`、`3.py` 无条件将 IP 拼为 `[IP]:port`，适用于 IPv6；`4.py` 已区分 IPv4/IPv6。如果用户提供 IPv4 且当前版本仍有这一限制，在本次工作目录创建临时副本，仅将前三个脚本的地址拼接改为 IPv6 用 `[host]:port`、IPv4 用 `host:port`，并将副本的 `REPO_ROOT` 指回真实仓库。保留其他逻辑和资源路径，不修改仓库原脚本。记录本次使用了地址兼容副本。

## 顺序调用

在仓库根目录运行以下调用；`LLMHUB_IP` 和 `LLMHUB_PORT` 是已核实的用户输入。逐条执行并等待结束，禁止并发、循环压测或自动增加请求数量。

```bash
python3 smoke/1.py "$LLMHUB_IP" "$LLMHUB_PORT"
python3 smoke/2.py "$LLMHUB_IP" "$LLMHUB_PORT"
python3 smoke/3.py "$LLMHUB_IP" "$LLMHUB_PORT"
python3 smoke/4.py "$LLMHUB_IP" "$LLMHUB_PORT"
```

1. `1.py`：回放一组 messages，记录返回内容及 input/output tokens、TTFT、TPOT。
2. `2.py`：发送不含 system prompt 的纯 user 文本，默认问题为“请用中文简要介绍一下你自己。”。
3. `3.py`：依次描述 `cat.png`、`cat.heif`，分别检查输出。
4. `4.py`：先验证样本图片滚动关系，再在同一 session/task 中依次回放各 step，记录返回内容、tokens、cached tokens、TTFT、TPOT。必须整段运行脚本以保留 session，不能拆成独立会话。

保留脚本默认参数，除非用户明确覆盖。当前每个 RPC 默认超时 120 秒；`3.py` 和 `4.py` 包含多次 RPC，不能把整个脚本的总时限误设为 120 秒。工具提前返回进程会话 ID 时，继续读取该会话直到退出，再执行下一项。

逐项保留 stdout、stderr、退出码和耗时。某项异常退出时记录失败，继续下一编号以完成覆盖；同一脚本内部失败后未执行的图片或 step 标为未执行，不自行拆开补跑。若目标持续不可达、用户取消或执行环境无法继续，则终止并列明剩余项。不要无边界重试。

## 结果判定与回复

这些脚本主要发送请求和打印结果，没有完整的语义断言。结合退出码、异常、响应内容判定，不能仅凭退出码 0 宣称通过：纯文本应有可读回答，两种图片应分别返回与图片相符的描述，多轮回放应完成所有 step 且无协议或服务错误。工具调用等非文本响应需要结合实际返回判断；只有空输出且无足够证据时标为“需检查”。指标为 `N/A` 不单独视为失败，也不补造数据。

用中文简要汇报目标地址、各脚本结果、图片/多轮子项及失败原因，提供已有性能指标和日志位置。只有全部项目完成且满足检查条件时才报告整体通过。若使用表格，放在代码块中输出 Markdown 原始字符串。
