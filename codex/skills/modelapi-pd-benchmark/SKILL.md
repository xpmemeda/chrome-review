---
name: modelapi-pd-benchmark
description: 使用 ocean.ocean-benchmark 对 ModelAPI 的 Prefill/Decode 分离服务做性能压测。适用于用户要求测量 Prefill 性能、Decode TPS、TPOT、并发扩展或最大并发；在 Decode 服务未确认启用 benchmark 请求的固定投机采样接纳率时不要启动测试。
---

# ModelAPI PD 性能测试

在 `~/workspace/byted/ocean.ocean-benchmark` 中执行测试。目标是分别测量 Prefill 和 Decode 节点，保留可复查的命令、日志与结果，并拒绝不满足前置条件的测试。

## 测试门槛

开始任何压测前，检查用户是否明确提供以下信息：

- ModelAPI 的 `--model`。
- ModelAPI 的 `--modelapi-env`。
- Decode server 已针对 benchmark 前缀请求启用固定接纳率，以及具体接纳率。若未调整、未生效或用户无法确认，停止，不运行 Prefill 或 Decode 测试；告诉用户缺失这一前置条件。
- 请求的 context 长度。
- 请求的平均 KV cache 命中长度；同时记录命中率 `平均命中长度 / context 长度`。
- 每个请求的固定输出长度。

如果用户给出的 aggregate 长度不足以确定 Prefill 多轮请求形态，再询问首轮长度、首轮命中长度、轮数、每轮 append token、image token、rewrite token 和最大历史图片数。不要静默套用示例参数冒充用户的目标负载。

测试前确认本地脚本包含用户需要的最新改动。不得擅自修改或重启服务；只有用户明确要求时才这样做。

## Prefill 测试

使用 `modelapi-rewrite-vlm-streaming.py`。根据用户目标替换模型、环境、请求形态、并发和输出长度。下面是命令结构示例，不是通用固定参数：

```bash
python modelapi-rewrite-vlm-streaming.py --model omniagent_m14_6b_0824 --modelapi-env ppe_model_center --first-round-tokens 19679 --first-round-hit-tokens 5145 --rounds 26 --append-tokens 251 --image-tokens 495 --rewrite-tokens 0 --process 3 --concurrency-per-process 4 --iterations 1 --timeout 120 --max-retries 0 --log-path omni-30b-0707.log --min-tokens 442 --max-tokens 442 --max-history-images 5
```

要求：

- 固定输出长度时令 `--min-tokens` 和 `--max-tokens` 相等。
- 用明确且唯一的 `--log-path` 保存本次结果，避免覆盖无法追溯。
- 检查成功请求数、失败原因、各轮输入/命中形态和 Prefill 延迟。出现失败时不要只报告成功请求的性能。
- 实际输入长度、命中长度或命中率偏离目标时，本轮数据无效；先解释偏差再决定是否重测。

## Decode 测试

使用 `benchmark-dec.py`。该脚本为每个 client 建立独立 session：warmup 使用 `system -> user`，之后每轮累计追加 `assistant("hi") -> user("hi")`。根据用户目标替换 context、并发 sweep、请求数和输出长度：

```bash
python benchmark-dec.py --context-len 25000 --client modelapi --model omniagent_m14_6b_0824 --modelapi-env ppe_model_center --concurrency-sweep 1 16 32 --num-requests-sweep 8 64 128 --warmup-requests 1 --min-tokens 442 --max-tokens 442 --timeout 180
```

要求：

- 保留每个并发点的 QPS、TPOT、Decode TPS、成功率、平均服务端输出 token、平均输入 token、平均 cached token 和命中率。
- 只把全部请求成功且平均服务端输出 token 等于目标输出长度的并发点视为有效容量数据。
- 固定输出长度为 `L` 时校验 `decode_tps / qps ≈ L`。若不成立，检查 `avg_server_toks`；不要把提前结束的请求当成固定长度结果。
- warmup 失败重试不计入正式性能，但必须报告；持续出现 `Cancelled by backend` 时停止扩大并发并说明服务已过载或需要结合服务日志定位。
- 最大并发是满足成功率、固定输出长度和用户延迟目标的最高已验证并发，不要仅以脚本能发出多少请求来定义。

## 结果交付

报告中写明：

- 完整可复现命令、模型、环境、固定接纳率、context/命中/输出长度。
- Prefill 请求形态、并发、成功率、关键延迟和异常。
- Decode 各有效并发点的 QPS、TPOT、Decode TPS、实际输出长度和 cache 命中。
- 未通过的并发点、失败类型及其是否构成容量上限证据。
- 日志和机器可读结果文件的绝对路径。

若前置条件不完整，只返回缺失项，不生成貌似有效的测试结论。
