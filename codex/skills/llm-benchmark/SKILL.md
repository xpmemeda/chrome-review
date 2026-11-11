---
name: llm-benchmark
description: 使用 ocean.ocean-benchmark 通过 ModelAPI 或 llmhub（Ultraman gRPC）客户端压测大模型推理服务，适用于 PD 分离场景，支持分别测试 Prefill 或 Decode。
---

# 大模型 PD 分离推理服务性能测试

在 `~/workspace/byted/ocean.ocean-benchmark` 中执行测试，分别测量 Prefill 和 Decode 节点，用户也可能指定只测量 Prefill 或者 Decode。

- 用户提供 llmhub `IP:port` 或明确指定 Ultraman 时，使用 `--client ultraman`。
- 否则使用 `--client modelapi`，并要求用户指定 `--model` 和 `--modelapi-env`。

有几个注意事项：

1. 场景或客户端类型尚不明确时，先向用户确认；当前对话已明确提供的参数直接复用。
2. 测试如果有失败，需要向用户包括失败的类型。目前已知的有如下几种。
   只有 code = 1 的错误是可以容忍的，其他错误可能暗示着服务已经被压挂了，需要重点提醒用户。
  - rpc error: code = 1 desc = Cancelled by backend [biz error]
  - rpc error: code = 14 desc = Socket closed [biz error]

## Prefill 测试

使用 `modelapi-rewrite-vlm-streaming.py`，根据模型架构选择配置：

- M12: `modelapi-rewrite-vlm-streaming-m12-30b.json`
- M13: `modelapi-rewrite-vlm-streaming-m13-12b.json`
- M14: `modelapi-rewrite-vlm-streaming-m14-6b.json`

复制对应配置文件到本次结果目录，再修改以下字段；不要改仓库中的模板：

- ModelAPI：`model` 和 `modelapi-env` 改为用户指定值。
- llmhub：`client` 设为 `ultraman`，填写 `ultraman-host`、`ultraman-port`，`model` 和 `modelapi-env` 设为空字符串。
- min-tokens: 固定为 1。
- max-tokens: 固定为 1。
- process: 从 1 开始，逐步增加，直到 uncached-prefill-tps 不发生明显变化。
- concurrency-per-process: 固定为 4。
- iterations: 固定为 3。

```bash
python modelapi-rewrite-vlm-streaming.py --config /path/to/updated-config.json
```

## Decode 测试

使用 `benchmark-dec.py --mode steady` 测量 Decode 吞吐，所有命令必须显式指定 `--mode steady`。该脚本为每个 client 建立独立 session：warmup 使用 `system -> user`，之后每轮累计追加 `assistant("hi") -> user("hi")`。
测试的参数如下：

- ModelAPI：传入用户指定的 `--model`、`--modelapi-env` 和 `--client modelapi`。
- llmhub：传入 `--client ultraman --ultraman-host <IP> --ultraman-port <PORT>`；不要沿用模板中的 ModelAPI 模型名或环境。
- min-tokens: 固定为 1024。
- max-tokens: 固定为 1024，延长每个 client 的 Decode 统计窗口。
- context-len: 固定为 25000。
- 测量时段：每档并发独立重复 3 次；每次使用统一的测量起止时间，默认测量 60 秒，用户指定时使用用户时长。
- steady-trim-sec: 每个成功窗口两端各裁剪 0.3 秒。
- steady-max-retries: 失败后最多额外重试 2 次。
- concurrency-sweep: 固定为 1、2、4、8、16、32、64，按顺序测试；结合共同窗口时长和吞吐判断是否饱和。

使用 `--steady-duration-sec 60 --steady-waves 3` 指定每次统一测量 60 秒、独立重复 3 次。`steady-waves` 表示统一测量时段数，不是每个 client 的请求数。

ModelAPI 基础参数：

```bash
python benchmark-dec.py --mode steady --context-len 25000 --client modelapi --model <MODEL> --modelapi-env <ENV> --concurrency-sweep 1 2 4 8 16 32 64 --steady-duration-sec 60 --steady-waves 3 --steady-trim-sec 0.3 --steady-max-retries 2 --min-tokens 1024 --max-tokens 1024 --timeout 180 --jsonl /path/to/steady.jsonl
```

llmhub 基础参数：

```bash
python benchmark-dec.py --mode steady --context-len 25000 --client ultraman --ultraman-host <IP> --ultraman-port <PORT> --concurrency-sweep 1 2 4 8 16 32 64 --steady-duration-sec 60 --steady-waves 3 --steady-trim-sec 0.3 --steady-max-retries 2 --min-tokens 1024 --max-tokens 1024 --timeout 180 --jsonl /path/to/steady.jsonl
```

Ultraman gRPC 客户端不能保证 `--min-tokens` 生效；记录实际输出长度，不能将短输出称为固定长度结果，但可以使用其成功生成的实际时间窗口计算吞吐。

每次测量预热后，协调所有 client 使用统一的测量起止时间。测量期间每个 client 独立持续请求，单个请求完成后继续发下一次，不在请求轮次之间等待其他 client；不因某个 client 先完成固定请求数而提前退出。到达统一截止时间后停止补发和重试，允许在途请求收尾，但统计窗口必须裁剪到统一测量时段内。重复测量之间可以统一同步。

每个 client 失败时在同一 session 上重试（最多额外 2 次，递增退避），保留所有尝试的错误和成功窗口，不因一个请求失败或短输出而废弃整个测量时段。每个成功请求提供自己的 Decode 时间窗口，两端按 `--steady-trim-sec` 裁剪；没有足够长度的窗口单独排除。

统计时先将每个 client 的成功窗口在时间轴上合并，再对所有目标 client 的窗口集合取交集。共同窗口可以由多段不连续时间组成；仅统计这些共同窗口内所有 client 实际收到的输出 token，总数除以共同窗口总时长得到 `steady_tps`。窗口采用左闭右开区间 `[start, end)`，分子和分母必须使用同一组共同窗口。不得将全部 client 的窗口取并集作为 steady 吞吐的分母，也不得把共同窗口之外的 token 计入分子。

单个请求失败或短输出不使整段测量作废；独立重试后取得的成功窗口照常参与交集。但必须包含全部目标 client，不能悄悄移除失败 client 来凑出窗口。若某个 client 全程没有有效成功窗口，或各 client 的成功窗口完全没有共同时间，则该测量没有满并发样本，标为无效并说明原因。

必须报告共同窗口总时长，窗口过短时说明代表性有限。`span_tps` 仅为共同窗口 token 数除以最早共同窗口开始到最晚共同窗口结束的跨度，不替代 `steady_tps`。流式 chunk 合并时 token 时间按整请求 token/chunk 比例估算，应披露此限制。

### Decode 结果交付

最终用一张纯英文、列宽对齐的 Markdown 表格交付所有 7 档并发的实际测试结果，格式如下。以下仅为格式模板，不是测试数据；无有效窗口时填 `N/A` 并说明原因。

| concurrency | valid_waves | window_s | tpot_p50 | steady_tps | concurrency/tpot | success/total |
| ----------- | ----------- | -------- | -------- | ---------- | ---------------- | ------------- |
| 1           | -           | -        | -        | -          | -                | -             |
| 2           | -           | -        | -        | -          | -                | -             |
| 4           | -           | -        | -        | -          | -                | -             |
| 8           | -           | -        | -        | -          | -                | -             |
| 16          | -           | -        | -        | -          | -                | -             |
| 32          | -           | -        | -        | -          | -                | -             |
| 64          | -           | -        | -        | -          | -                | -             |

字段口径：

- `valid_waves`：存在全部目标 client 同时成功生成的共同窗口的测量时段数 / 计划测量时段数，通常分母为 3；不要求每个请求成功，不再表示 client 的请求轮次。
- `window_s`：所有有效测量时段的共同窗口时长之和，单位秒；每个时段独立取全部目标 client 的窗口交集，再相加。
- `tpot_p50` 与 `steady_tps` 使用同一组共同有效窗口。对每个成功请求，将其裁剪后的 Decode 窗口与共同窗口相交，累加所有重叠片段的时长和其中的输出 token 数。该请求的稳态 TPOT = 重叠总时长 / 片段内 token 总数（秒/token），片段之间的空档不计时。汇集该并发档所有有效测量时段中贡献了 token 的请求，每个请求计一个样本，取中位数。整段请求的 TPOT 单独记为 `request_tpot`，用于分析单请求体验。
- `steady_tps`：所有有效测量时段共同窗口内所有目标 client 的输出 token 总数 / `window_s`，单位 tokens/s；按窗口时长加权，不直接平均各次 TPS。
- `concurrency/tpot` 用目标并发除以窗口内的 `tpot_p50` 估算吞吐（tokens/s）。它取决于请求 TPOT 的中位数，可能与按窗口总 token 数计算的 `steady_tps` 有差异。
- `success/total`：测量请求成功数 / 实际尝试总数，包含失败重试，不包含预热。

随表保留原始结果与 JSONL，包含各次测量的起止时间、成功窗口、实际输出长度、失败类型和重试数、`span_tps`。文字简要说明吞吐最高档位、共同窗口时长、失败情况，以及实际输出长度与目标 1024 tokens 的偏差。
