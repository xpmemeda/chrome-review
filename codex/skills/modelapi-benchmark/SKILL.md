---
name: modelapi-benchmark
description: 使用 ocean.ocean-benchmark 对 ModelAPI 的大模型推理服务做性能压测。适用于 PD 分离场景、PD 混合场景。
---

# ModelAPI 性能测试

在 `~/workspace/byted/ocean.ocean-benchmark` 中执行测试，有两类测试场景。

1. PD 分离场景：目标是分别测量 Prefill 和 Decode 节点，保留可复查的命令、日志与结果，并拒绝不满足前置条件的测试。
2. PD 混合场景：目标是测量目标机器的整体性能，保留可复查的命令、日志与结果，并拒绝不满足前置条件的测试。

有几个注意事项：

1. 测之前先向用户确认场景是 PD 分离还是 PD 混合。
2. 测试如果有失败，需要向用户包括失败的类型。目前已知的有如下几种。
   只有 code = 1 的错误是可以容忍的，其他错误可能暗示着服务已经被压挂了，需要重点提醒用户。
  - rpc error: code = 1 desc = Cancelled by backend [biz error]
  - rpc error: code = 14 desc = Socket closed [biz error]

## PD 分离场景

### 需要用户提供的信息

开始任何压测前，检查用户是否明确提供以下信息：

- ModelAPI 的 `--model`。
- ModelAPI 的 `--modelapi-env`。
- 模型的架构，目前只支持 M12、M13、M14。
- Decode server 是否已针对 benchmark 前缀请求启用固定接纳率。

### Prefill 测试

使用 `modelapi-rewrite-vlm-streaming.py`，根据模型架构选择配置：

- M12: `modelapi-rewrite-vlm-streaming-m12-30b.json`
- M13: `modelapi-rewrite-vlm-streaming-m13-12b.json`
- M14: `modelapi-rewrite-vlm-streaming-m14-6b.json`

另外，配置文件中的有些部分需要更改：

- model: 改成用户指定的模型名。
- modelapi-env: 改成用户指定的环境。
- min-tokens: 固定为 1。
- max-tokens: 固定为 1。
- process: 从 1 开始，逐步增加，直到 uncached-prefill-tps 不发生明显变化。
- concurrency-per-process: 固定为 4。
- iterations: 固定为 3。

```bash
python modelapi-rewrite-vlm-streaming.py --config /path/to/config.json
```

### Decode 测试

使用 `benchmark-dec.py`。该脚本为每个 client 建立独立 session：warmup 使用 `system -> user`，之后每轮累计追加 `assistant("hi") -> user("hi")`。
测试的参数如下：

- model: 用户指定的模型名。
- modelapi-env: 用户指定的环境。
- min-tokens: 从选取的配置文件中读取。
- max-tokens: 从选取的配置文件中读取。
- context-len: 固定为 25000。

参考命令：

```bash
python benchmark-dec.py --context-len 25000 --client modelapi --model omniagent_m14_6b_0824 --modelapi-env ppe_model_center --concurrency-sweep 1 16 32 --num-requests-sweep 8 64 128 --warmup-requests 1 --min-tokens 442 --max-tokens 442 --timeout 180
```

### 结果交付

- Prefill 测试脚本输出的原始表格，不同 process 测试的表格都给出来。像下面这样。

| scope                   | qps   | input_tokens | cached_tokens | cache_hit_rate | uncached_tps | avg_output_tokens | decode_tps |
| ----------------------- | ----- | ------------ | ------------- | -------------- | ------------ | ----------------- | ---------- |
| theoretical(prefix)     | 5.881 | 47012688     | 31752144      | 67.54%         | 47942.095    | N/A               | N/A        |
| theoretical(stream_llm) | 5.881 | 47012688     | 36270000      | 77.15%         | 33748.926    | N/A               | N/A        |
| actual(server)          | 5.881 | 47714368     | 0             | 0.00%          | 149898.114   | 1.0               | 119.470    |

- Decode 的测试脚本输出的原始表格。像下面这样。

| concurrency | qps   | tpot   | decode_tps | success_rate | success/total |
| ----------- | ----- | ------ | ---------- | ------------ | ------------- |
| 1           | 0.294 | 0.0108 | 85.4       | 100.00%      | 8/8           |
| 16          | 2.814 | 0.0136 | 816.2      | 100.00%      | 64/64         |
| 32          | 5.962 | 0.0148 | 1729.0     | 100.00%      | 128/128       |

- 上面测试都是 1P1D 的结果，根据测试数据给出服务的 Prefill-Decode 数量配比表格。像下面这样。

| uncached_tokens | uncached-prefill-TPS | prefill-QPS | output_tokens | decode-TPS | decode-QPS |
| --------------- | -------------------- | ----------- | ------------- | ---------- | ---------- |
| 5.74k           | 33748.93             | 5.881       | 290           | 1729.0     | 5.962      |

## PD 混合场景

### 需要用户提供的信息

- ModelAPI 的 `--model`。
- ModelAPI 的 `--modelapi-env`。
- 测试数据集。
- Session 模式：`request` 或 `client`。

### 测试

参考下面这条命令：

```bash
python benchmark-closeloop.py -d messages-jsonl --dataset-path ~/workspace/datasets/cot-summary/202608071750-messages-1000.jsonl --client modelapi --modelapi-session-mode request --model omniagent_m12_2b5_0905 --modelapi-env ppe_model_center --concurrency-sweep 8 --num-requests-sweep 128 -w 1
```

- 使用闭环测试，并发数 `--concurrency-sweep` 从 1 开始翻倍增加，直到 TPS 不再明显增加。
- `--num-requests-sweep` 取值暂且定为 `--concurrency-sweep` 的 16 倍。如果发现测试很慢，可以适当缩小这个倍数。

### 结果交付

输出 `benchmark-closeloop.py` 最后打印的原始表格即可。