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
- 模型的架构，目前只支持 M12、M13、M14。
- Decode server 是否已针对 benchmark 前缀请求启用固定接纳率，以及具体接纳率。
  - 若已启用，则正常进入 Prefill 和 Decode 测试。
  - 若未启用，则询问用户是否需要自行启用接纳率改造，若用户拒绝，则不进行测试，若用户同意，则索要 Prefill 和 Decode 的服务地址。

## Decode 服务的接纳率改造

1. 操作服务实例（可以用 remote-dev-agent 技能），把本地 ~/workspace/byted/seed.llmserver 仓库 xiongpeng.123/1.0.0.4676-patch 分支下的 xiongpeng.123 写的 commit 应用到服务器上。如果有冲突请自行解决。
2. 重启服务器：`bernard service restart`。
3. 等待服务器重启完成之后，开始 Prefill & Decode 测试。

## Prefill 测试

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

## Decode 测试

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

## 结果交付

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

- 根据测试结果给出服务的 Prefill-Decode 数量配比表格。像下面这样。

| uncached_tokens | uncached-prefill-TPS | prefill-QPS | output_tokens | decode-TPS | decode-QPS |
| --------------- | -------------------- | ----------- | ------------- | ---------- | ---------- |
| 5.74k           | 33748.93             | 5.881       | 290           | 1729.0     | 5.962      |
