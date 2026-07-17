# Benchmark

这个目录用于逐步集成不同类型的模型压测。当前支持 OpenAI-compatible chat/completions 文本测试、VLM 图文测试、ModelApi / Ark chat/completions 测试、Ultraman gRPC 流式测试，以及 diffusion multipart 服务测试。

当前有两个入口：

- `benchmark-closeloop.py`：闭环压测，用来找极限吞吐。
- `benchmark-openloop.py`：开环压测，用来观察服务端在指定全局 QPS 下的延迟和错误表现。

## 闭环测试

闭环测试会始终保持最多 `concurrency` 个请求在路上。一个请求结束后，立刻补下一个请求，直到完成当前测试点配置的统计请求数。

闭环测试回答的问题是：服务端在不同 in-flight 并发下，吞吐最高能到多少，延迟什么时候开始明显恶化。

关键参数：

- `--concurrency-sweep`：闭环测试的全局 in-flight 并发值列表。
- `--num-requests-sweep`：每个并发点统计多少个请求，需要和 `--concurrency-sweep` 等长。
- `-w, --warmup-requests`：每个并发点正式统计前的预热请求数，不计入最终结果。
- warmup 如果有 client 失败，会只对失败 client 自动重试，最多 3 轮。
- `--jsonl`：可选，把每个测试点的完整 summary 追加写入 JSONL。

文本闭环示例：

```bash
python3 benchmark/benchmark-closeloop.py \
  -d synthetic-txt \
  --client openai \
  --base-url http://localhost:8000/v1 \
  --model your-text-model \
  --synthetic-txt-num-prompt-tokens 256 \
  --concurrency-sweep 1 2 4 8 16 \
  --num-requests-sweep 64 64 64 64 64 \
  -w 8 \
  --max-tokens 64
```

VLM 闭环示例：

```bash
python3 benchmark/benchmark-closeloop.py \
  -d synthetic-vlm \
  --client openai \
  --base-url http://localhost:8009/v1 \
  --model your-vlm-model \
  --synthetic-vlm-num-prompt-tokens 128 \
  --synthetic-vlm-image-width 448 \
  --synthetic-vlm-image-height 448 \
  --concurrency-sweep 1 2 4 8 16 \
  --num-requests-sweep 64 64 64 64 64 \
  -w 8 \
  --max-tokens 64
```

## 开环测试

开环测试会先生成一份全局请求到达计划，然后按计划发请求。`--qps-sweep` 表示服务端整体看到的请求到达率，不是每个 client 的 QPS。

关键参数：

- `--arrival`：全局请求到达过程，可选 `poisson` 或 `constant`。
- `--qps-sweep`：服务端整体请求到达率，支持单个 QPS 或多个 QPS 点。
- `--arrival-seed`：生成 Poisson 到达计划的随机种子；Poisson 时间戳会后验缩放到目标整体 QPS。
- `-c, --concurrency`：客户端侧最大 in-flight 请求数保护。
- `--num-requests-sweep`：每个 QPS 点统计多少个请求，需要和 `--qps-sweep` 等长。
- `-w, --warmup-requests`：每个 QPS 点正式统计前的预热请求数。
- 开环 warmup 会按当前 `--qps-sweep` 点的 QPS 间隔逐个启动请求，避免 client 很多时形成 burst。
- warmup 如果有 client 失败，会只对失败 client 自动重试，最多 3 轮。

文本开环示例：

```bash
python3 benchmark/benchmark-openloop.py \
  -d synthetic-txt \
  --client openai \
  --base-url http://localhost:8000/v1 \
  --model your-text-model \
  --synthetic-txt-num-prompt-tokens 256 \
  --arrival poisson \
  --qps-sweep 1 2 4 8 16 \
  -c 128 \
  --num-requests-sweep 64 64 64 64 64 \
  -w 8 \
  --max-tokens 64
```

VLM 开环示例：

```bash
python3 benchmark/benchmark-openloop.py \
  -d synthetic-vlm \
  --client openai \
  --base-url http://localhost:8009/v1 \
  --model your-vlm-model \
  --synthetic-vlm-num-prompt-tokens 128 \
  --synthetic-vlm-image-width 448 \
  --synthetic-vlm-image-height 448 \
  --arrival poisson \
  --qps-sweep 8 \
  -c 128 \
  --num-requests-sweep 64 \
  -w 8 \
  --max-tokens 64
```

ModelApi 文本开环示例：

```bash
python3 benchmark/benchmark-openloop.py \
  -d synthetic-txt \
  --client modelapi \
  --model your-model-name \
  --modelapi-env your-ppe-env \
  --synthetic-txt-num-prompt-tokens 256 \
  --arrival poisson \
  --qps-sweep 8 \
  -c 128 \
  --num-requests-sweep 64 \
  -w 8 \
  --max-tokens 64
```

Ark 文本开环示例：

```bash
python3 benchmark/benchmark-openloop.py \
  -d synthetic-txt \
  --client ark \
  --api-key your-ark-api-key \
  --model your-ark-endpoint \
  --synthetic-txt-num-prompt-tokens 256 \
  --arrival poisson \
  --qps-sweep 8 \
  -c 128 \
  --num-requests-sweep 64 \
  -w 8 \
  --max-tokens 64
```

`--qps-sweep`、`--concurrency-sweep` 和 `--num-requests-sweep` 都使用空格分隔多个值：

```bash
--qps-sweep 8
--qps-sweep 1 2 4 8
--num-requests-sweep 20 30 60 100
--concurrency-sweep 1 2 4 8
```

## Dataset 参数

当前支持四个 dataset：

- `-d synthetic-txt`：普通文本 synthetic dataset，只发送 OpenAI chat 文本消息。
- `-d synthetic-vlm`：VLM synthetic dataset，发送文本加内存生成 PNG 图片。
- `-d jsonl`：从 JSONL 文件加载文本 messages。
- `-d omni-multi-message`：从 omni multi-message JSON 模板加载完整 messages，并按请求变异部分字段。

Synthetic text 参数：

- `--synthetic-txt-num-prompt-tokens`：每个请求的近似 prompt token 数，必填。
- `--synthetic-txt-prompt-prefix-hit-rate`：prompt 前缀复用比例，范围 `[0, 1]`；会把这部分共享前缀作为独立的 `system` message。
- `--text-seed`：`-d synthetic-txt` 的 prompt seed，默认 `0`。

VLM synthetic 参数：

- `--synthetic-vlm-num-prompt-tokens`：每个请求的近似 prompt token 数，必填。
- `--synthetic-vlm-prompt-prefix-hit-rate`：prompt 前缀复用比例，范围 `[0, 1]`；会把这部分共享前缀作为独立的 `system` message。
- `--synthetic-vlm-image-width`：合成图片宽度，`-d synthetic-vlm` 必填。
- `--synthetic-vlm-image-height`：合成图片高度，`-d synthetic-vlm` 必填。
- `--synthetic-vlm-image-seed`：合成图片的基础随机种子；不传时随机生成。
- synthetic prompt 会默认加入短输出长度指令，目标为 `--min-tokens` 或 `--max-tokens`；该指令会计入对应 `--synthetic-*-num-prompt-tokens` 的 suffix 预算。

`-d synthetic-vlm` 会基于同一张基础图片 patch 少量像素，避免每个请求完全相同。不同 sweep 点使用不同 request id 区间，减少前一个测试点对后一个测试点的缓存影响。

JSONL dataset 参数：

- `--dataset-path`：JSONL 文件路径，`-d jsonl` 必填。

JSONL 每行支持以下文本格式之一：

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."}]}
```

```json
{"system_prompt":"...","prompt":"..."}
```

```json
{"prompt":"..."}
```

```json
"plain text prompt"
```

如果请求数超过 JSONL 行数，dataset 会按行循环复用样本。

Warmup 不会消耗 JSONL 文件中的正式样本。JSONL warmup 会保留第一条样本里的 `system` / `developer` 消息，并生成独立的 `warmup_unique_id` user 消息。JSONL 中的 `system` 消息会保留在 `StdChatApiRequest["messages"]` 里，client 发送时仍然作为独立的 `system` message。

Omni multi-message dataset 参数：

- `--omni-template`：请求体模板路径，默认读取 `~/workspace/ocean/service_shell/benchmark/omni_multi_message.json`。
- `--omni-seed`：控制 app list shuffle 和图片噪声的随机种子，默认 `0`。

`-d omni-multi-message` 只会从模板中读取 `messages` 并返回 `{"messages": ...}`。每个请求会改写倒数第二条 user 消息里的 `The available app list` 顺序，并把命中缓存的图片 URL 替换成 base64；最后一张图片会额外追加随机噪声。`model`、`max_tokens`、`temperature`、`top_p` 等请求参数仍由 benchmark CLI 参数控制。

## Client 参数

当前支持六个 client：

- `--client mock`：本地空响应 client，用于快速验证 benchmark 框架本身。
- `--client openai`：OpenAI-compatible `/chat/completions` 流式接口，支持 `-d synthetic-txt` 和 `-d synthetic-vlm`。
- `--client modelapi`：ModelApi `/chat/completions` 流式接口，使用 OpenAI SDK，支持 `-d synthetic-txt` 和 `-d synthetic-vlm`。
- `--client ark`：Ark `/chat/completions` 流式接口，使用 OpenAI SDK，支持 `-d synthetic-txt` 和 `-d synthetic-vlm`。
- `--client diffusion`：multipart diffusion 服务，只支持带图片的 `-d synthetic-vlm`。
- `--client ultraman`：Ultraman gRPC `StreamingCall` 流式接口，支持 `-d synthetic-txt`、`-d synthetic-vlm` 和 `-d jsonl`。

OpenAI client 参数：

- `--base-url`：OpenAI-compatible API base URL，例如 `http://localhost:8000/v1`。
- `--api-key`：API key，默认 `dummy`。
- `--model`：模型 ID，必填。
- `--timeout`：单请求网络超时时间，单位秒。
- `--max-tokens`：传给 `/chat/completions` 的 `max_tokens`；Ark client 会改用 `max_completion_tokens`，用于约束 reasoning 和最终回答的总生成长度。
- `--min-tokens`：通过 `extra_body.min_tokens` 传给服务端。
- `--temperature`：采样温度。
- `--top-p`：可选 top-p 采样参数。
- `--extra-body`：JSON 对象，会合并进请求 `extra_body`。

ModelApi client 参数：

- `--model`：ModelApi model name，必填。
- `--base-url`：可选，默认 `https://device-intelligence.bytedance.net/api/v1`。
- `--modelapi-env`：可选，作为 `x-tt-env` header；不传时 `x-use-ppe` 为 `0`。
- `--timeout`：单请求网络超时时间，单位秒。
- `--max-tokens`、`--min-tokens`、`--temperature`、`--top-p`、`--extra-body`：含义同 OpenAI client。

Ark client 参数：

- `--api-key`：Ark API key，必填。
- `--model`：Ark endpoint / model ID，必填。
- `--base-url`：可选，默认 `https://ark.cn-beijing.volces.com/api/v3`。
- `--timeout`：单请求网络超时时间，单位秒。
- `--max-tokens`、`--min-tokens`、`--temperature`、`--top-p`、`--extra-body`：含义同 OpenAI client。

Diffusion client 参数：

- `--base-url`：完整 diffusion `/generate` endpoint。
- `--diffusion-style`：固定 style；不传时使用 dataset prompt。
- `--diffusion-seed`：基础 seed，request index 会加到这个 seed 上。
- `--diffusion-steps`：multipart `steps` 字段。
- `--diffusion-extra-fields`：JSON 对象，会合并进 multipart form 字段。

Ultraman client 参数：

- `--base-url`：可选，`host:port` 或 `grpc://host:port`，默认 `127.0.0.1:50050`。
- `--ultraman-host`、`--ultraman-port`：可选，显式覆盖 `--base-url` 里的 host / port。
- `--ultraman-proto-path`：包含 `llmserver.proto.ultraman_pb2` 的 Python import 路径，默认使用仓库内置的 proto 模块。
- `--model`：可选，写入 Ultraman `InferenceRequest.model_name`；不传时为空串。
- `--max-tokens`：Ultraman server 会保留 8 个输出 token，client 实际写入 `max_new_tokens = value + 8`，并打印 warning。
- `--min-tokens`：如果指定，client 实际写入 `min_new_tokens/min_tokens = value + 8`，但 Ultraman server 不保证该参数生效，会打印 warning；synthetic prompt 里的输出长度指令仍使用用户传入值。
- `--temperature`、`--top-p`：分别写入 `temperature`、`top_p`。
- `--ultraman-top-k`：写入 `top_k`，默认 `1`。
- `--ultraman-repetition-penalty`：写入 `repetition_penalty`，默认 `1.1`。

Ultraman VLM 闭环示例：

```bash
python3 benchmark/benchmark-closeloop.py \
  -d synthetic-vlm \
  --client ultraman \
  --base-url 127.0.0.1:50050 \
  --synthetic-vlm-num-prompt-tokens 128 \
  --synthetic-vlm-image-width 448 \
  --synthetic-vlm-image-height 448 \
  --concurrency-sweep 1 2 4 8 \
  --num-requests-sweep 64 64 64 64 \
  -w 8 \
  --max-tokens 64
```

## 输出指标

每个测试点会打印：

- `ok/errors`：成功和失败请求数。
- `elapsed`：统计总耗时。
- `rps`：成功请求完成速率。
- `output_tokens/s`：本地使用 `tiktoken` 统计的输出 token 吞吐。
- `chunks/s`：流式输出 chunk 吞吐。
- `TTFT avg/p50/p90/p99`：从发出请求到收到第一个流式事件的时间。
- `ITL avg/p50/p90/p99`：相邻流式事件之间的间隔。
- `E2E avg/p50/p90/p99`：端到端请求耗时。
- `avg_output_tokens`：每个成功请求的平均输出 token 数。
- `avg_output_chars`：每个成功请求的平均流式输出字符数。

每个测试点结束后还会按 client 对象打印局部统计表：该 client 承载请求的 `TTFT avg/p50/p90`、`E2E avg/p50/p90`、`tpot`、`avg_o200k_toks`、`avg_server_toks` 和 `avg_out_chars`。
其中 `tpot` 用 `(e2e_avg - ttft_avg) / avg_output_tokens` 计算，优先使用服务端返回 token 数，缺失时使用本地 `o200k` token 数；`avg_o200k_toks` 本地使用 `tiktoken` 统计，不依赖服务端 usage；`avg_server_toks` 来自服务端返回的输出 token 数，没有时显示 `N/A`。
OpenAI/Ark 兼容接口会同时统计 thinking 模型的 `reasoning_content` 和最终可见的 `content` 文本。
字符列统计实际流式输出文本长度。运行文本生成 client 时需要本地 Python 环境可导入 `tiktoken`。

如果传入 `--log-path benchmark.log`，每个请求的明细输出会以 JSONL 追加写入 `benchmark.log.detail.log`。
每行包含延迟、图片大小、响应 ID、日志 ID、TPOT 和完整 `output_text` 文本；`ttft` / `tpot` 固定保留 3 位小数，方便直接查看时对齐。

OpenAI 和 ModelApi client 仍会设置 `stream_options.include_usage=true`，但 benchmark 的 output token 指标以本地 `tiktoken` 统计为准。
