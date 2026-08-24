---
name: images-edits-benchmark
description: 使用 ocean-benchmark 启动、验证并测试任何兼容 OpenAI `/v1/images/edits` 的服务，包括接收目标图、参考图和蒙版图的服务。用于测量图像编辑延迟、吞吐量、并发扩展、图像尺寸扩展、GPU 显存、量化或多 GPU 配置；不要用于通用 HTTP API 或不含性能测试的图像质量评估。
---

# 图像编辑性能测试

执行可复现的闭环性能测试，向用户提供有证据支撑的结果表和明确的服务最终状态。

## 必需的测试路径

凡端点为 `/v1/images/edits` 的正式性能测试，都必须使用 ocean-benchmark 的 `benchmark-closeloop.py --client images-edits`，无论服务实现是 vLLM、vLLM-Omni、flask-diffsynth 或其他兼容服务。直接 curl 或临时客户端仅可用于发现 schema 和冒烟验证，不得作为正式性能结果。

## 已知 flask-diffsynth 默认值

仅在确认服务为 flask-diffsynth 后将以下值作为可选默认值，不得套用到其他服务：

- 远程别名：`H20`
- 远程仓库：`~/workspace/github/chrome-review`
- 服务脚本：`diffuser-models/flask-diffsynth.py`
- 模型根目录：`~/workspace/models`
- Python 环境：`~/envs/cpython-3.12/diffsynth-e/bin/activate`
- 服务端点：`http://127.0.0.1:8512/v1/images/edits`
- 本地 benchmark 仓库：`~/workspace/byted/ocean/ocean-benchmark`
- 远程暂存目录：`~/workspace/benchmarks/ocean-benchmark`
- 默认 GPU：GPU 0

向用户提问前，先从仓库或远程机器发现缺失值。保留用户明确指定的模型、GPU、step、端点、提示词、图像尺寸、并发数、请求数、warmup 和 timeout。

## 远程网络设置

执行需要联网的远程命令前，检测机器类型，并在同一 shell 中导出对应代理变量：

- `env | grep -q '^[^=]*MERLIN[^=]*='` 成功时为 MERLIN：HTTP/HTTPS 使用 `http://sys-proxy-rd-relay.byted.org:8118`，并设置标准 ByteDance `NO_PROXY` 域名。
- 主机名以 `di-` 开头时为火山引擎：HTTP/HTTPS 使用 `http://100.66.18.103:3128`，同时设置小写变量、`NO_PROXY=localhost,127.0.0.1,mirrors.ivolces.com,pypi.org,files.pythonhosted.org,pypi.python.org` 和 `PIP_INDEX_URL=https://mirrors.ivolces.com/pypi/simple`。

若网络请求失败且可能由代理造成，仅针对该命令清除全部 HTTP/HTTPS/ALL 代理变量后重试一次；不要在后续 shell 中永久取消代理。

## 工作流程

1. **先检查，再改变状态**
   - 确认远程主机名、GPU 型号/利用率/显存、端口所属进程、仓库状态、模型路径、服务实现、OpenAPI schema 或 multipart 字段以及必需 CLI 参数。
   - 除非用户将现有进程明确纳入范围，否则不要停止它。只能终止当前任务启动的服务；无关服务占用端口时向用户确认。
2. **在远程准备 benchmark**
   - 已有且版本合适时复用远程副本。
   - 否则传输本地 benchmark 仓库，并排除 `.git`、缓存、日志和 macOS 元数据。不要仅为远程暂存而修改本地 checkout。
3. **启动目标服务**
   - 激活指定环境，并在服务 shell 中设置 `CUDA_VISIBLE_DEVICES`。
   - 将 stdout/stderr 重定向到名称唯一的 `/tmp` 日志，并记录 PID。
   - 使用实现专属的量化和并行参数。flask-diffsynth 搭配内置 encoder 的双 TorchAO W8A8 使用 `--fp8-dit --fp8-text-encoder`；其他服务必须检查其 CLI 和实时配置。
   - flask-diffsynth 显式提供 `--text-encoder-model` 时必须保留，否则允许服务从 `--model` 的 `text_encoder/` 和 `tokenizer/` 加载。
4. **等待就绪并验证配置**
   - 端点监听后才能开始测试。
   - 从启动日志或实时配置验证量化和并行设置。flask-diffsynth FP8 必须确认每个目标组件都报告量化 Linear 数量；仅启动成功不足以证明配置正确。
   - 可用时记录模型加载后的 CUDA allocated/reserved 显存。
5. **按用户指定尺寸与并发构建矩阵**
   - 宽高必须是模型所需除数的倍数；该 FLUX.2 pipeline 通常要求 16 的倍数。
   - 新图像 shape 使用独立 warmup，排除编译/自动调优时间。
   - 比较组必须使用相同提示词、step、请求数和其他设置。
   - 每个图像角色映射到服务真实 multipart schema：目标通常为 `image`，参考图常为 `reference_image`，蒙版可能为 `mask` 或 `mask_image`。请求包含三种图时，每次都发送三份真实文件，不得用生成占位图替代。
   - 通过 `--images-edits-extra-fields` 传递 step、CFG、model 和 response format 等控制项，并记录准确映射；例如某服务可能需同时设置 `guidance_scale=0` 和 `true_cfg_scale=0` 才表示关闭 CFG。
6. **逐组串行运行闭环测试**，替换以下命令中的请求参数：

```bash
python benchmark-closeloop.py \
  -d synthetic-vlm \
  --client images-edits \
  --base-url http://127.0.0.1:8512/v1/images/edits \
  --synthetic-vlm-num-prompt-tokens 16 \
  --synthetic-vlm-image-width WIDTH \
  --synthetic-vlm-image-height HEIGHT \
  --images-edits-prompt "PROMPT" \
  --images-edits-size WIDTHxHEIGHT \
  --images-edits-reference-image /path/to/reference.png \
  --images-edits-mask-image /path/to/mask.png \
  --images-edits-reference-field reference_image \
  --images-edits-mask-field mask_image \
  --images-edits-extra-fields '{"num_inference_steps":4,"guidance_scale":0}' \
  --concurrency-sweep CONCURRENCY \
  --num-requests-sweep REQUESTS \
  -w WARMUPS \
  --timeout TIMEOUT
```

7. **验证每一组**
   - 成功请求数必须符合预期；失败请求应明确报告，不能从平均值中悄悄移除。
   - 从 benchmark 输出获取 QPS 和 E2E average/P50/P90/P99。
   - 检查服务错误日志，并在有记录时收集请求后的 CUDA allocated/reserved/peak 显存。
8. **保留证据并说明最终状态**
   - 报告 benchmark 和服务日志路径、服务 PID、端口以及服务是否仍在运行。
   - 除非用户要求停止，否则成功完成测试的服务保持运行。

## 报告

先给出紧凑表格，至少包含图像尺寸、并发数、平均延迟、P50、P90、吞吐量（images/s）、成功率和请求数。用户只要求少量指标时突出这些指标，但仍保留足以解释结果的证据。

明确区分：

- 首次请求编译/自动调优时间与稳态延迟；
- 进程/PyTorch 显存与整卡显存，特别是其他容器或进程存在基线占用时；
- 排队并发与真实模型并行执行。`flask-diffsynth.py` 通常只有一个 GPU worker，因此并发 2 可能主要增加延迟而几乎不提高吞吐；
- 实测事实与关于 kernel 行为的推断。

不得把历史性能数字当作当前结果；必须测试当前任务要求的实时服务配置。
