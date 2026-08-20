---
name: images-edits-benchmark
description: Start, validate, and benchmark OpenAI-compatible /v1/images/edits services backed by flask-diffsynth on remote GPU development machines. Use when asked to measure image-edit latency, throughput, concurrency scaling, image-size scaling, GPU memory, or FP8 configurations with ocean-benchmark. Do not use for generic HTTP APIs or image-quality evaluation without performance testing.
---

# Images Edits Benchmark

Run a reproducible closed-loop benchmark and leave the user with an evidence-backed result table and a known service state.

## Known environment defaults

Treat these as defaults, not overrides of user-provided values:

- Remote alias: `H20`
- Remote repository: `~/workspace/github/chrome-review`
- Service script: `diffuser-models/flask-diffsynth.py`
- Model root: `~/workspace/models`
- Python environment: `~/envs/cpython-3.12/diffsynth-e/bin/activate`
- Service endpoint: `http://127.0.0.1:8512/v1/images/edits`
- Local benchmark repository: `~/workspace/byted/ocean/ocean-benchmark`
- Remote benchmark staging directory: `~/workspace/benchmarks/ocean-benchmark`
- Default GPU: GPU 0

Discover missing values from the repository or remote machine before asking the user. Preserve explicit model, GPU, step, endpoint, prompt, image-size, concurrency, request-count, warmup, and timeout choices.

## Remote network setup

Before remote commands that access the network, detect the machine and export the matching proxy variables in the same shell:

- MERLIN when `env | grep -q '^[^=]*MERLIN[^=]*='`: use `http://sys-proxy-rd-relay.byted.org:8118` for HTTP and HTTPS, with the standard ByteDance `NO_PROXY` domains.
- Volcano Engine when the hostname starts with `di-`: use `http://100.66.18.103:3128` for HTTP and HTTPS; set lowercase variants, `NO_PROXY=localhost,127.0.0.1,mirrors.ivolces.com,pypi.org,files.pythonhosted.org,pypi.python.org`, and `PIP_INDEX_URL=https://mirrors.ivolces.com/pypi/simple`.

If a network request fails and the proxy may be responsible, retry that command once with all HTTP/HTTPS/ALL proxy variables cleared. Do not unset proxies for the rest of the shell.

## Workflow

1. Inspect before changing state:
   - Confirm the remote hostname, GPU model/utilization/memory, port 8512 owner, repository status, model paths, and required CLI flags.
   - Do not stop an existing process unless the user placed it in scope. Only terminate a service started during the current task, or obtain user direction when an unrelated service owns the port.
2. Make the benchmark available remotely:
   - Reuse a suitable remote copy when present and current.
   - Otherwise transfer the local benchmark repository while excluding `.git`, caches, logs, and macOS metadata. Do not modify the user's local benchmark checkout merely to stage it.
3. Start the requested service:
   - Activate the specified environment and set `CUDA_VISIBLE_DEVICES` in the service shell.
   - Redirect stdout/stderr to a uniquely named `/tmp` log and record the PID.
   - For double TorchAO W8A8 with the bundled encoder, the relevant flags are `--fp8-dit --fp8-text-encoder`.
   - If `--text-encoder-model` is supplied, preserve it; otherwise allow the service to load `text_encoder/` and `tokenizer/` from `--model`.
4. Wait for readiness and verify configuration:
   - Require the endpoint to listen before benchmarking.
   - For FP8, check that startup logs report quantized Linear counts for every requested component. A successful start alone is insufficient evidence of the requested quantization.
   - Record model-load CUDA allocated/reserved memory when available.
5. Build the benchmark matrix from the user's requested image sizes and concurrency levels:
   - Image width and height should be multiples of the model's required division factor, normally 16 for this FLUX.2 pipeline.
   - Use independent warmups for a new image shape so compilation/autotuning is excluded from steady-state measurements.
   - Use the same prompt, steps, request count, and other settings across comparison groups.
6. Run closed-loop benchmarks serially, one group at a time. Use this command shape and substitute requested values:

```bash
python benchmark-closeloop.py \
  -d synthetic-vlm \
  --client images-edits \
  --base-url http://127.0.0.1:8512/v1/images/edits \
  --synthetic-vlm-num-prompt-tokens 16 \
  --synthetic-vlm-image-width WIDTH \
  --synthetic-vlm-image-height HEIGHT \
  --images-edits-prompt "PROMPT" \
  --concurrency-sweep CONCURRENCY \
  --num-requests-sweep REQUESTS \
  -w WARMUPS \
  --timeout TIMEOUT
```

7. Validate each group:
   - Require the expected number of successful requests and report failures rather than averaging them away.
   - Capture achieved QPS and E2E average/P50/P90/P99 from the benchmark output.
   - Check service logs for errors and collect post-request CUDA allocated/reserved/peak memory when logged.
8. Preserve evidence and final state:
   - Report benchmark and service-log paths, service PID, port, and whether the service remains running.
   - Leave a successfully benchmarked service running unless the user asks to stop it.

## Reporting

Lead with a compact table containing image size, concurrency, average latency, P50, P90, throughput in images/s, success rate, and request count. If the user requests a narrower metric set, emphasize it while retaining enough evidence to interpret the result.

Explicitly distinguish:

- First-request compile/autotuning time from steady-state latency.
- Process/PyTorch memory from whole-GPU memory when another container or process owns a baseline allocation.
- Queueing concurrency from actual parallel model execution. `flask-diffsynth.py` commonly has one GPU worker, so concurrency 2 may mostly increase latency with little throughput gain.
- Measured facts from inferences about kernel behavior.

Do not reuse historical performance numbers as current results. Benchmark the live service configuration requested in the current task.
