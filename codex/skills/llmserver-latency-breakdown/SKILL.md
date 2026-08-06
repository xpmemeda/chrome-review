---
name: llmserver-latency-breakdown
description: Analyze llmserver PREFILL/DECODE p.log and d.log files along two main axes: data statistics and latency statistics. Use it to report input length, KV cache hit length, PD transfer length, cache hit rate, and P-node latency breakdowns into frontend, dispatch, KV prepare, sp_d2d, embedding wait, compute, transfer, and d2h post-processing.
---

# Llmserver Latency Breakdown

## Inputs

Require both PREFILL and DECODE logs. Ask the user for paths if they did not provide them. Default to `p.log` and `d.log` in the current workspace only when those files exist.

Use local llmserver source if the user asks what a phase means. Prefer `~/workspace/byted/seed/llmserver` and cite exact files/lines after reading them.

## Quick Workflow

1. First produce data statistics for the requested request set: input length, KV cache hit length, cache hit rate, and PD transfer length.
2. Then produce latency statistics when requested or useful: the P-node latency breakdown, including frontend, dispatch, KV prepare, sp_d2d, embedding wait, compute, transfer, and d2h post-processing.
3. Run `scripts/analyze_pd_latency.py --p-log <p.log> --d-log <d.log> summary` when available to count complete requests and prompt categories before detailed reporting.
4. Run `scripts/analyze_pd_latency.py --p-log <p.log> --d-log <d.log> story-hits` only when the user asks about `Tell me a long story.` requests.
5. Run `scripts/analyze_pd_latency.py --p-log <p.log> --d-log <d.log> p-breakdown --story --cache-hit 13113` or the matching filters for P-node latency breakdown of a specific request class.
6. Explain that all latency values are milliseconds and that `d2h_post` is after the P-node critical path unless the user asks to include cache writeback.

## Definitions

Complete request: same `task_id` has an entry line (`call req` in p.log, `stream call req` in d.log) and a DONE line (`finish task ... TaskStatus.DONE`) in the same log. End-to-end complete request: complete in both logs.

Story request: a `vlm_processor.py:710` `New Chat Messages` line contains `Tell me a long story`. Repeated-hi request: that line contains `hi hi`.

Prompt length: use frontend `call add task <task_id>, prompt_token_len: <N>`.

KV hit length: use DECODE frontend response `finish return resp task <task_id> ... cache_hit_len=<N>` for user-facing decode hit statistics. For P-node internals, use PREFILL `allocate kv for task ... hit_length <N>` / `post_allocate_kv ... hit_length <N>`.

PD transfer length: use DECODE transfer worker `kv_cache.scatter_copy time ... seq [<begin>, <end>])` and report `<end> - <begin>` tokens when present. If scatter-copy seq ranges are missing, infer the transferred suffix from input length and KV hit length only when that matches nearby transfer logs; label it as inferred.

Data statistics: report input length, KV cache hit length, cache hit rate, and PD transfer length. This is the primary default table for recent-request summaries.

Latency statistics: report the P-node latency breakdown. Keep P-node critical path latency separate from post-DONE cache writeback unless the user asks to include writeback.

## P-Node Latency Breakdown

The script reports:

- `total`: `call req` to `send transfer result` on PREFILL. This is the P-node critical path used in prior analysis.
- `frontend`: `call req` to `DriverProxy send tasks`. Includes request verification/preprocess, task creation, shared-memory IPC creation, and handoff to driver proxy.
- `dispatch_to_kv`: `DriverProxy send tasks` to PREFILL compute `allocate kv for task ... hit_length`. Includes proxy registration, control-driver admission, queueing, and initial KV allocation/query.
- `kv_prepare`: PREFILL `allocate kv for task ... hit_length` to `post_allocate_kv ... allocate kv block`. Includes waiting for VLM embedding when present and post-allocation bookkeeping.
- `sp_d2d`: explicit `kv_cache.sp_d2d.copy time cost`. This is SP cache device-to-device copy of hit prefix KV into this request's device blocks, not model compute.
- `embed_wait`: PREFILL `allocate kv for task ... hit_length` to embedding worker `to input_embed_shm, is_finish: True`. This is normally almost equal to `kv_prepare` for VLM tasks because post-allocation bookkeeping is thin.
- `compute_wall`: `post_allocate_kv ... allocate kv block` to compute-driver DONE from `SignalSource.COMPUTE_DRIVER`. This is the critical-path wall-clock prefill compute window after KV/embedding readiness.
- `infer_exec_sum`: sum of `prefill tasks [...] execute time: <s>` lines for that task. This is cumulative inferencer worker execution/postprocess time across chunks or pipeline steps, so it can exceed `compute_wall` when stages overlap.
- `infer_exec_count`: number of `prefill tasks [...] execute time` records summed into `infer_exec_sum`.
- `transfer`: compute-driver DONE to `send transfer result`. This covers P-to-D KV transfer completion and notify/result handling.
- `d2h_post`: explicit `kv_cache.d2h.copy time cost`. This is post-DONE device-to-host cache writeback for future reuse and should be listed separately from critical-path total.

Important: columns are not always additive. `sp_d2d` can overlap with embedding work, `embed_wait` is a sub-interval inside `kv_prepare`, and `infer_exec_sum` is cumulative worker time rather than a wall-clock interval.

## Code Interpretation Hints

When mapping phases to code, search these source files:

- `llmserver/rpc/handler/base_handler.py`: `call`, `stream_call`, preprocessing, `call add task`, `Processing request`.
- `llmserver/driver/proxy/driver_proxy.py`: `DriverProxy send tasks` and ControlDriver selection.
- `llmserver/driver/loops/control_driver_loop.py`: `preprocess add req`, `send transfer result`.
- `llmserver/driver/kv_manager/kv_manager.py`: `first_allocate`, KV query/allocation, logged hit length.
- `llmserver/driver/loops/kv_allocate_loop.py`: compute allocate, embedding worker dispatch, `_post_allocate_kv`.
- `llmserver/processor/vlm/embedding/worker.py`: image/audio embedding copied into `input_embed_shm`.
- `bytedkvcache/manager/kvcache_manager.py` and `kvcache_worker.py`: `SPD2DCmdParam`, `D2HCmdParam`, `ops.sp_d2d`, d2h writeback.
- `llmserver/driver/loops/compute_driver_loop.py`: waiting queue, inferencer submission, postprocess DONE.
- `llmserver/driver/loops/kv_transfer_loop.py`: KV transfer and transfer result checks.

## Reporting Style

Start with the requested table. For recent-request summaries, default to the data statistics table: input length, KV cache hit length, hit rate, and PD transfer length. Add latency columns or a second latency table when the user asks about timing.

Briefly state the extraction口径. Mention boundary-truncated tasks separately: tasks that only have DONE but no entry, or entry but no DONE.

For latency tables, abbreviate task IDs only if the user does not need exact IDs; otherwise include full IDs. Always state whether `d2h_post` is included in total.
