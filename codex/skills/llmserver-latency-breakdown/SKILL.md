---
name: llmserver-latency-breakdown
description: 从数据统计和延迟统计两个维度分析 llmserver PREFILL/DECODE 的 p.log 与 d.log，报告输入长度、KV cache 命中长度、PD 传输长度、命中率，以及 frontend、dispatch、KV prepare、sp_d2d、embedding wait、compute、transfer 和 d2h 后处理等 P 节点延迟分解。
---

# llmserver 延迟分解

## 输入

必须同时提供 PREFILL 和 DECODE 日志。用户未提供路径时应询问；仅当当前工作区确实存在 `p.log` 和 `d.log` 时才默认使用它们。

用户询问某阶段含义时，使用本地 llmserver 源码解释。优先读取 `~/workspace/byted/seed/llmserver`，并引用实际读取过的准确文件和行号。

## 快速流程

1. 先对目标请求集统计输入长度、KV cache 命中长度、命中率和 PD 传输长度。
2. 用户要求或确有帮助时，再统计 P 节点延迟分解：frontend、dispatch、KV prepare、sp_d2d、embedding wait、compute、transfer 和 d2h 后处理。
3. 可用时运行 `scripts/analyze_pd_latency.py --p-log <p.log> --d-log <d.log> summary`，在详细报告前统计完整请求数和提示词类别。
4. 仅当用户询问 `Tell me a long story.` 请求时运行 `story-hits`。
5. 针对特定请求类别的 P 节点分解，运行 `p-breakdown --story --cache-hit 13113` 或使用相应筛选条件。
6. 明确所有延迟单位均为毫秒；除非用户要求把 cache writeback 纳入统计，否则 `d2h_post` 位于 P 节点关键路径之后。

## 定义

- **完整请求**：同一 `task_id` 在同一日志中既有入口行（p.log 的 `call req` 或 d.log 的 `stream call req`），又有 `finish task ... TaskStatus.DONE`。端到端完整请求需在两份日志中都完整。
- **Story 请求**：`vlm_processor.py:710` 的 `New Chat Messages` 行包含 `Tell me a long story`。Repeated-hi 请求则包含 `hi hi`。
- **提示词长度**：取 frontend 的 `call add task <task_id>, prompt_token_len: <N>`。
- **KV 命中长度**：面向用户的 decode 命中统计使用 DECODE frontend 响应 `finish return resp task <task_id> ... cache_hit_len=<N>`；P 节点内部统计使用 PREFILL 的 `allocate kv for task ... hit_length <N>` 或 `post_allocate_kv ... hit_length <N>`。
- **PD 传输长度**：取 DECODE transfer worker 的 `kv_cache.scatter_copy time ... seq [<begin>, <end>])`，计算 `<end> - <begin>`。缺少 seq 范围时，仅在输入长度与 KV 命中长度之差符合附近传输日志时推断传输后缀，并明确标为推断值。
- **数据统计**：输入长度、KV cache 命中长度、命中率和 PD 传输长度。这是最近请求摘要的默认主表。
- **延迟统计**：P 节点延迟分解。除非用户要求，否则将关键路径与 DONE 后的 cache writeback 分开。

## P 节点延迟分解

- `total`：PREFILL 中从 `call req` 到 `send transfer result`，即 P 节点关键路径。
- `frontend`：从 `call req` 到 `DriverProxy send tasks`，包含请求校验/预处理、任务创建、共享内存 IPC 创建及交给 driver proxy。
- `dispatch_to_kv`：从 `DriverProxy send tasks` 到 PREFILL compute 的 `allocate kv for task ... hit_length`，包含 proxy 注册、control-driver 准入、排队及首次 KV 分配/查询。
- `kv_prepare`：从 PREFILL `allocate kv ... hit_length` 到 `post_allocate_kv ... allocate kv block`，包含等待 VLM embedding 和分配后记账。
- `sp_d2d`：显式 `kv_cache.sp_d2d.copy time cost`，表示把命中前缀 KV 从 SP cache 复制到本请求设备块，不是模型计算。
- `embed_wait`：从 PREFILL `allocate kv ... hit_length` 到 embedding worker 的 `to input_embed_shm, is_finish: True`。VLM 任务中通常接近 `kv_prepare`。
- `compute_wall`：从 `post_allocate_kv ... allocate kv block` 到来自 `SignalSource.COMPUTE_DRIVER` 的 compute-driver DONE，即 KV/embedding 就绪后的 prefill 关键路径墙钟计算窗口。
- `infer_exec_sum`：该任务所有 `prefill tasks [...] execute time: <s>` 之和，是跨 chunk 或 pipeline 阶段的 inferencer worker 累计执行/后处理时间；阶段重叠时可大于 `compute_wall`。
- `infer_exec_count`：计入 `infer_exec_sum` 的记录数。
- `transfer`：从 compute-driver DONE 到 `send transfer result`，包含 P-to-D KV 传输完成和通知/结果处理。
- `d2h_post`：显式 `kv_cache.d2h.copy time cost`，表示 DONE 后为未来复用执行的设备到主机 cache writeback，应与关键路径总时间分列。

这些列不一定可直接相加：`sp_d2d` 可与 embedding 重叠，`embed_wait` 是 `kv_prepare` 的子区间，而 `infer_exec_sum` 是累计 worker 时间而非墙钟区间。

## 源码解释提示

- `llmserver/rpc/handler/base_handler.py`：`call`、`stream_call`、预处理、`call add task`、`Processing request`。
- `llmserver/driver/proxy/driver_proxy.py`：`DriverProxy send tasks` 和 ControlDriver 选择。
- `llmserver/driver/loops/control_driver_loop.py`：`preprocess add req`、`send transfer result`。
- `llmserver/driver/kv_manager/kv_manager.py`：`first_allocate`、KV 查询/分配和命中长度日志。
- `llmserver/driver/loops/kv_allocate_loop.py`：compute allocate、embedding worker 调度、`_post_allocate_kv`。
- `llmserver/processor/vlm/embedding/worker.py`：把图像/音频 embedding 复制到 `input_embed_shm`。
- `bytedkvcache/manager/kvcache_manager.py` 和 `kvcache_worker.py`：`SPD2DCmdParam`、`D2HCmdParam`、`ops.sp_d2d`、d2h writeback。
- `llmserver/driver/loops/compute_driver_loop.py`：等待队列、inferencer 提交和后处理 DONE。
- `llmserver/driver/loops/kv_transfer_loop.py`：KV 传输和传输结果检查。

## 报告格式

先给出用户要求的表。最近请求摘要默认先给数据统计表；用户询问延迟时再添加延迟列或第二张表。简要说明提取口径，并单独列出只有 DONE 没有入口、或只有入口没有 DONE 的边界截断任务。

延迟表中，仅当用户不需要精确 ID 时才缩写任务 ID；否则保留完整 ID。始终说明 `total` 是否包含 `d2h_post`。
