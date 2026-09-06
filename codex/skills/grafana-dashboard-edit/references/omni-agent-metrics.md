# Omni-Agent 看板与指标口径

以下是 2026-09-17 整理的历史验证结果，不保证未来配置或部署仍相同。使用时重新读取看板，核对指标有数据后再保存；不把这里的 panel ID 当作跨看板通用 ID。

## 定位

- 专用看板：`Omni-Agent 主模型监控`，UID `WTYR07lDz`，入口 https://grafana.byted.org/v10/d/WTYR07lDz 。
- 原始参考看板：UID `dSd1OEVvz`，https://grafana.byted.org/d/dSd1OEVvz/seed-llmserver-2-0 。只作为参考，不因编辑专用看板而修改原始看板。
- 曾验证服务：`ocean.assistant.omniagent_m12_30b_0902_npu`；不是固定默认 PSM，使用用户当前服务。
- 常见数据源变量：`${tsdb_loc}` 为 bytetsd，`${bosun_loc}` 为 Bosun。复用现场 target 模板，不假定所有看板或环境的数据源 UID 相同。
- 常见过滤：`dc=$dc`、`bernard_service_id=$service_id`。调整分组时保留这些筛选。
- 本地代码：`~/workspace/byted/seed.llmserver`；缓存逻辑在 `bytedkvcache/manager/kvcache_manager.py`。仓库路径和实现需现场确认。

## Session 次数与长度

历史 panel 35–37 为首轮输入、首轮命中、后续轮计算长度；panel 46 为 Session 命中 / 未命中次数。

埋点依据 `hist_ctx.session_exist`，Prefill 上报：

- 未命中：`${PSM}.kv_cache.prefill_session_miss_input_length`，值为输入长度。
- 未命中：`${PSM}.kv_cache.prefill_session_miss_hit_length`，值为命中长度。
- 命中：`${PSM}.kv_cache.prefill_session_hit_prefill_length`，值为本次 `context_len - hit_length`。

长度查询后缀 `.avg` / `.pct99`；观察次数使用 **`.counter`**，历史实测 `.count` 无数据。该环境 `.counter` 为上报周期样本次数，不是单调累计 counter，不套 rate。空间和时间都用 sum，图例用 sum 显示范围内次数；精确性仍依赖数据完整性和查询降采样。

一次未命中有两个长度埋点，次数只取其中一个，不能相加。只取 Prefill，避免与 Decode 重复计数。`kv_cache.miss_reason` 的零 token 命中与 session 不存在不是同一口径。

历史验证租户为 **`seed.infra_serving`**，上述 panel 固定此租户。新部署需再次核对，不能把其他已有指标的 `$tenant` 全局替换。

估算平均轮次 = 范围内命中次数 / 范围内未命中次数 + 1。前提是每个 session 首轮恰好一次未命中、后续轮都命中；窗口截断、淘汰、重启、重试会破坏此假设。分母为零/无数据不能计算，不先逐点求比值再平均。

## SP 种类数

历史 panel 45：最近 30 分钟 SP 种类数，最终为 Prefill、Decode 两条进程均值曲线。

- `${PSM}.kv_cache.prefill_sp_unique_count_30m`
- `${PSM}.kv_cache.decode_sp_unique_count_30m`
- 混合部署另有 `${PSM}.kv_cache.sp_unique_count_30m`；用户只要两条 PD 曲线时不加入 Mixed。

这是本地 cache manager 进程滚动 30 分钟去重 gauge，约每 30 秒清理和上报，进程重启后重新累计。默认 host 上下文加 `process_id` 区分；多个进程的种类数不能求和当全局去重数。均值展示用 avg/avg，取消 host、process_id 分组，alias 为 Prefill / Decode 均值，通常保留两位小数。历史租户也是 `seed.infra_serving`。

## Decode 三个均值面板

- panel 25：`${PSM}.batch_size.decode.avg`，运行 Batch Size。
- panel 26：`${PSM}.kv_cache.decode_block_occupancy.avg`，KV block 使用率，0–1，单位 `percentunit`。
- panel 27：`${PSM}.throughput.accept_length.avg`，投机采样接受长度，埋点**已经含 bonus token**，不能再加 1。

最终均为 avg/avg，取消 host 分组，一条均值线，保留 `$tenant` 和用户筛选。这里是上报序列均值，不是请求数加权或卡容量加权。

## Uncached TPS

历史 panel 44 使用 Bosun。Prefill token 数来自 `${PSM}.kv_cache.prefill_miss_length.sum`；Decode 来自 `${PSM}.length.output_ids.sum`，新增输出 token。历史上报窗口为 30 秒，因此除以 30 转为 token/s；换埋点或窗口时重新确认，不能对所有 `.sum` 指标一律除以 30。

最终用户要求跨 IDC、实例**总量**，两条线。历史有效表达式：

```text
aggr(q("sum:[$tenant]${PSM}.kv_cache.prefill_miss_length.sum{dc=$dc,bernard_service_id=$service_id}", "$start", ""), "", "sum") / 30
aggr(q("sum:[$tenant]${PSM}.length.output_ids.sum{dc=$dc,bernard_service_id=$service_id}", "$start", ""), "", "sum") / 30
```

原先单实例均值按 dc 分线，使用 `avg:` 和 `aggr(..., "dc", "avg")`。改总量必须连同内部聚合、外部聚合、alias、标题和说明一起改；只删除 alias 中 `$tag_dc` 不会合并数据。保留 dc 筛选，`All` 才覆盖所有 IDC。

## SLO

历史 panel 38 为 ModelAPI 调用成功率，不是进程存活率。确切公式需从原始看板或代码读取，不能猜测过滤和分母。内部压测标签未允许时，上游可返回 `request forbidden`，后端没有收到请求也能导致调用 SLO 下跌；不要仅凭 SLO 判断 llmserver 宕机。若计算结果出现负成功率，先核对分子分母口径，不通过裁剪到 0–100% 掩盖问题。
