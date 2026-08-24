---
name: kvcache-session-turn-stats
description: 按 session 和轮次分析 llmserver Bernard 日志中的请求长度、KV cache 命中长度与命中率，并统计符合条件会话的第 15–25 轮 token 增长量和重写量。适用于用户要求统计请求形态、 session 请求长度、命中率、轮次增长或重写情况。
---

# KV Cache 会话轮次统计

- 日志所在的路径是 `/opt/tiger/toutiao/log/run/bernard_stdout_log.<YYYYMMDD>-0000`。
- 可以将 `scripts/session_turn_stats.py` 临时上传到远端 /tmp，使用该脚本对 Bernard 标准输出日志进行确定性、只读聚合；不得复制或修改原日志。执行结束后删除本次上传的脚本。

## 生产环境安全要求

- 在生产机器上，不要修改原日志、更改权限、重启服务或改变服务状态。
- 对于临时远程机器，先遵守 `remote-dev-agent` 的健康检查要求，并获得执行只读命令的授权。

## 必需的会话筛选规则

- 从 `kv_manager.py:174 allocate kv` 解析请求，并根据 `kvcache_manager.py:2835 ... release` 将每个任务映射到对应会话。
- 按分配记录中的时间戳排列请求。一个请求视为一轮。
- 排除日志日期当天 06:00 之前出现过任何请求的所有会话。该规则比仅丢弃早期请求更严格，可避免把 06:00 前已活跃的会话重新标记为新会话。
- 只保留当日完整请求数在 10–50 之间（含边界）的会话。
- 如果 session 包含图片，所有观察到的图片必须严格为 **宽 632 × 高 1400**；任意一张为其他尺寸（包括 630×1400、1400×632），排除整个 session，而不是仅丢弃该图片或请求。没有观察到图片的 session 仍可参与统计；已观察到图片但尺寸缺失时，无法验证尺寸，也排除该 session。
- 从带 task 标识的 `ImageMetadata(num_tokens=..., width=..., height=..., ..., image_id=...)` 记录提取图片信息，再通过 release 记录关联 session。检查该 session 的全部已关联图片记录，不仅检查首轮或新增图片；不要求图片记录与 allocate/release 按固定顺序出现。
- 对每个符合条件的会话，将最早的请求视为第一轮，其余请求视为后续轮次。
- 使用 `token_num` 作为提示词 token 数，使用同一条分配记录中的 `hit_length` 作为命中 token 数。

## 图片 token 统计

- 最终报告纳入统计会话中 632×1400 图片对应的 `num_tokens`，不写死为 220 或 495。
- 图片按 `image_id` 去重；没有 ID 时优先使用 hash，均缺失时使用 task 与记录位置作为退化标识，并说明去重局限。同一图片的历史重现、重复 worker 记录不重复计数；同一 ID 出现不同 token 元数据时保留不同观测并报告分布，不假定它们一致。
- 所有有效观测一致时报告“每张 N token”；存在多个值时报告各 token 值的图片观测数和加权均值，不能宣称所有图片 token 数固定。没有合格图片或 token 字段缺失时明确报告无法确定，不能当作 0 token。
- `images_632x1400` 是通过全部会话筛选后的图片统计，不是全日志图片分布。额外报告因图片尺寸不符或未知而排除的会话数。
- 图片筛选以日志实际记录为限。只记录送往视觉编码器的图片可能漏掉缓存复用图片；没有图片记录不等于证明是纯文本。报告无法关联 task/session 的图片记录诊断；日志不足时明确覆盖局限，不声称已验证所有历史图片。

## 第 15–25 轮

对于每个保留的会话，在相应轮次存在时，计算第 15 至 25 轮（含边界）的以下数值：

- `new_tokens = current token_num - previous token_num`
- `compute_tokens = current token_num - current hit_length`
- `rewrite_tokens = compute_tokens - new_tokens`，等价于 `previous token_num - current hit_length`

不要求相邻轮次的 `token_num` 递增；当当前轮请求长度小于上一轮时，`new_tokens` 可以为负数。

按轮次分别跨会话聚合。始终打印全部 11 个轮次行，包括请求数、平均新增 token 数和平均重写 token 数。在第 25 轮之后，打印一个按第 15–25 轮所有请求加权的 `average` 行；不要对 11 个逐轮平均值再次做等权平均。

运行：

```bash
python3 scripts/session_turn_stats.py /path/to/bernard_stdout_log.YYYYMMDD-0000
```

解析器只流式读取文件一次，绝不写入文件。可以在本地运行；如果希望避免下载大日志，也可以通过已授权的 remote Agent 运行。

## 报告要求

始终报告：

- 符合条件的会话数和请求数；
- 每个会话的平均轮数；
- 第一轮的平均提示词 token 数和命中 token 数；
- 后续轮次的平均提示词 token 数和命中 token 数；
- 第 15 至 25 轮每一轮的请求数、平均新增 token 数和平均重写 token 数；
- 第 15–25 轮按请求加权的最终平均行；
- 合格会话中每张 632×1400 图片的 token 数（或分布）、图片样本数及图片尺寸筛选排除数。

同时说明统计范围仅包含首次出现时间不早于 06:00、请求数为 10–50（含边界），且所有观察到的图片均为 632×1400（无图片记录也允许）的会话。将第一轮标注为日志中首次观察到的请求；仅根据单日日志，无法识别前一天创建但在 06:00 前没有活动的会话。命中率或平均计算 token 数只能作为补充指标。

报告日志路径，并说明分析是在本地运行，还是通过只读远程流式处理运行。
