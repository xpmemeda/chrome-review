---
name: model-architecture-sizing
description: 根据模型 config JSON 和 modeling 源码估算 Transformer 或 MoE 模型的总参数量、激活参数量、每 token KV cache 字节数及每 GiB 可容纳 token 数。适用于 Doubao/Seed 风格架构说明，尤其是公式校验、Markdown 对齐、MTP 层、YOCO/KV 共享、FA/LA/SWA cache 变体及配置字段解释。
---

# 模型架构容量计算

## 概述

采用可复现的模型架构计算方式：读取模型配置，在 modeling 源码中确认特殊行为，再用显式公式计算参数量和 KV cache 容量。

优先输出便于直接阅读源码的简洁 Markdown；需要表格时使用对齐的 Markdown 表格。

## 工作流程

0. 参数量或 KV cache 容量计算缺少 config 路径或 modeling 源码路径，且无法从本地上下文可靠推断时，请用户同时提供两者。
1. 修改公式前，先读取目标 Markdown、模型 config JSON 和相关 modeling 文件。
2. 识别架构家族和命名约定，不要假设所有 config 使用同一 schema。
3. 参数量与 KV cache 必须使用一致的层范围；存在 MTP 时将其纳入。
4. 只使用二进制单位 KiB、MiB、GiB，不提供十进制与二进制两套结果。
5. KV cache 容量同时报告 bytes/token 和 `tokens/GiB`。
6. 写文档时展示足够的中间项，使读者可根据源码复核结果。

## 参数量计算

除非用户要求精确 checkpoint 参数总数，否则统计主要可学习矩阵，通常忽略 norm、bias、小型标量 buffer、量化 scale 和 embedding 分片元数据。

主要分类：Attention、MoE 普通专家、共享专家、Router、LM head、over-encoding table、逐 token over-encoding projection，以及 MTP 层。

标准 Attention：

```text
QKV = hidden_size * (q_heads + kv_heads + kv_heads) * head_dim
O   = hidden_size * q_heads * head_dim
```

M13 风格 FA + SWA：

```text
QKV = hidden_size * (fa_q_heads + fa_kv_heads + fa_kv_heads
                   + swa_q_heads + swa_kv_heads + swa_kv_heads) * head_dim
O   = hidden_size * (fa_q_heads + swa_q_heads) * head_dim
```

M12 风格 FA + LA：

```text
FA_QKV = hidden_size * (fa_q_heads + fa_kv_heads + fa_kv_heads) * head_dim
FA_O   = hidden_size * fa_q_heads * head_dim
LA_QKV = hidden_size * (la_q_heads * la_k_dim
                      + la_kv_heads * la_k_dim
                      + la_kv_heads * la_v_dim)
LA_O   = hidden_size * la_q_heads * la_v_dim
LA_gate ~= hidden_size * la_kv_heads * 2
```

MoE 层：

```text
ordinary_expert_total = num_experts * 3 * expert_dim * hidden_size
ordinary_expert_live  = top_k       * 3 * expert_dim * hidden_size
shared_expert_params  = 3 * shared_expert_dim * hidden_size
router_params         = hidden_size * num_experts
```

若共享专家以普通宽度专家数量表示：

```text
shared_expert_dim = shared_expert_count * expert_dim
```

LM head：

```text
lm_head = vocab_size * hidden_size
```

默认使用 forward-active 口径统计激活参数：包含实际激活的普通专家、共享专家、Router、Attention、LM head，以及 OE projection 等每 token 都执行的稠密辅助投影；不含 token embedding 行和 OE table lookup 等纯表查询。总参数量包含全部普通专家和实际实例化的稠密辅助投影。

## 配置字段

使用结构化 JSON 解析 config；有歧义的字段必须结合 modeling 源码确认。

| 含义 | 常见字段名 |
| --- | --- |
| 隐藏层维度 | `hidden_size`, `n_embed`, `dim` |
| 层数 | `num_hidden_layers`, `num_layers`, `n_layers` |
| Query head 数 | `num_attention_heads`, `num_heads`, `n_heads` |
| KV head 数 | `num_key_value_heads`, `n_kv_heads` |
| Head dim | `head_dim`，或 `hidden_size / num_attention_heads` |
| 普通专家数 | `num_experts`, `n_routed_experts` |
| 激活专家数 | `num_experts_per_tok`, `top_k`, `moe_topk` |
| 共享专家数 | `n_shared_experts`, `moe_shared_expert_num` |
| 共享专家维度 | `moe_shared_ffn_internal_dim`，或由数量乘维度得到 |
| M8/M12 专家维度 | `moe_ffn_internal_dim` |
| M13 专家维度 | `intermediate_size` |
| HF 风格 MTP 额外 head | `mtp_n_heads`；额外 MTP 层数为 `mtp_n_heads - 1` |
| Infir 风格 MTP 层 | `num_mtp_layers` |

用户要求不依赖 config 判断 MTP 是否存在时，检查 modeling 源码和该模型家族约定；字段存在时仍应说明具体由哪个 config 字段表示数量。

## 层范围

参数量和 KV cache 的层数必须明确。

若 HF 风格 config 的 `num_hidden_layers` 已包含 MTP head：

```text
extra_mtp_layers = mtp_n_heads - 1
backbone_layers  = num_hidden_layers - extra_mtp_layers
total_layers     = num_hidden_layers
```

若 MTP 在 config 中单独计数：

```text
extra_mtp_layers = num_mtp_layers
backbone_layers  = num_layers
total_layers     = num_layers + extra_mtp_layers
```

modeling 源码显示 MTP 拥有独立 full-history KV cache 时，将 MTP 纳入 KV cache；若只复用主模型 cache，不得虚构独立 MTP cache。对于本 Skill 对应的 Doubao/Seed 说明，存在 MTP 时同时计入参数量和 KV cache。

## KV Cache

默认使用 C16：

```text
element_bytes = 2
GiB = 1,073,741,824 bytes
```

Full-attention 每 token KV cache：

```text
bytes_per_token = physical_kv_layers * kv_heads * head_dim * 2 * element_bytes
tokens_per_GiB  = 1,073,741,824 / bytes_per_token
```

其中 `2` 分别代表 K 和 V。存在多组 KV 时先分别计算物理层/head：

```text
bytes_per_token = sum(group_physical_layers * group_kv_heads * group_head_dim * 2 * element_bytes)
```

SWA 的固定窗口开销与 full-attention 的逐 token 开销分开：

```text
swa_fixed_bytes = swa_physical_layers * window_tokens * swa_kv_heads * head_dim * 2 * element_bytes
```

计算 `tokens/GiB` 时默认忽略 SWA 固定开销，除非用户要求纳入每请求固定成本。

LA state 以 modeling 源码为准，常见形状：

```text
la_state_bytes = la_physical_layers * la_kv_heads * la_k_dim * la_v_dim * state_element_bytes
```

根据讨论的源码/运行时路径说明 `state_element_bytes` 是 2 还是 4。

## KV 共享与镜像规则

模型存在 KV 共享或镜像时，不能直接按逻辑层数统计；必须检查 modeling 源码。

KV mirror config：

```text
physical_kv_layers = total_layers - len(kv_mirror_layers)
```

`kv_mirror_layers` 是不保存独立物理 cache 的层，`kv_mirror_imitated_layers` 是被复用 K/V 的源层。许多部署 config 使用从 1 开始的层 ID，与 Python 从 0 开始的 ID 混用前必须通过源码确认。

YOCO 或 `use_kv_share=true` 通常按以下方式统计物理存储，而非逻辑消费者层数：

```text
physical_layers = layers_before_share_start
                + yoco_saved_layers_after_share_start
                + extra_mtp_layers_if_they_have_cache
```

若模型有 full/sparse 或 FA/SWA 等独立 cache 组，应按各组自己的物理层集合和 head 数分别统计。

## 文档风格

- 匹配用户现有说明风格，优先使用简短中文解释并在下方给公式。
- 中文 Markdown 表格在源码中难以对齐时可以保留英文表头。
- 周围文档使用 FA、LA、SWA 缩写时保持一致。
- 不要在每节重复通用范围声明；只使用二进制单位。
- `model-arch/` 下默认匹配 `doubao-seed-2.1-lite-M13-12B.md` 的紧凑章节风格。
- 默认章节顺序：`# 模型架构`、`# 总权重及激活权重数量`、`## OE`、可选 `## OE Projection`、`## MoE`、`## Attention`、`## LM Head`、`# C16 缓存大小`，以及参考文档使用的可选部署容量章节。
- 总结尽量限制为三项：`OE table 权重`、`总权重（不含 OE table）`、`激活权重（不含 OE table）`。
- `OE` 章节只按参考说明的紧凑形式统计 OE embedding table。若 over-encoding 含每 token 都执行的稠密投影，单独增加 `## OE Projection`，并计入不含 OE table 的总参数和激活参数。
- norm、VWN、gate、标量 buffer 或 MTP helper projection 等相对可忽略的小模块不要单独成节；可在架构要点中说明或忽略。
- `Attention` 只保留 QKV 与 O 公式；参考说明没有 `Attention total` 时不要额外添加。
- `LM Head` 只统计 LM head；除非用户要求，不包含 token embedding。
- H20/H800 风格容量章节沿用参考说明的项目符号和表格结构，并尽量减少假设说明。

紧凑说明骨架：

```markdown
# 模型架构

- ...

# 总权重及激活权重数量

- OE table 权重：...
- 总权重（不含 OE table）：...
- 激活权重（不含 OE table）：...

## OE

- 词表个数：...
- 词表大小：~...
- 词表维度：...

**OE 权重随词表数量线性增长。**

- 每个 OE 词表：...
- 全部 OE 词表：...
```

表格示例：

```markdown
| 项目                 | 公式                            | 结果          |
| -------------------- | ------------------------------- | ------------- |
| FA KV / token        | `87 * 8 * 128 * 2 * 2`          | `356,352 B`   |
| Tokens / GiB         | `1,073,741,824 / 356,352`       | `3,013.1 tok` |
```
