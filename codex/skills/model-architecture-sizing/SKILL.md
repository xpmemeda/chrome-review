---
name: model-architecture-sizing
description: Use this skill when estimating transformer or MoE model parameter counts, active parameter counts, KV cache bytes per token, and tokens per GiB from model config JSON files plus modeling source code. Use it for Doubao/Seed-style architecture notes, especially when the user asks to verify formulas, align Markdown docs, include MTP layers, handle YOCO/KV sharing, FA/LA/SWA cache variants, or explain which config fields drive the calculation.
---

# Model Architecture Sizing

## Overview

This skill defines a reproducible calculation style for model architecture notes: read the model config, confirm special behavior in the modeling code, then compute parameter count and KV cache capacity with explicit formulas.

Prefer concise Markdown that the user can read in source form. Use aligned Markdown tables when tables are useful.

## Workflow

0. For parameter/KV-cache sizing requests, ask the user for both the model config path and modeling source path when either is missing and cannot be confidently inferred from local context.
1. Read the target Markdown, model config JSON, and relevant modeling file before changing formulas.
2. Identify the architecture family and naming conventions instead of assuming one config schema.
3. Count parameters and KV cache with the same layer scope. Include MTP layers when they exist.
4. Use Binary units only: KiB, MiB, GiB. Do not present Decimal-vs-Binary alternatives.
5. For KV cache capacity, report both bytes per token and `tokens/GiB`.
6. When writing docs, show formulas with enough intermediate terms that the result can be checked from the source file.

## Parameter Counting

Count the major learned matrices unless the user asks for exact checkpoint parameter totals. Normally ignore norms, biases, small scalar buffers, quantization scales, and embedding sharding metadata.

Use these buckets:

- Attention parameters.
- MoE ordinary expert parameters.
- Shared expert parameters.
- Router parameters.
- LM head parameters.
- Over-encoding tables and per-token over-encoding projection, if present.
- MTP layers, if present.

For standard attention:

```text
QKV = hidden_size * (q_heads + kv_heads + kv_heads) * head_dim
O   = hidden_size * q_heads * head_dim
```

For M13-style FA + SWA attention:

```text
QKV = hidden_size * (fa_q_heads + fa_kv_heads + fa_kv_heads
                   + swa_q_heads + swa_kv_heads + swa_kv_heads) * head_dim
O   = hidden_size * (fa_q_heads + swa_q_heads) * head_dim
```

For M12-style FA + LA attention:

```text
FA_QKV = hidden_size * (fa_q_heads + fa_kv_heads + fa_kv_heads) * head_dim
FA_O   = hidden_size * fa_q_heads * head_dim
LA_QKV = hidden_size * (la_q_heads * la_k_dim
                      + la_kv_heads * la_k_dim
                      + la_kv_heads * la_v_dim)
LA_O   = hidden_size * la_q_heads * la_v_dim
LA_gate ~= hidden_size * la_kv_heads * 2
```

For MoE layers:

```text
ordinary_expert_total = num_experts * 3 * expert_dim * hidden_size
ordinary_expert_live  = top_k       * 3 * expert_dim * hidden_size
shared_expert_params  = 3 * shared_expert_dim * hidden_size
router_params         = hidden_size * num_experts
```

If shared experts are represented as a count of ordinary-width experts, derive:

```text
shared_expert_dim = shared_expert_count * expert_dim
```

LM head:

```text
lm_head = vocab_size * hidden_size
```

Active parameters in these notes use the forward-active convention by default: include active ordinary experts, shared experts, router, attention, LM head, and dense auxiliary projections that run for every token, such as OE projection. Exclude pure embedding-table gathers such as token embedding rows and OE table lookups from active parameters, while still counting their full tables in total/OE table sections when requested. Total parameters include all ordinary experts and instantiated dense auxiliary projections.

## Config Fields

Use structured JSON parsing for config files. Confirm ambiguous fields against the modeling source.

Common field meanings:

| Meaning                  | Common field names                                   |
| ------------------------ | ---------------------------------------------------- |
| Hidden size              | `hidden_size`, `n_embed`, `dim`                      |
| Layer count              | `num_hidden_layers`, `num_layers`, `n_layers`        |
| Query heads              | `num_attention_heads`, `num_heads`, `n_heads`        |
| KV heads                 | `num_key_value_heads`, `n_kv_heads`                  |
| Head dim                 | `head_dim`, or `hidden_size / num_attention_heads`   |
| Ordinary experts         | `num_experts`, `n_routed_experts`                    |
| Active experts           | `num_experts_per_tok`, `top_k`, `moe_topk`           |
| Shared expert count      | `n_shared_experts`, `moe_shared_expert_num`          |
| Shared expert dim        | `moe_shared_ffn_internal_dim`, derived count * dim   |
| M8/M12 expert dim        | `moe_ffn_internal_dim`                               |
| M13 expert dim           | `intermediate_size`                                  |
| HF-style MTP extra heads | `mtp_n_heads`; extra MTP layers = `mtp_n_heads - 1`  |
| Infir-style MTP layers   | `num_mtp_layers`                                    |

When the user says not to rely on the config for MTP presence, inspect the modeling code and existing model-family convention. Still use config fields to explain which names carry the count when they are present.

## Layer Scope

Keep the parameter layer count and KV cache layer count explicit.

For HF-style configs where `num_hidden_layers` includes MTP heads:

```text
extra_mtp_layers = mtp_n_heads - 1
backbone_layers  = num_hidden_layers - extra_mtp_layers
total_layers     = num_hidden_layers
```

For configs where MTP is separate:

```text
extra_mtp_layers = num_mtp_layers
backbone_layers  = num_layers
total_layers     = num_layers + extra_mtp_layers
```

If modeling code shows MTP has its own full-history KV cache, include MTP in KV cache. If it only reuses the main model cache, do not invent separate MTP cache. For the Doubao/Seed notes this skill was created from, include MTP layers in both parameter count and KV cache when present.

## KV Cache

Use C16 by default:

```text
element_bytes = 2
GiB = 1,073,741,824 bytes
```

Full-attention KV cache per token:

```text
bytes_per_token = physical_kv_layers * kv_heads * head_dim * 2 * element_bytes
tokens_per_GiB  = 1,073,741,824 / bytes_per_token
```

The `2` is for K and V.

For architectures with multiple KV groups, sum physical KV head groups first:

```text
bytes_per_token = sum(group_physical_layers * group_kv_heads * group_head_dim * 2 * element_bytes)
```

For SWA, separate fixed window cost from per-token full-attention cost:

```text
swa_fixed_bytes = swa_physical_layers * window_tokens * swa_kv_heads * head_dim * 2 * element_bytes
```

When computing `tokens/GiB`, ignore SWA fixed overhead unless the user explicitly asks to include fixed per-request cost.

For LA state, follow the modeling code. A common shape is:

```text
la_state_bytes = la_physical_layers * la_kv_heads * la_k_dim * la_v_dim * state_element_bytes
```

Document whether `state_element_bytes` is 2 or 4 from the code/runtime path being discussed.

## KV Sharing And Mirror Rules

Do not count logical layers blindly when the model shares or mirrors KV cache. Inspect the modeling code.

For KV mirror configs:

```text
physical_kv_layers = total_layers - len(kv_mirror_layers)
```

`kv_mirror_layers` are layers that do not keep their own physical cache. `kv_mirror_imitated_layers` are the source layers whose saved K/V are reused. Many deployment configs list these layers as 1-based ids; verify with code before mixing them with 0-based Python layer ids.

For YOCO or `use_kv_share=true` configs, count physical KV storage, not logical consumer layers. A typical pattern is:

```text
physical_layers = layers_before_share_start
                + yoco_saved_layers_after_share_start
                + extra_mtp_layers_if_they_have_cache
```

If the model has separate full/sparse or FA/SWA cache groups, count each group with its own physical layer set and head count.

## Documentation Style

Match the user's existing note style:

- Prefer short Chinese explanations with formulas below.
- Use English table headers if Chinese Markdown table alignment is hard to read in source.
- Use FA, LA, and SWA names when the surrounding docs use those abbreviations.
- Avoid repeating generic scope disclaimers in every section.
- Keep calculation rows aligned in Markdown source.
- Use Binary units only.
- For notes under `model-arch/`, match the compact section style of `doubao-seed-2.1-lite-M13-12B.md` unless the user asks for more detail.
- Start with `# 模型架构`, then `# 总权重及激活权重数量`, followed by `## OE`, optional `## OE Projection`, `## MoE`, `## Attention`, `## LM Head`, `# C16 缓存大小`, and optionally the deployment capacity section used by the reference note.
- Keep the summary section to three bullets when possible: `OE table 权重`, `总权重（不含 OE table）`, and `激活权重（不含 OE table）`. Do not add scope prose there unless needed to resolve an ambiguity.
- In the `OE` section, count only the OE embedding table in the same compact style as the reference note. If over-encoding has a dense projection that runs every token, add a separate `## OE Projection` section and include it in both total parameters excluding OE table and active parameters excluding OE table.
- Do not create separate sections for tiny auxiliary modules such as norms, VWN, gates, scalar buffers, or MTP helper projections when their parameter count is negligible relative to the main buckets. Mention them in the architecture bullets or ignore them in the parameter totals.
- Keep `Attention` to QKV and O formulas. Avoid adding an `Attention total` line if the reference note does not have one.
- Keep `LM Head` limited to LM head parameters. Do not include token embedding there unless the user asks to count input embeddings explicitly.
- In H20/H800-style capacity sections, use the same bullet-and-table structure as the reference note, and keep notes about assumptions minimal.

Compact model-arch note skeleton:

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

Example table style:

```markdown
| Item                 | Formula                         | Result        |
| -------------------- | ------------------------------- | ------------- |
| FA KV / token        | `87 * 8 * 128 * 2 * 2`          | `356,352 B`   |
| Tokens / GiB         | `1,073,741,824 / 356,352`       | `3,013.1 tok` |
```
