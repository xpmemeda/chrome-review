import torch
import triton
import triton.language as tl

# def moegate(
#     gating_output: torch.Tensor,
#     correction_bias: torch.Tensor,
#     topk: int,
#     renormalize: bool,
#     num_expert_group: int = 0,
#     topk_group: int = 0,
# ):
#     scores = gating_output.sigmoid()
#     num_token = scores.shape[0]
#     num_experts = scores.shape[1]
#     scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
#     group_scores = (
#         scores_for_choice.view(num_token, num_expert_group, -1)
#         .topk(2, dim=-1)[0]
#         .sum(dim=-1)
#     )  # [n, n_group]
#     group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
#         1
#     ]  # [n, top_k_group]
#     group_mask = torch.zeros_like(group_scores)  # [n, n_group]
#     group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
#     score_mask = (
#         group_mask.unsqueeze(-1)
#         .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
#         .reshape(num_token, -1)
#     )  # [n, e]
#     tmp_scores = scores_for_choice.masked_fill(
#         ~score_mask.bool(), float("-inf")
#     )  # [n, e]
#     _, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
#     topk_weights = scores.gather(1, topk_ids)

#     if renormalize:
#         topk_weights_sum = topk_weights.sum(dim=-1, keepdim=True)
#         topk_weights = topk_weights / topk_weights_sum

#     return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


@triton.jit
def _moegate_kernel(
    gating_output,  # [N, E]
    correction_bias,  # [E]
    E: tl.constexpr,
    TOPK: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    NUM_EXPERT_GROUP: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tl.static_assert(E == BLOCK_SIZE)

    row = tl.program_id(0)
    gating_output += E * row

    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(gating_output + cols).to(tl.float32)
    b = tl.load(correction_bias + cols).to(tl.float32)
    x = x + b
