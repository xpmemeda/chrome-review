import torch

from .kernels import moe_gate as _moegate


def moegate(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
):
    assert renormalize == True, "not support."

    _moegate(gating_output, correction_bias, num_expert_group, topk_group, topk)
