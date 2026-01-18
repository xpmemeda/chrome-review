import torch

from .kernels import layernorm_v3 as _layernorm


def layernorm(x, w, b, eps):
    out = torch.empty_like(x)
    _layernorm(x, out, w, b, eps)
    return out
