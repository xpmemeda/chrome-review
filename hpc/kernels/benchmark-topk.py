import torch
import kcuda.kernels as mykernels

src = torch.rand(10, 1024, dtype=torch.half, device="cuda")
dst = torch.empty(10, 32, dtype=torch.half, device="cuda")
mykernels.topk(src, dst, 32)
print(dst)