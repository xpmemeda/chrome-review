import kernels
import torch

x = torch.ones(32, 32, dtype=torch.float16, device="cuda:0")
print(hex(x.data_ptr()))
print(x.data_ptr() % 16)
print(x.data_ptr() % 32)
print(x.data_ptr() % 128)
print(x.data_ptr() % 512)
print(x.data_ptr() % 1024)
print(x)
kernels.copy(x, x)
print(x)