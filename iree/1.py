import torch
import iree.compiler.tools as ireec
import iree.runtime as ireert
import numpy as np


class M(torch.nn.Module):
    def forward(self, x, y):
        return x @ y


m = M().eval()
x = torch.randn(4, 4)
y = torch.randn(4, 4)

torch.onnx.export(
    m,
    (x, y),
    "matmul.onnx",
    input_names=["A", "B"],
    output_names=["C"],
    opset_version=13,
)


# 编译
vmfb = ireec.compile_file(
    "matmul.mlir",
    target_backends=["llvm-cpu"],
)

# Runtime 配置
config = ireert.Config("local-task")

# ① 用 VmModule.from_flatbuffer 创建模块
vm_module = ireert.VmModule.from_flatbuffer(config.vm_instance, vmfb)

# ② 创建 SystemContext 并挂载模块
ctx = ireert.SystemContext(
    config=config,
    vm_modules=[vm_module],
)

# ③ 通过 context 取出函数（注意 module 名一般就是 "module" 或你 MLIR 里的名字）
f = ctx.modules.module["main"]

A = np.ones((4, 4), dtype=np.float32)
B = np.ones((4, 4), dtype=np.float32)
C = f(A, B)

print("C:", C)
print(np.array(C))
