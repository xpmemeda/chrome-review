import torch


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
