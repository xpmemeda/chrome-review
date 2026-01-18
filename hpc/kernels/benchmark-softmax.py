import triton
import torch

from ktriton.softmax import softmax as triton_softmax
from kcuda.softmax import softmax as cuda_softmax


def test_softmax(M, N, dtype, device=torch.device("cuda:0")):
    x = torch.rand(M, N, dtype=dtype, device=device)
    triton_output = triton_softmax(x)
    cuda_output = cuda_softmax(x)
    torch_output = torch.softmax(x, axis=-1)

    torch.testing.assert_close(torch_output, triton_output)
    torch.testing.assert_close(torch_output, cuda_output)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[1, 128, 1024, 8192],
        line_arg="provider",
        line_vals=["triton", "torch", "cuda"],
        line_names=["Triton", "Torch", "CUDA"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="GB/s",
        plot_name="softmax-4096",
        args={"M": 4096},
    )
)
def benchmark(M, N, provider, device=torch.device("cuda:0")):
    x = torch.randn(M, N, device=device, dtype=torch.float32)

    if provider == "torch":
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == "triton":
        ms = triton.testing.do_bench(lambda: triton_softmax(x))
    if provider == "cuda":
        ms = triton.testing.do_bench(lambda: cuda_softmax(x))

    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms)


if __name__ == "__main__":
    benchmark.run(print_data=True, save_path="")
