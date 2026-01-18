import torch
import triton

from ktriton.layernorm import layernorm as triton_layernorm
from kcuda.layernorm import layernorm as cuda_layernorm


def test_layer_norm(M, N, dtype, eps=1e-5, device=torch.device("cuda:0")):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    # forward pass
    y_tri = triton_layernorm(x, weight, bias, eps)
    y_cud = cuda_layernorm(x, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    # compare
    assert torch.allclose(y_cud, y_ref, atol=1e-2, rtol=0)
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg="provider",
        line_vals=["triton", "torch", "cuda"],
        line_names=["Triton", "Torch", "CUDA"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="GB/s",
        plot_name="layernorm",
        args={"M": 4096, "dtype": torch.float16},
    )
)
def bench_layer_norm(M, N, dtype, provider, eps=1e-5, device=torch.device("cuda:0")):
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)

    def y_fwd():

        if provider == "triton":
            return triton_layernorm(x, weight, bias, eps)

        if provider == "torch":
            return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)

        if provider == "cuda":
            return cuda_layernorm(x, weight, bias, eps)

    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    test_layer_norm(38, 2644, torch.bfloat16)
    test_layer_norm(1151, 8192, torch.float16)
    bench_layer_norm.run(save_path=".", print_data=True)
