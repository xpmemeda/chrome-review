import torch
import triton

from ktorch.moegate import moegate as torch_moegate
from kcuda.moegate import moegate as cuda_moegate


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[i for i in range(8, 512, 8)],
        line_arg="provider",
        line_vals=["torch", "cuda"],
        line_names=["Torch", "CUDA"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="moegate",
        args={"E": 256, "dtype": torch.bfloat16},
    )
)
def bench_moegate(N, E, dtype, provider, device=torch.device("cuda:0")):
    x = torch.rand([N, E], dtype=dtype, device=device)
    correction_bias = torch.rand([E], dtype=dtype, device=device)
    topk = 8
    renormalize = True
    num_expert_group = 8
    topk_group = 4

    def fwd():
        if provider == "torch":
            return torch_moegate(
                x, correction_bias, topk, renormalize, num_expert_group, topk_group
            )
        if provider == "cuda":
            return cuda_moegate(
                x, correction_bias, topk, renormalize, num_expert_group, topk_group
            )

    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fwd, quantiles=quantiles, rep=500)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    bench_moegate.run(save_path=".", print_data=True)
