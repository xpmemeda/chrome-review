import triton
import torch

import kcuda.kernels as mykernels


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        # 4B, 64B, 1KB, 16KB, 256KB, 4MB, 64MB
        x_vals=[1, 4, 16, 64, 256, 1024, 4096],
        line_arg="provider",
        line_vals=[
            "green_ctx_y",
            "green_ctx_n",
        ],
        line_names=[
            "GreenCtx-Y(GB/s)",
            "GreenCtx-N(GB/s)",
        ],
        ylabel="GB/s",
        plot_name="greenctx",
        args={},
    )
)
def benchmark_greenctx(N, provider):

    x = torch.empty(N, N, dtype=torch.float, device="cuda")
    y = torch.empty(N, N, dtype=torch.float, device="cuda")

    if provider == "green_ctx_y":
        ms = triton.testing.do_bench(lambda: mykernels.greenctx_y(x, y))
    if provider == "green_ctx_n":
        ms = triton.testing.do_bench(lambda: mykernels.greenctx_n(x, y))

    gbps = lambda ms: y.nbytes * 1e-9 / (ms * 1e-3)

    return gbps(ms)


if __name__ == "__main__":
    benchmark_greenctx.run(print_data=True, save_path=".")
