import triton
import torch

import kcuda.kernels as mykernels


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[1, 16, 128, 1024, 8192],
        line_arg="provider",
        line_vals=["torch", "naive"],
        line_names=["Torch(GB/s)", "Naive(GB/s)"],
        ylabel="GB/s",
        plot_name="sgemm-k-4096-n-4096",
        args={"K": 4096, "N": 4096},
    )
)
def benchmark_sgemm(M, K, N, provider):
    x = torch.rand(M, K, device="cuda", dtype=torch.float) - 0.5
    w = torch.rand(K, N, device="cuda", dtype=torch.float) - 0.5
    ref = torch.matmul(x, w)

    if provider == "naive":
        r = torch.empty(M, N, device="cuda", dtype=torch.float)
        mykernels.sgemm_naive(r, x, w)
        torch.testing.assert_close(ref, r, rtol=1e-4, atol=1e-4)

        ms = triton.testing.do_bench(lambda: mykernels.sgemm_naive(r, x, w))

    if provider == "torch":
        ms = triton.testing.do_bench(lambda: torch.matmul(x, w))

    gbps = lambda ms: x.nbytes * 1e-9 / (ms * 1e-3)

    return gbps(ms)


if __name__ == "__main__":
    benchmark_sgemm.run(print_data=True)
