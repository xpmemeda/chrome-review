import triton
import torch

import kcuda.kernels as mykernels


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[1, 128, 1024, 8192],
        line_arg="provider",
        line_vals=["torch", "naive", "wmma", "wmma_cpasync", "mma_cpasync", "cutlass"],
        line_names=[
            "Torch(GB/s)",
            "Naive(GB/s)",
            "Wmma(GB/s)",
            "WmmaCpAsync(GB/s)",
            "MmaCpAsync(GB/s)",
            "Cutlass(GB/s)",
        ],
        ylabel="GB/s",
        plot_name="hgemm-k-4096-n-4096",
        args={"K": 4096, "N": 4096},
    )
)
def benchmark_hgemm(M, K, N, provider):
    # x = torch.ones(M, K, device="cuda", dtype=torch.half)
    x = torch.rand(M, K, device="cuda", dtype=torch.half) - 0.5
    # w = torch.ones(K, N, device="cuda", dtype=torch.half)
    w = torch.rand(K, N, device="cuda", dtype=torch.half) - 0.5
    ref = torch.matmul(x, w).to(torch.float)

    if provider == "naive":
        r = torch.zeros(M, N, device="cuda", dtype=torch.float)
        mykernels.hgemm_naive(r, x, w)
        torch.testing.assert_close(ref, r, rtol=1e-3, atol=1e-3)

        ms = triton.testing.do_bench(lambda: mykernels.hgemm_naive(r, x, w))
    if provider == "wmma":
        r = torch.zeros(M, N, device="cuda", dtype=torch.float)
        mykernels.hgemm_wmma(x, w, r, 1.0, 1.0)
        torch.testing.assert_close(ref, r, rtol=1e-3, atol=1e-3)

        ms = triton.testing.do_bench(lambda: mykernels.hgemm_wmma(x, w, r, 1.0, 1.0))

    if provider == "wmma_cpasync":
        r = torch.zeros(M, N, device="cuda", dtype=torch.float)
        mykernels.hgemm_wmma_cpasync(x, w, r, 1.0, 1.0)
        torch.testing.assert_close(ref, r, rtol=1e-3, atol=1e-3)

        ms = triton.testing.do_bench(
            lambda: mykernels.hgemm_wmma_cpasync(x, w, r, 1.0, 1.0)
        )

    if provider == "mma_cpasync":
        r = torch.zeros(M, N, device="cuda", dtype=torch.float)
        mykernels.hgemm_mma_cpasync(x, w, r, 1.0, 1.0)
        torch.testing.assert_close(ref, r, rtol=1e-3, atol=1e-3)

        ms = triton.testing.do_bench(
            lambda: mykernels.hgemm_mma_cpasync(x, w, r, 1.0, 1.0)
        )

    if provider == "cutlass":
        r = torch.zeros(M, N, device="cuda", dtype=torch.float)
        mykernels.hgemm_cutlass(x, w, r, 1.0, 1.0)
        torch.testing.assert_close(ref, r, rtol=1e-3, atol=1e-3)

        ms = triton.testing.do_bench(lambda: mykernels.hgemm_cutlass(x, w, r, 1.0, 1.0))

    if provider == "torch":
        ms = triton.testing.do_bench(lambda: torch.matmul(x, w))

    gbps = lambda ms: x.nbytes * 1e-9 / (ms * 1e-3)

    return gbps(ms)


if __name__ == "__main__":
    benchmark_hgemm.run(print_data=True)
