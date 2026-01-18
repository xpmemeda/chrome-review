import triton
import torch

import kcuda.kernels as mykernels


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[1024, 4096],
        line_arg="provider",
        line_vals=[
            "bank_conflict_01",
            "bank_conflict_02",
            "bank_conflict_16",
            "bank_conflict_17",
        ],
        line_names=[
            "BankConflict01(GB/s)",
            "BankConflict02(GB/s)",
            "BankConflict16(GB/s)",
            "BankConflict17(GB/s)",
        ],
        ylabel="GB/s",
        plot_name="bankconflict",
        args={},
    )
)
def benchmark_bankconflict(N, provider):

    x = torch.empty(N, dtype=torch.float, device="cuda")
    y = torch.empty(N, dtype=torch.float, device="cuda")

    if provider == "bank_conflict_01":
        ms = triton.testing.do_bench(lambda: mykernels.benchmark_bankconflict(x, 1))
    if provider == "bank_conflict_02":
        ms = triton.testing.do_bench(lambda: mykernels.benchmark_bankconflict(x, 2))
    if provider == "bank_conflict_16":
        ms = triton.testing.do_bench(lambda: mykernels.benchmark_bankconflict(x, 16))
    if provider == "bank_conflict_17":
        ms = triton.testing.do_bench(lambda: mykernels.benchmark_bankconflict(x, 17))

    gbps = lambda ms: y.nbytes * 1e-9 / (ms * 1e-3)

    return gbps(ms)


if __name__ == "__main__":
    benchmark_bankconflict.run(print_data=True, save_path=".")
