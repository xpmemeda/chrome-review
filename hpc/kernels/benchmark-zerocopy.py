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
            "zero_copy_y",
            "zero_copy_n",
            "zero_copy_n_pin",
            "cpu",
            "gpu",
        ],
        line_names=[
            "zeroCp(GB/s)",
            "2Cp(GB/s)",
            "2Cp+Pin(GB/s)",
            "PureCpu(GB/s)",
            "PureGpu(GB/s)",
        ],
        ylabel="GB/s",
        plot_name="zerocopy",
        args={},
    )
)
def benchmark_d2d(N, provider):

    y = torch.empty(N, N, dtype=torch.int)
    n = torch.empty(N, N, dtype=torch.int)
    n_pin = n.pin_memory()
    c = torch.empty(N, N, dtype=torch.int)
    g = torch.empty(N, N, dtype=torch.int, device="cuda")

    if provider == "zero_copy_y":
        ms = triton.testing.do_bench(lambda: mykernels.zero_copy_y(y))
    if provider == "zero_copy_n":
        ms = triton.testing.do_bench(lambda: mykernels.zero_copy_n(n))
    if provider == "zero_copy_n_pin":
        ms = triton.testing.do_bench(lambda: mykernels.zero_copy_n(n_pin))
    if provider == "cpu":
        torch.set_num_threads(1)
        ms = triton.testing.do_bench(lambda: c.mul_(2))
    if provider == "gpu":
        ms = triton.testing.do_bench(lambda: g.mul_(2))

    gbps = lambda ms: y.nbytes * 1e-9 / (ms * 1e-3)

    return gbps(ms)


if __name__ == "__main__":
    benchmark_d2d.run(print_data=True, save_path=".")
