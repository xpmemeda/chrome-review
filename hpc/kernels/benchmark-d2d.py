import triton
import torch


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["ChunkSize(MB)"],
        x_vals=[1, 8, 128, 1024, 4096],
        line_arg="provider",
        line_vals=[
            "cp",
        ],
        line_names=[
            "Cp(GB/s)",
        ],
        ylabel="GB/s",
        plot_name="d2d",
        args={},
    )
)
def benchmark_d2d(*args, **kwargs):
    chunk_size = kwargs["ChunkSize(MB)"]
    provider = kwargs["provider"]

    x = torch.empty(chunk_size, 1024 * 1024, device="cuda", dtype=torch.uint8)
    y = torch.empty(chunk_size, 1024 * 1024, device="cuda", dtype=torch.uint8)

    if provider == "cp":
        ms = triton.testing.do_bench(lambda: (y.copy_(x), torch.cuda.synchronize()))

    gbps = lambda ms: x.nbytes * 1e-9 / (ms * 1e-3)

    return gbps(ms)


if __name__ == "__main__":
    benchmark_d2d.run(print_data=True, save_path=".")
