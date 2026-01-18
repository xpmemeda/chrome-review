import triton
import torch


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["ChunkSize(MB)"],
        x_vals=[1, 8, 128, 1024, 4096],
        line_arg="provider",
        line_vals=[
            "to",
            "cp_pag",
            "cp_pin",
        ],
        line_names=[
            "To(GB/s)",
            "CpPag(GB/s)",
            "CpPin(GB/s)",
        ],
        ylabel="GB/s",
        plot_name="h2d",
        args={},
    )
)
def benchmark_h2d(*args, **kwargs):
    chunk_size = kwargs["ChunkSize(MB)"]
    provider = kwargs["provider"]

    x = torch.empty(chunk_size, 1024 * 1024, device="cuda", dtype=torch.uint8)
    y = torch.ones(chunk_size, 1024 * 1024, dtype=torch.uint8)
    z = y.pin_memory()

    if provider == "to":
        ms = triton.testing.do_bench(lambda: y.to("cuda"))
    if provider == "cp_pag":
        ms = triton.testing.do_bench(lambda: (x.copy_(y), torch.cuda.synchronize()))
    if provider == "cp_pin":
        ms = triton.testing.do_bench(
            lambda: (x.copy_(z, non_blocking=True), torch.cuda.synchronize())
        )

    gbps = lambda ms: x.nbytes * 1e-9 / (ms * 1e-3)

    return gbps(ms)


if __name__ == "__main__":
    benchmark_h2d.run(print_data=True, save_path=".")
