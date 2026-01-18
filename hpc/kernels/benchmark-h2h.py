import triton
import torch


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["ChunkSize(MB)"],
        x_vals=[1, 8, 128, 1024, 4096],
        line_arg="provider",
        line_vals=[
            "cp_pag_to_pag",
            "cp_pag_to_pin",
            "cp_pin_to_pin",
            "cp_pin_to_pag",
        ],
        line_names=[
            "CpPagToPag(GB/s)",
            "CpPagToPin(GB/s)",
            "CpPinToPin(GB/s)",
            "CpPinToPag(GB/s)",
        ],
        ylabel="GB/s",
        plot_name="h2h",
        args={},
    )
)
def benchmark_h2h(*args, **kwargs):
    chunk_size = kwargs["ChunkSize(MB)"]
    provider = kwargs["provider"]

    x = torch.empty(chunk_size, 1024 * 1024, dtype=torch.uint8)
    y = torch.empty(chunk_size, 1024 * 1024, dtype=torch.uint8)

    a = torch.empty(chunk_size, 1024 * 1024, dtype=torch.uint8).pin_memory()
    b = torch.empty(chunk_size, 1024 * 1024, dtype=torch.uint8).pin_memory()

    if provider == "cp_pag_to_pag":
        ms = triton.testing.do_bench(lambda: y.copy_(x))
    if provider == "cp_pag_to_pin":
        ms = triton.testing.do_bench(lambda: b.copy_(x))
    if provider == "cp_pin_to_pin":
        ms = triton.testing.do_bench(lambda: b.copy_(a))
    if provider == "cp_pin_to_pag":
        ms = triton.testing.do_bench(lambda: y.copy_(a))

    gbps = lambda ms: x.nbytes * 1e-9 / (ms * 1e-3)

    return gbps(ms)


if __name__ == "__main__":
    benchmark_h2h.run(print_data=True, save_path=".")
