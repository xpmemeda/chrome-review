import triton
import torch

import kcuda.kernels as mykernels


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[1, 8, 128, 1024, 8192],
        line_arg="provider",
        line_vals=[
            "to",
            "cp_pag",
            "cp_pin",
            "kernel_pin",
        ],
        line_names=["To(GB/s)", "CpPag(GB/s)", "CpPin(GB/s)", "KernelPin(GB/s)"],
        ylabel="GB/s",
        plot_name="d2h",
        args={"N": 4096, "dtype": torch.uint16},
    )
)
def benchmark_d2h(M, N, dtype, provider):
    x = torch.rand(M, N, device="cuda").mul_(100).to(dtype)
    ref = x.to("cpu")

    if provider == "to":
        r = x.to("cpu")
        torch.testing.assert_close(ref, r)

        ms = triton.testing.do_bench(lambda: x.to("cpu"))

    if provider == "cp_pag":
        r = torch.zeros_like(x, device="cpu")
        r.copy_(x)
        torch.testing.assert_close(ref, r)

        ms = triton.testing.do_bench(lambda: r.copy_(x))

    if provider == "cp_pin":
        r = torch.zeros_like(x, device="cpu").pin_memory()
        r.copy_(x, non_blocking=True)
        torch.cuda.synchronize()
        torch.testing.assert_close(ref, r)

        ms = triton.testing.do_bench(lambda: r.copy_(x, non_blocking=True))

    if provider == "kernel_pin":
        r = torch.zeros_like(x, device="cpu")

        register = mykernels.CudaMappedTensorManager.get()
        register.register(r)

        mykernels.d2h(x, r)
        torch.cuda.synchronize()
        torch.testing.assert_close(ref, r)

        ms = triton.testing.do_bench(lambda: mykernels.d2h(x, r))

        register.release(r)

    gbps = lambda ms: x.nbytes * 1e-9 / (ms * 1e-3)

    return gbps(ms)


if __name__ == "__main__":
    benchmark_d2h.run(print_data=True)
