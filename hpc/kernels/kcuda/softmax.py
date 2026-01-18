if __name__ != "__main__":
    import torch

    from .kernels import softmax_v3, softmax_v4
    from .kernels import softmax_v5 as softmax_v5_internal

    def softmax(x: torch.Tensor):
        r = torch.empty_like(x)
        if x.size(-1) <= 4096:
            softmax_v4(x, r)
        else:
            softmax_v3(x, r)
        return r

else:

    import triton
    import torch
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.__file__))

    import kernels

    torch.manual_seed(0)

    def test():
        x = torch.rand(1, 1025, device="cuda:0")
        r = torch.empty_like(x)
        ref = torch.nn.functional.softmax(x, dim=-1)
        kernels.softmax_v1(x, r)
        torch.testing.assert_close(r, ref)
        kernels.softmax_v2(x, r)
        torch.testing.assert_close(r, ref)
        kernels.softmax_v3(x, r)
        torch.testing.assert_close(r, ref)
        kernels.softmax_v4(x, r)
        torch.testing.assert_close(r, ref)
        kernels.softmax_v5(x, r)
        torch.testing.assert_close(r, ref)

    test()

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[128 * i for i in range(2, 100)],
            line_arg="provider",
            line_vals=["v1", "v2", "v3", "v4", "v5"],
            line_names=["v1", "v2", "v3", "v4", "v5"],
            styles=[
                ("blue", "-"),
                ("green", "-"),
                ("red", "-"),
                ("yellow", "-"),
                ("black", "-"),
            ],
            ylabel="GB/s",
            plot_name="softmax",
            args={"M": 4096},
        )
    )
    def benchmark(M, N, provider, device=torch.device("cuda:0")):
        x = torch.randn(M, N, device=device, dtype=torch.float32)
        r = torch.empty_like(x)

        if provider == "v1":
            ms = triton.testing.do_bench(lambda: kernels.softmax_v1(x, r))
        if provider == "v2":
            ms = triton.testing.do_bench(lambda: kernels.softmax_v2(x, r))
        if provider == "v3":
            ms = triton.testing.do_bench(lambda: kernels.softmax_v3(x, r))
        if provider == "v4":
            ms = triton.testing.do_bench(lambda: kernels.softmax_v4(x, r))
        if provider == "v5":
            ms = triton.testing.do_bench(lambda: kernels.softmax_v5(x, r))

        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

        return gbps(ms)

    benchmark.run(print_data=True, save_path=".")
