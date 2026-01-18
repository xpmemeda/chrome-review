import triton
import torch
import kcuda.kernels as mykernels

BLOCK_SIZE = 16


class From_0_13_0:
    def __init__(self, k, v):
        import vllm._custom_ops as ops

        self.ops = ops

        self.n = n = k.size(1)
        self.d = d = k.size(3)
        self.x = x = 16 // k.element_size()

        self.k = (
            k.view(-1, n, BLOCK_SIZE, d // x, x).permute(0, 1, 3, 2, 4).contiguous()
        )
        self.v = v.permute(0, 1, 3, 2).contiguous()
        self.k_scale = self.v_scale = torch.tensor(
            1.0, dtype=torch.float32, device="cuda"
        )

    def forward(
        self,
        out,
        q,
        scale,
        block_tables,
        context_lens,
        max_context_len,
        alibi_slopes,
    ):
        self.ops.paged_attention_v1(
            out,
            q,
            self.k,
            self.v,
            self.n,
            scale,
            block_tables,
            context_lens,
            BLOCK_SIZE,
            max_context_len,
            alibi_slopes,
            "auto",
            self.k_scale,
            self.v_scale,
        )


def ref_fn(
    q: torch.Tensor,  # [b, n, d]
    k: torch.Tensor,  # [num_blocks, n, block_size, d]
    v: torch.Tensor,  # [num_blocks, n, block_size, d]
    scale,
    block_tables,
    context_lens,
    max_context_len,
    alibi_slopes,
):
    out = torch.empty_like(q)

    impl = From_0_13_0(k, v)
    impl.forward(
        out, q, scale, block_tables, context_lens, max_context_len, alibi_slopes
    )

    return out


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["b"],
        x_vals=[1, 8, 32, 128],
        line_arg="provider",
        line_vals=["vllm-integrated", "vllm-standalone", "tfcc"],
        line_names=["vllm-integrated", "vllm-standalone", "tfcc"],
        ylabel="GB/s",
        plot_name="spa-s-1024-n-32-d-128",
        args={"s": 1024, "n": 32, "d": 128},
    )
)
def benchmark_pa(b, s, n, d, provider):
    assert s % BLOCK_SIZE == 0, "s must be multiple of BLOCK_SIZE"
    num_blocks = b * s // BLOCK_SIZE

    torch.manual_seed(0)

    out = torch.empty(b, n, d, device="cuda", dtype=torch.half)
    q = torch.rand(b, n, d, device="cuda", dtype=torch.half)
    k = torch.rand(num_blocks, n, BLOCK_SIZE, d, device="cuda", dtype=torch.half)
    v = torch.rand(num_blocks, n, BLOCK_SIZE, d, device="cuda", dtype=torch.half)
    block_tables = torch.arange(num_blocks, device="cuda", dtype=torch.int32).reshape(
        b, -1
    )
    context_lens = torch.ones(b, device="cuda", dtype=torch.int32) * s
    max_context_len = s
    scale = 1.0 / (d**0.5)
    alibi_slopes = None

    ref = ref_fn(
        q, k, v, scale, block_tables, context_lens, max_context_len, alibi_slopes
    )

    ms = None
    if provider == "vllm-integrated":
        impl = From_0_13_0(k, v)

        ms = triton.testing.do_bench(
            lambda: impl.forward(
                out, q, scale, block_tables, context_lens, max_context_len, alibi_slopes
            )
        )

        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

    if provider == "vllm-standalone":
        x = 16 // q.element_size()
        k = k.view(-1, n, BLOCK_SIZE, d // x, x).permute(0, 1, 3, 2, 4).contiguous()
        v = v.permute(0, 1, 3, 2).contiguous()

        ms = triton.testing.do_bench(
            lambda: mykernels.vllm_spa(
                out,
                q,
                k,
                v,
                scale,
                block_tables,
                context_lens,
                max_context_len,
                alibi_slopes,
            )
        )

        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

    if provider == "tfcc":
        ms = triton.testing.do_bench(
            lambda: mykernels.tfcc_spa(
                out,
                q,
                k,
                v,
                scale,
                block_tables,
                context_lens,
                max_context_len,
                alibi_slopes,
            )
        )

        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

    gbps = lambda ms: out.nbytes * 1e-9 / (ms * 1e-3)

    return gbps(ms)


if __name__ == "__main__":
    benchmark_pa.run(print_data=True)
