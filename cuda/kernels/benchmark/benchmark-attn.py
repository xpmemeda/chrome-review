import argparse
import enum
import torch
import kernels
import random

from dobench import do_bench


class Layout_K(enum.Enum):
    CR = 1  # [num_blocks, num_heads, block_size, head_size]
    VLLM = 2  # [num_blocks, num_heads, head_size // 8, block_size, 8]


class Layout_V(enum.Enum):
    CR = 1  # [num_blocks, num_heads, block_size, head_size]
    VLLM = 2  # [num_blocks, num_heads, head_size, block_size]


class Benchmark:
    def __init__(
        self,
        num_seqs,
        num_heads,
        num_heads_kv,
        head_size,
        max_context_len,
        max_num_blocks,
        block_size,
        use_alibi,
        warmup=25,
        rep=1000,
    ):
        random.seed(0)
        torch.manual_seed(0)

        self.num_seqs = num_seqs
        self.num_heads = num_heads
        self.max_num_blocks = max_num_blocks
        self.num_heads_kv = num_heads_kv
        self.block_size = block_size
        self.head_size = head_size
        self.max_context_len = max_context_len
        self.num_blocks = (max_context_len + block_size - 1) // block_size

        self.q = torch.rand(
            num_seqs, num_heads, head_size, dtype=torch.half, device="cuda:0"
        )

        self.kc_A, self.kc_B = self.getKc()
        self.vc_A, self.vc_B = self.getVc()

        self.block_tables = self.getBlockTables(
            max_num_blocks, num_seqs, self.num_blocks
        )

        self.random_context_lens = self.getContextLens(num_seqs, max_context_len, True)
        self.fixed_context_lens = self.getContextLens(num_seqs, max_context_len, False)

        self.scale = 0.5
        self.alibi_slopes = self.getAlibiSlopes(use_alibi)

        self.warmup = warmup
        self.rep = rep

    def getKc(self):
        kc_A = torch.rand(
            self.max_num_blocks,
            self.num_heads_kv,
            self.block_size,
            self.head_size,
            dtype=torch.half,
            device="cuda:0",
        )

        kc_B = (
            kc_A.reshape(
                self.max_num_blocks,
                self.num_heads_kv,
                self.block_size,
                self.head_size // 8,
                8,
            )
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        return kc_A, kc_B

    def getVc(self):
        vc_A = torch.rand(
            self.max_num_blocks,
            self.num_heads_kv,
            self.block_size,
            self.head_size,
            dtype=torch.half,
            device="cuda:0",
        )

        vc_B = vc_A.permute(0, 1, 3, 2).contiguous()

        return vc_A, vc_B

    def getBlockTables(self, max_num_blocks, num_seqs, num_blocks):
        assert num_seqs * num_blocks <= max_num_blocks, "%d,%d,%d" % (
            num_seqs,
            num_blocks,
            max_num_blocks,
        )
        block_list = random.sample(range(max_num_blocks), num_seqs * num_blocks)
        return torch.tensor(block_list, dtype=torch.int32, device="cuda:0").reshape(
            num_seqs, num_blocks
        )

    def getContextLens(self, num_seqs, max_context_len, random_len):
        if random_len:
            context_lens = [
                random.sample(range(1, max_context_len + 1), 1) for _ in range(num_seqs)
            ]
        else:
            context_lens = [max_context_len for _ in range(num_seqs)]

        return torch.tensor(context_lens, dtype=torch.int32, device="cuda:0")

    def getAlibiSlopes(self, use_alibi):
        if not use_alibi:
            return None
        return torch.ones(self.num_heads, dtype=torch.float, device="cuda:0") / 10

    def clearL2Cache(self):
        cache = torch.empty(256 << 20, dtype=torch.int8, device="cuda:0")
        cache.zero_()
        return

    def calcKernel(self, kernel_name, kc_layout, vc_layout):
        out = torch.zeros(
            self.num_seqs,
            self.num_heads,
            self.head_size,
            dtype=torch.half,
            device="cuda:0",
        )
        self.clearL2Cache()
        kc = self.kc_A if kc_layout == Layout_K.CR else self.kc_B
        vc = self.vc_A if vc_layout == Layout_V.CR else self.vc_B

        kernel = getattr(kernels, kernel_name)
        kernel(
            out,
            self.q,
            kc,
            vc,
            self.scale,
            self.block_tables,
            self.random_context_lens,
            self.max_context_len,
            self.alibi_slopes,
        )

        return out

    def benchKernel(self, kernel_name, kc_layout, vc_layout):
        out = torch.zeros(
            self.num_seqs,
            self.num_heads,
            self.head_size,
            dtype=torch.half,
            device="cuda:0",
        )
        kc = self.kc_A if kc_layout == Layout_K.CR else self.kc_B
        vc = self.vc_A if vc_layout == Layout_V.CR else self.vc_B

        kernel = getattr(kernels, kernel_name)
        min_value, max_value, mean_value, median_value = do_bench(
            lambda: kernel(
                out,
                self.q,
                kc,
                vc,
                self.scale,
                self.block_tables,
                self.fixed_context_lens,
                self.max_context_len,
                self.alibi_slopes,
            ),
            warmup=self.warmup,
            rep=self.rep,
        )
        print(
            "%s: min=%.5f, max=%.5f, mean=%.5f, median=%.5f"
            % (kernel_name, min_value, max_value, mean_value, median_value)
        )

        return mean_value

    def checkResult(self, kernel_name, kc_layout, vc_layout):
        x = self.calcKernel("paged_attention_v1_1", Layout_K.VLLM, Layout_V.VLLM)
        y = self.calcKernel(kernel_name, kc_layout, vc_layout)
        if not torch.allclose(x, y, atol=1e-3, rtol=1e-3):
            print(f"{kernel_name} err.")
        torch.testing.assert_close(x, y, atol=1e-3, rtol=1e-3)

    def checkResults(self):
        self.checkResult("paged_attention_v1_1", Layout_K.VLLM, Layout_V.VLLM)
        self.checkResult("paged_attention_v1_2", Layout_K.VLLM, Layout_V.VLLM)
        # self.checkResult("paged_attention_v1_4", Layout_K.CR, Layout_V.VLLM)
        # self.checkResult("paged_attention_v1_5", Layout_K.VLLM, Layout_V.CR)
        self.checkResult("paged_attention_v1_8", Layout_K.CR, Layout_V.CR)
        self.checkResult("paged_attention_v1_9", Layout_K.CR, Layout_V.CR)
        self.checkResult("paged_attention_v2_1", Layout_K.VLLM, Layout_V.VLLM)
        self.checkResult("paged_attention_v2_2", Layout_K.VLLM, Layout_V.VLLM)
        self.checkResult("paged_attention_v2_8", Layout_K.CR, Layout_V.CR)

    def doBench(self):
        self.benchKernel("paged_attention_v1_1", Layout_K.VLLM, Layout_V.VLLM)
        self.benchKernel("paged_attention_v1_2", Layout_K.VLLM, Layout_V.VLLM)
        # self.benchKernel("paged_attention_v1_4", Layout_K.CR, Layout_V.VLLM)
        # self.benchKernel("paged_attention_v1_5", Layout_K.VLLM, Layout_V.CR)
        self.benchKernel("paged_attention_v1_8", Layout_K.CR, Layout_V.CR)
        self.benchKernel("paged_attention_v1_9", Layout_K.CR, Layout_V.CR)
        self.benchKernel("paged_attention_v2_1", Layout_K.VLLM, Layout_V.VLLM)
        self.benchKernel("paged_attention_v2_2", Layout_K.VLLM, Layout_V.VLLM)
        self.benchKernel("paged_attention_v2_8", Layout_K.CR, Layout_V.CR)


if __name__ == "__main__":
    r"""
    Model           num_heads   num_heads_kv    head_size
    -----------------------------------------------------
    Baichuan-7B     32          32              128
    Baichuan2-7B    32          32              128
    Baichuan2-13B   40          40              128
    chatglm3-6b     32          2               128
    chatglm4-9b     32          2               128
    Hunyuan0.5B     10          10              128
    Hunyuan7B       32          32              128
    Qwen1.5-0.5B    16          16              64
    Qwen1.5-1.8B    16          16              128
    Qwen1.5-7B      32          32              128
    Qwen2-0.5B      14          2               64
    Qwen2-1.5B      12          2               128
    Qwen2-7B        28          4               128
    Qwen2-VL-2B     12          2               128
    Qwen2.5-0.5B    14          2               64
    Qwen2.5-1.8B    12          2               128
    Qwen2.5-7B      28          4               128
    Qwen2.5-14B     40          8               128
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", type=int, required=True)
    parser.add_argument("-s", type=int, required=True)
    parser.add_argument("-nq", type=int, required=True)
    parser.add_argument("-nk", type=int, required=True)
    parser.add_argument("-d", type=int, required=True)
    arguments = parser.parse_args()
    num_seqs = arguments.b
    context_len = arguments.s
    num_heads = arguments.nq
    num_heads_kv = arguments.nk
    head_size = arguments.d

    block_size = 16
    num_blocks_per_seq = (context_len + block_size - 1) // block_size
    num_blocks = num_blocks_per_seq * num_seqs * 2
    use_alibi = True

    benchmark = Benchmark(
        num_seqs,
        num_heads,
        num_heads_kv,
        head_size,
        context_len,
        num_blocks,
        block_size,
        use_alibi,
    )
    benchmark.checkResults()
    benchmark.doBench()
