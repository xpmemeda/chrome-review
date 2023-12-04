import kernels
import torch
import flashinfer
from dobench import do_bench


def topkp_1(logits, ks, ps):
    probs = torch.empty_like(logits)
    kernels.topkp_1(probs, logits, ks, ps)
    return probs


def topkp_2(logits, ks, ps):
    masked_logits = flashinfer.sampling.top_k_mask_logits(logits, ks)
    masked_logits = masked_logits.softmax(-1)
    masked_logits = flashinfer.sampling.top_p_renorm_probs(masked_logits, ps)
    return masked_logits


class Benchmark:
    def __init__(self, num_tokens, vocab_size):
        self.num_tokens = num_tokens
        self.vocab_size = vocab_size

        self.logits = torch.rand(
            num_tokens, vocab_size, device="cuda:0", dtype=torch.float32
        )
        self.ks = torch.randint(
            2, 5, size=(num_tokens,), device="cuda:0", dtype=torch.int64
        )
        self.ps = (
            torch.rand(num_tokens, device="cuda:0", dtype=torch.float32) * 0.5 + 0.5
        )

    def verifyResults(self):
        probs1 = topkp_1(self.logits, self.ks, self.ps)
        probs2 = topkp_2(self.logits, self.ks, self.ps)
        torch.testing.assert_close(probs1, probs2)

    def doBench(self):
        min_value, max_value, mean_value, median_value = do_bench(
            lambda: topkp_1(self.logits, self.ks, self.ps)
        )
        print(
            "topkp_1: min=%.5f, max=%.5f, mean=%.5f, median=%.5f"
            % (min_value, max_value, mean_value, median_value)
        )

        min_value, max_value, mean_value, median_value = do_bench(
            lambda: topkp_2(self.logits, self.ks, self.ps)
        )
        print(
            "topkp_2: min=%.5f, max=%.5f, mean=%.5f, median=%.5f"
            % (min_value, max_value, mean_value, median_value)
        )


if __name__ == "__main__":
    torch.manual_seed(0)

    benchmark = Benchmark(16, 152064)
    benchmark.verifyResults()
    benchmark.doBench()
