import kernels
import torch
from dobench import do_bench


def sort_1(x):
    r = torch.empty_like(x)
    kernels.sort_1(r, x)
    return r


def sort_2(x):
    r, _ = torch.sort(x)
    return r


class Benchmark:
    def __init__(self, num_tokens, vocab_size):
        self.src = torch.rand(num_tokens, vocab_size, dtype=torch.float16).to(0)

    def checkResults(self):
        torch.testing.assert_close(sort_1(self.src), sort_2(self.src))

    def doBench(self):
        min_value, max_value, mean_value, median_value = do_bench(
            lambda: sort_1(self.src)
        )
        print(
            "sort_1: min=%.5f, max=%.5f, mean=%.5f, median=%.5f"
            % (min_value, max_value, mean_value, median_value)
        )

        min_value, max_value, mean_value, median_value = do_bench(
            lambda: sort_2(self.src)
        )
        print(
            "sort_2: min=%.5f, max=%.5f, mean=%.5f, median=%.5f"
            % (min_value, max_value, mean_value, median_value)
        )


if __name__ == "__main__":
    benchmark = Benchmark(32, 152400)
    benchmark.checkResults()
    benchmark.doBench()
