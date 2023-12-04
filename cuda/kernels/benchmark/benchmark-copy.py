import sys
import kernels
import torch

from dobench import do_bench


class Benchmark:
    def __init__(self, n, c, h, w):
        self.n = n
        self.c = c
        self.h = h
        self.w = w

        self.src = torch.rand(n, c, h, w, device="cuda:0", dtype=torch.half)

    def checkResults(self):
        out_1 = torch.zeros(
            self.n, self.c, self.h, self.w, device="cuda:0", dtype=torch.half
        )
        kernels.copy_1(out_1, self.src)
        torch.testing.assert_close(out_1, self.src)

        out_2 = torch.zeros(
            self.n, self.c, self.h, self.w, device="cuda:0", dtype=torch.half
        )
        kernels.copy_2(out_2, self.src)
        torch.testing.assert_close(out_2, self.src)

        out_3 = torch.zeros(
            self.n, self.c, self.h, self.w, device="cuda:0", dtype=torch.half
        )
        kernels.copy_3(out_3, self.src)
        torch.testing.assert_close(out_3, self.src)

        out_4 = torch.zeros(
            self.n, self.c, self.h, self.w, device="cuda:0", dtype=torch.half
        )
        kernels.copy_4(out_4, self.src)
        torch.testing.assert_close(out_4, self.src)

    def doBench(self):
        out = torch.zeros(
            self.n, self.c, self.h, self.w, device="cuda:0", dtype=torch.half
        )
        min_value, max_value, mean_value, median_value = do_bench(
            lambda: kernels.copy_1(out, self.src)
        )
        print(
            "copy_1: min=%.5f, max=%.5f, mean=%.5f, median=%.5f"
            % (min_value, max_value, mean_value, median_value)
        )

        min_value, max_value, mean_value, median_value = do_bench(
            lambda: kernels.copy_2(out, self.src)
        )
        print(
            "copy_2: min=%.5f, max=%.5f, mean=%.5f, median=%.5f"
            % (min_value, max_value, mean_value, median_value)
        )

        min_value, max_value, mean_value, median_value = do_bench(
            lambda: kernels.copy_3(out, self.src)
        )
        print(
            "copy_3: min=%.5f, max=%.5f, mean=%.5f, median=%.5f"
            % (min_value, max_value, mean_value, median_value)
        )

        min_value, max_value, mean_value, median_value = do_bench(
            lambda: kernels.copy_4(out, self.src)
        )
        print(
            "copy_4: min=%.5f, max=%.5f, mean=%.5f, median=%.5f"
            % (min_value, max_value, mean_value, median_value)
        )


if __name__ == "__main__":
    benchmark = Benchmark(32, 32, 1024, 1024)
    benchmark.checkResults()
    benchmark.doBench()
