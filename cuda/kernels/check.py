import argparse
import torch
import kernels


def softmax_v1(x):
    r = torch.zeros_like(x, dtype=torch.float32, device="cuda:0")
    kernels.softmax_v1(x, r)
    return r


def softmax_v2(x):
    r = torch.zeros_like(x, dtype=torch.float32, device="cuda:0")
    kernels.softmax_v2(x, r)
    return r


def softmax_v3(x):
    r = torch.zeros_like(x, dtype=torch.float32, device="cuda:0")
    kernels.softmax_v3(x, r)
    return r


def softmax_v4(x):
    r = torch.zeros_like(x, dtype=torch.float32, device="cuda:0")
    if x.size(-1) <= 1024:
        kernels.softmax_v4(x, r)
    else:
        kernels.softmax_v3(x, r)
    return r


def check_softmax():
    for numel in [1, 7, 8, 33, 256, 257, 1023, 1024, 4399]:
        x = torch.rand(numel, device="cuda:0").contiguous()
        expected = torch.nn.functional.softmax(x, dim=0)
        torch.testing.assert_close(softmax_v1(x), expected)
        torch.testing.assert_close(softmax_v2(x), expected)
        torch.testing.assert_close(softmax_v3(x), expected)
        torch.testing.assert_close(softmax_v4(x), expected)

        x = torch.rand(64, numel, device="cuda:0").contiguous()
        expected = torch.nn.functional.softmax(x, dim=len(x.size()) - 1)
        torch.testing.assert_close(softmax_v1(x), expected)
        torch.testing.assert_close(softmax_v2(x), expected)
        torch.testing.assert_close(softmax_v3(x), expected)
        torch.testing.assert_close(softmax_v4(x), expected)

        x = torch.rand(64, 64, numel, device="cuda:0").contiguous()
        expected = torch.nn.functional.softmax(x, dim=len(x.size()) - 1)
        torch.testing.assert_close(softmax_v1(x), expected)
        torch.testing.assert_close(softmax_v2(x), expected)
        torch.testing.assert_close(softmax_v3(x), expected)
        torch.testing.assert_close(softmax_v4(x), expected)


def gemm_v1(a, b):
    m = a.size(0)
    n = b.size(1)
    c = torch.zeros(m, n, dtype=torch.float32, device="cuda:0")
    kernels.gemm_v1(a, b, c)
    return c


def gemm_v2(a, b):
    m = a.size(0)
    n = b.size(1)
    c = torch.zeros(m, n, dtype=torch.float32, device="cuda:0")
    kernels.gemm_v2(a, b, c)
    return c


def check_gemm():
    a = torch.rand(1024, 1024, dtype=torch.float32, device="cuda:0")
    b = torch.rand(1024, 1024, dtype=torch.float32, device="cuda:0")
    expected = torch.matmul(a, b)
    torch.testing.assert_close(gemm_v1(a, b), expected)
    torch.testing.assert_close(gemm_v2(a, b), expected, atol=5e-2, rtol=1e-3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, choices=["softmax", "gemm"])
    args = parser.parse_args()
    if args.target == "softmax":
        return check_softmax()
    if args.target == "gemm":
        return check_gemm()
    assert False


if __name__ == "__main__":
    main()
