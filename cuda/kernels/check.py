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


def check_conv():
    dtype = torch.float32
    dtype = torch.float16

    group = 1
    paddings = [1, 0]
    strides = [1, 2]
    dilates = [1, 1]
    x = torch.rand(1, 64, 80, 41, dtype=dtype, device="cuda:0")
    w = torch.rand(192, 64, 3, 3, dtype=dtype, device="cuda:0")

    group = 1
    paddings = [1, 3]
    strides = [5, 1]
    dilates = [1, 1]
    x = torch.rand(1, 32, 402, 40, dtype=dtype, device="cuda:0")
    w = torch.rand(64, 32, 7, 7, dtype=dtype, device="cuda:0")

    group = 1
    paddings = [2, 2]
    strides = [1, 1]
    dilates = [1, 1]
    x = torch.rand(1, 1, 402, 40, dtype=dtype, device="cuda:0")
    w = torch.rand(32, 1, 5, 5, dtype=dtype, device="cuda:0")
    expected = torch.nn.functional.conv2d(x, w, None, strides, paddings, dilates, group)
    actual = torch.zeros_like(expected, dtype=dtype, device="cuda:0")
    kernels.conv_v1(x, w, actual, group, paddings, strides, dilates)
    torch.testing.assert_close(actual, expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str)
    args = parser.parse_args()
    if args.target == "softmax":
        return check_softmax()
    if args.target == "gemm":
        return check_gemm()
    if args.target == "conv":
        return check_conv()
    assert False


if __name__ == "__main__":
    main()
