import torch

import triton
import triton.language as tl

try:
    # This is https://github.com/NVIDIA/apex, NOT the apex on PyPi, so it
    # should not be added to extras_require in setup.py.
    import apex

    HAS_APEX = True
except ModuleNotFoundError:
    HAS_APEX = False

# DEVICE = triton.runtime.driver.active.get_active_torch_device()
DEVICE = torch.device("cuda:0")


@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)


def layernorm(x, weight, bias, eps):
    # allocate output
    y = torch.empty_like(x)
    # reshape input data into 2D tensor
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    mean = torch.empty((M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    # enqueue kernel
    _layer_norm_fwd_fused[(M,)](  #
        x_arg,
        y,
        weight,
        bias,
        mean,
        rstd,  #
        x_arg.stride(0),
        N,
        eps,  #
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_ctas=1,
    )
    return y


def test_layer_norm(M, N, dtype, eps=1e-5, device=DEVICE):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    # forward pass
    y_tri = layernorm(x, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    # compare
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg="provider",
        line_vals=["triton", "torch"] + (["apex"] if HAS_APEX else []),
        line_names=["Triton", "Torch"] + (["Apex"] if HAS_APEX else []),
        styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
        ylabel="GB/s",
        plot_name="layer-norm",
        args={"M": 4096, "dtype": torch.float16},
    )
)
def bench_layer_norm(M, N, dtype, provider, eps=1e-5, device=DEVICE):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():

        if provider == "triton":
            return layernorm(x, weight, bias, eps)  # noqa: F811, E704

        if provider == "torch":
            return torch.nn.functional.layer_norm(
                x, w_shape, weight, bias, eps
            )  # noqa: F811, E704

        if provider == "apex":
            apex_layer_norm = (
                apex.normalization.FusedLayerNorm(w_shape).to(x.device).to(x.dtype)
            )
            return apex_layer_norm(x)  # noqa: F811, E704

    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    test_layer_norm(1151, 8192, torch.float16)
    bench_layer_norm.run(save_path=".", print_data=True)
