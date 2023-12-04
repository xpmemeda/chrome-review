import torch
import kernels
import argparse


def do_bench(
    fn, warmup=25, rep=1000, grad_to_none=None, quantiles=None, fast_flush=True
):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    Args:
        fn: function to benchmark
        warmup: warmup time (in ms)
        rep: repetition time (in ms)
        grad_to_none: reset the gradient of the provided tensor to None
        quantiles: performance percentile to return in addition to the median.
        fast_flush: use faster kernel to flush L2 between measurements

    Returns:
       min, max, mean, median time cose in ms.
    """

    # Estimate the runtime of the function
    fn()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5
    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    # n_repeat = rep
    n_repeat = max(1, int(rep / estimate_ms))
    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    times = torch.tensor(sorted(times))
    drop_count = len(times) // 50
    drop_count = min(drop_count, 5)
    if drop_count > 0:
        times = times[drop_count:-drop_count]
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    min_value, max_value, mean_value, median_value = (
        torch.min(times),
        torch.max(times),
        torch.mean(times),
        torch.median(times),
    )
    return min_value, max_value, mean_value, median_value


def do_bench_and_report(name, fn):
    min_value, max_value, mean_value, median_value = do_bench(fn)
    print(
        "%s: min=%.5f, max=%.5f, mean=%.5f, median=%.5f"
        % (name, min_value, max_value, mean_value, median_value)
    )


def benchmark_softmax(sizes):
    x = torch.rand(*sizes, dtype=torch.float32, device="cuda:0")
    r = torch.zeros_like(x, dtype=torch.float32, device="cuda:0")
    do_bench_and_report("torch", lambda: torch.nn.functional.softmax(x, dim=-1))
    do_bench_and_report("softmax_v1", lambda: kernels.softmax_v1(x, r))
    do_bench_and_report("softmax_v2", lambda: kernels.softmax_v2(x, r))
    do_bench_and_report("softmax_v3", lambda: kernels.softmax_v3(x, r))
    do_bench_and_report("softmax_v4", lambda: kernels.softmax_v4(x, r))


def benchmark_gemm(sizes):
    x = torch.rand(*sizes, dtype=torch.float32, device="cuda:0")
    y = torch.rand(*sizes, dtype=torch.float32, device="cuda:0")
    r = torch.zeros(*sizes, dtype=torch.float32, device="cuda:0")
    do_bench_and_report("torch", lambda: torch.matmul(x, y))
    do_bench_and_report("gemm_v1", lambda: kernels.gemm_v1(x, y, r))
    do_bench_and_report("gemm_v2", lambda: kernels.gemm_v2(x, y, r))


def benchmark_conv(*args, **kwargs):
    group = 1
    paddings = [2, 2, 0]
    strides = [1, 1, 1]
    dilates = [1, 1, 1]
    x = torch.rand(1, 32, 402, 40, 1, dtype=torch.float32, device="cuda:0")
    w = torch.rand(32, 1, 5, 5, 1, dtype=torch.float32, device="cuda:0")
    r = torch.nn.functional.conv3d(x, w, None, strides, paddings, dilates, group)
    do_bench_and_report(
        "torch",
        lambda: torch.nn.functional.conv3d(
            x, w, None, strides, paddings, dilates, group
        ),
    )
    do_bench_and_report(
        "conv_v1", lambda: kernels.conv_v1(x, w, r, group, paddings, strides, dilates)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str)
    parser.add_argument("--sizes", nargs="+", type=int)
    args = parser.parse_args()
    if args.target == "softmax":
        return benchmark_softmax(args.sizes)
    if args.target == "gemm":
        return benchmark_gemm(args.sizes)
    if args.target == "conv":
        return benchmark_conv()
    assert False


if __name__ == "__main__":
    main()
