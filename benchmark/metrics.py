import math
import statistics
import typing as ty

JsonDict = ty.Dict[str, ty.Any]


class RequestMetrics(ty.NamedTuple):
    req_idx: int
    ok: bool
    ttft: float
    e2e: float
    output_tokens: int
    output_chars: int
    output_chunks: int
    itls: ty.Tuple[float, ...] = ()
    error: str = ""
    output_text: str = ""
    server_output_tokens: ty.Optional[int] = None
    server_input_tokens: ty.Optional[int] = None
    server_cached_tokens: ty.Optional[int] = None
    server_usage: ty.Optional[JsonDict] = None
    server_raw_chunks: ty.Tuple[str, ...] = ()
    x_tt_logid: ty.Optional[str] = None
    client_send_timestamp: ty.Optional[str] = None


def percentile(values: ty.List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    xs = sorted(values)
    rank = (len(xs) - 1) * p / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return xs[int(rank)]
    return xs[lo] * (hi - rank) + xs[hi] * (rank - lo)


def mean(values: ty.List[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def summarize(metrics: ty.List[RequestMetrics], elapsed: float, label: str) -> JsonDict:
    oks = [m for m in metrics if m.ok]
    errs = [m for m in metrics if not m.ok]
    ttfts = [m.ttft for m in oks]
    e2es = [m.e2e for m in oks]
    itls = [itl for m in oks for itl in m.itls]
    output_tokens = sum(m.output_tokens for m in oks)
    output_chars = sum(m.output_chars for m in oks)
    output_chunks = sum(m.output_chunks for m in oks)
    return {
        "label": label,
        "requests": len(metrics),
        "success": len(oks),
        "errors": len(errs),
        "success_rate": len(oks) / len(metrics) if metrics else 0.0,
        "elapsed_sec": elapsed,
        "rps": len(oks) / elapsed if elapsed > 0 else 0.0,
        "output_tokens_per_sec": output_tokens / elapsed if elapsed > 0 else 0.0,
        "output_chars_per_sec": output_chars / elapsed if elapsed > 0 else 0.0,
        "output_chunks_per_sec": output_chunks / elapsed if elapsed > 0 else 0.0,
        "avg_output_tokens": output_tokens / len(oks) if oks else 0.0,
        "avg_output_chars": output_chars / len(oks) if oks else 0.0,
        "ttft_avg": mean(ttfts),
        "ttft_p50": percentile(ttfts, 50),
        "ttft_p90": percentile(ttfts, 90),
        "ttft_p99": percentile(ttfts, 99),
        "itl_avg": mean(itls),
        "itl_p50": percentile(itls, 50),
        "itl_p90": percentile(itls, 90),
        "itl_p99": percentile(itls, 99),
        "e2e_avg": mean(e2es),
        "e2e_p50": percentile(e2es, 50),
        "e2e_p90": percentile(e2es, 90),
        "e2e_p99": percentile(e2es, 99),
        "first_errors": [m.error for m in errs[:5]],
    }
