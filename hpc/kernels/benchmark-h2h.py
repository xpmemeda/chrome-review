import math
import os
import pathlib
import random
import re
import time

import triton
import torch


kDefaultLlcSizeBytes = 32 * 1024 * 1024
kPrintedSystemMemoryInfo = False
kPrintedTorchThreadInfo = False
kPrintedAffinityInfo = False


def format_bytes(num_bytes):
    """Formats a byte count into a human-readable IEC string."""
    if num_bytes >= 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024 * 1024):.2f} GiB"
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MiB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KiB"
    return f"{num_bytes} B"


def detect_cache_info():
    """Reads CPU cache metadata from sysfs and returns per-level capacities."""
    cache_root = pathlib.Path("/sys/devices/system/cpu/cpu0/cache")
    if not cache_root.exists():
        return {}, kDefaultLlcSizeBytes

    llc_size_bytes = None
    llc_level = -1
    cache_info = {}
    for index_dir in cache_root.glob("index*"):
        try:
            level = int((index_dir / "level").read_text().strip())
            cache_type = (index_dir / "type").read_text().strip()
            size_text = (index_dir / "size").read_text().strip()
        except (FileNotFoundError, ValueError, OSError):
            continue

        match = re.fullmatch(r"(\d+)([KMG])", size_text)
        if not match:
            continue

        value = int(match.group(1))
        unit = match.group(2)
        scale = {
            "K": 1024,
            "M": 1024 * 1024,
            "G": 1024 * 1024 * 1024,
        }
        size_bytes = value * scale[unit]
        cache_key = f"L{level}{cache_type[0].lower()}"
        if cache_type == "Unified":
            cache_key = f"L{level}"
        cache_info[cache_key] = max(cache_info.get(cache_key, 0), size_bytes)

        if cache_type not in ("Unified", "Data") or size_bytes is None:
            continue
        if level > llc_level:
            llc_level = level
            llc_size_bytes = size_bytes
        elif level == llc_level and llc_size_bytes is not None:
            llc_size_bytes = max(llc_size_bytes, size_bytes)

    if llc_size_bytes is None:
        llc_size_bytes = kDefaultLlcSizeBytes
    return cache_info, llc_size_bytes


def detect_memory_size_bytes():
    """Reads the total system memory size from /proc/meminfo."""
    meminfo_path = pathlib.Path("/proc/meminfo")
    if not meminfo_path.exists():
        return None

    try:
        for line in meminfo_path.read_text().splitlines():
            match = re.fullmatch(r"MemTotal:\s+(\d+)\s+kB", line)
            if match:
                return int(match.group(1)) * 1024
    except OSError:
        return None
    return None


def print_torch_thread_info():
    """Prints PyTorch thread settings and how to override them."""
    global kPrintedTorchThreadInfo
    if kPrintedTorchThreadInfo:
        return
    print("TorchThreads: ")
    print(f"  intra_op: {torch.get_num_threads()}")
    print(f"  inter_op: {torch.get_num_interop_threads()}")
    print(
        "  Hint: set OMP_NUM_THREADS and MKL_NUM_THREADS to control CPU thread counts"
    )

    kPrintedTorchThreadInfo = True


def print_affinity_info():
    """Prints whether the current process is restricted to a CPU affinity mask."""
    global kPrintedAffinityInfo
    if kPrintedAffinityInfo:
        return

    total_cpu_count = os.cpu_count()
    try:
        allowed_cpus = sorted(os.sched_getaffinity(0))
    except AttributeError:
        print("AffinityInfo:")
        print("  status: unavailable")
        kPrintedAffinityInfo = True
        return

    allowed_cpu_count = len(allowed_cpus)
    affinity_enabled = (
        total_cpu_count is not None and allowed_cpu_count < total_cpu_count
    )
    print("AffinityInfo:")
    print(f"  enabled: {'yes' if affinity_enabled else 'no'}")
    if total_cpu_count is not None:
        print(f"  allowed_cpus: {allowed_cpu_count}/{total_cpu_count}")
    else:
        print(f"  allowed_cpus: {allowed_cpu_count}")
    if allowed_cpus:
        preview = ",".join(str(cpu_id) for cpu_id in allowed_cpus[:16])
        suffix = "..." if allowed_cpu_count > 16 else ""
        print(f"  cpu_list_preview: {preview}{suffix}")
    print(
        "  Hint: use numactl --cpunodebind=<node> --membind=<node> to control "
        "CPU affinity and memory placement."
    )

    kPrintedAffinityInfo = True


def print_system_memory_info():
    """Prints cache and memory information once before benchmarks start."""
    global kPrintedSystemMemoryInfo
    if kPrintedSystemMemoryInfo:
        return

    cache_info, llc_size_bytes = detect_cache_info()
    memory_size_bytes = detect_memory_size_bytes()

    print("SystemMemoryInfo:")
    if cache_info:
        for cache_name in sorted(cache_info.keys()):
            print(f"  {cache_name}: {format_bytes(cache_info[cache_name])}")
    else:
        print("  Cache: unavailable")
    print(f"  LLC: {format_bytes(llc_size_bytes)}")
    if memory_size_bytes is not None:
        print(f"  MemTotal: {format_bytes(memory_size_bytes)}")
    else:
        print("  MemTotal: unavailable")

    kPrintedSystemMemoryInfo = True


def allocate_host_buffer(num_chunks, chunk_bytes, pinned):
    """Allocates a host buffer, optionally backed by pinned pages."""
    tensor = torch.empty(num_chunks * chunk_bytes, dtype=torch.uint8)
    if pinned:
        tensor = tensor.pin_memory()
    return tensor


def touch_buffer_pages(buffer, page_bytes=4096):
    """Touches one byte per page so page faults do not pollute the benchmark."""
    numel = buffer.numel()
    for offset in range(0, numel, page_bytes):
        buffer[offset] = 0
    if numel > 0:
        buffer[numel - 1] = 0


def build_rotating_copy_fn(src, dst, chunk_bytes, llc_size_bytes, seed=0):
    """Builds a copy closure that interleaves chunks across LLC-sized regions."""
    total_chunks = src.numel() // chunk_bytes
    if total_chunks <= 1:
        chunk_order = [0]
    else:
        chunks_per_region = max(1, llc_size_bytes // chunk_bytes)
        num_regions = max(1, math.ceil(total_chunks / chunks_per_region))
        regions = []
        for region_index in range(num_regions):
            start = region_index * chunks_per_region
            end = min(start + chunks_per_region, total_chunks)
            regions.append(list(range(start, end)))

        rng = random.Random(seed)
        for region in regions:
            rng.shuffle(region)

        chunk_order = []
        region_indices = list(range(num_regions))
        while len(chunk_order) < total_chunks:
            active_regions = [index for index in region_indices if regions[index]]
            rng.shuffle(active_regions)
            for region_index in active_regions:
                if regions[region_index]:
                    chunk_order.append(regions[region_index].pop())

    index = {"value": 0}

    def copy_once():
        chunk_index = chunk_order[index["value"]]
        start = chunk_index * chunk_bytes
        end = start + chunk_bytes
        dst[start:end].copy_(src[start:end])
        index["value"] = (index["value"] + 1) % total_chunks

    return copy_once


def run_host_benchmark(
    copy_fn,
    warmup_iters=3,
    min_measure_iters=5,
    max_measure_iters=100,
    max_measure_ms=1000,
):
    """Measures average host copy latency with a time budget and minimum samples."""
    for _ in range(warmup_iters):
        copy_fn()

    start_ns = time.perf_counter_ns()
    measure_iters = 0
    while measure_iters < max_measure_iters:
        copy_fn()
        measure_iters += 1
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
        if measure_iters >= min_measure_iters and elapsed_ms >= max_measure_ms:
            break

    end_ns = time.perf_counter_ns()
    total_elapsed_ms = (end_ns - start_ns) / 1e6
    if total_elapsed_ms > max_measure_ms:
        print(
            f"WARNING: benchmark elapsed {total_elapsed_ms:.2f} ms exceeds "
            f"max_measure_ms={max_measure_ms}, measure_iters={measure_iters}"
        )
    return total_elapsed_ms / measure_iters


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["ChunkSize(MB)"],
        x_vals=[1, 8, 128, 1024, 4096],
        line_arg="provider",
        line_vals=[
            "cp_pag_to_pag",
            "cp_pag_to_pin",
            "cp_pin_to_pin",
            "cp_pin_to_pag",
        ],
        line_names=[
            "CpPagToPag(GB/s)",
            "CpPagToPin(GB/s)",
            "CpPinToPin(GB/s)",
            "CpPinToPag(GB/s)",
        ],
        ylabel="GB/s",
        plot_name="h2h",
        args={},
    )
)
def benchmark_h2h(*args, **kwargs):
    chunk_size = kwargs["ChunkSize(MB)"]
    provider = kwargs["provider"]

    print_system_memory_info()
    print_torch_thread_info()
    print_affinity_info()

    chunk_bytes = int(chunk_size * 1024 * 1024)
    _, llc_size_bytes = detect_cache_info()
    min_chunks = 4 if chunk_bytes < 1024 * 1024 * 1024 else 2
    target_working_set_bytes = max(
        llc_size_bytes * 4, chunk_bytes * min_chunks, 1024 * 1024 * 1024
    )
    target_working_set_bytes = min(target_working_set_bytes, 8 * 1024 * 1024 * 1024)
    num_chunks = max(min_chunks, math.ceil(target_working_set_bytes / chunk_bytes))

    if provider == "cp_pag_to_pag":
        src = allocate_host_buffer(num_chunks, chunk_bytes, pinned=False)
        dst = allocate_host_buffer(num_chunks, chunk_bytes, pinned=False)
    elif provider == "cp_pag_to_pin":
        src = allocate_host_buffer(num_chunks, chunk_bytes, pinned=False)
        dst = allocate_host_buffer(num_chunks, chunk_bytes, pinned=True)
    elif provider == "cp_pin_to_pin":
        src = allocate_host_buffer(num_chunks, chunk_bytes, pinned=True)
        dst = allocate_host_buffer(num_chunks, chunk_bytes, pinned=True)
    elif provider == "cp_pin_to_pag":
        src = allocate_host_buffer(num_chunks, chunk_bytes, pinned=True)
        dst = allocate_host_buffer(num_chunks, chunk_bytes, pinned=False)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    touch_buffer_pages(src)
    touch_buffer_pages(dst)

    ms = run_host_benchmark(
        build_rotating_copy_fn(src, dst, chunk_bytes, llc_size_bytes, seed=0)
    )

    gbps = lambda ms: chunk_bytes * 1e-9 / (ms * 1e-3)

    return gbps(ms)


if __name__ == "__main__":
    benchmark_h2h.run(print_data=True, save_path=".")
