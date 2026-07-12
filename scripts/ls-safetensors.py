import os
import argparse
import math
from collections import defaultdict

from safetensors.torch import safe_open


QUANT_META_MARKERS = (
    ".input_scale",
    ".weight_quants.",
    ".scale_zero",
    ".scale",
)

DTYPE_BYTES = {
    "torch.bool": 1,
    "torch.int8": 1,
    "torch.uint8": 1,
    "torch.int16": 2,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.int32": 4,
    "torch.float32": 4,
    "torch.int64": 8,
    "torch.float64": 8,
}


def print_table(headers, rows):
    table = [headers] + rows
    widths = [max(len(str(row[i])) for row in table) for i in range(len(headers))]
    separator = "-+-".join("-" * width for width in widths)

    print(" | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers))))
    print(separator)
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def shape_text(shape):
    return "x".join(str(dim) for dim in shape)


def element_count(shape):
    return math.prod(shape)


def is_quant_meta(key):
    return any(marker in key for marker in QUANT_META_MARKERS)


def is_parameter_tensor(key):
    if is_quant_meta(key):
        return False
    return key.endswith(".weight") or key.endswith(".preprocessed_weight")


def estimate_bytes(tensors):
    total = 0
    unknown_dtypes = set()
    for tensor in tensors:
        dtype_bytes = DTYPE_BYTES.get(tensor["dtype"])
        if dtype_bytes is None:
            unknown_dtypes.add(tensor["dtype"])
            continue
        total += tensor["numel"] * dtype_bytes
    return total, unknown_dtypes


def format_billions(numel):
    return f"{numel / 1_000_000_000:.3f}B"


def format_size(byte_count):
    gib = byte_count / 1024**3
    gb = byte_count / 1_000_000_000
    return f"{gib:.3f} GiB / {gb:.3f} GB"


def print_numel_row(name, numel):
    print(f"{name:<32} {numel:>18,} {format_billions(numel):>12}")


def print_size_row(name, byte_count):
    print(f"{name:<32} {byte_count:>18,} {format_size(byte_count):>24}")


def print_summary(tensors, include_quant_meta):
    all_numel = sum(tensor["numel"] for tensor in tensors)
    parameter_tensors = [
        tensor
        for tensor in tensors
        if include_quant_meta or is_parameter_tensor(tensor["key"])
    ]
    parameter_numel = sum(tensor["numel"] for tensor in parameter_tensors)
    parameter_bytes, parameter_unknown_dtypes = estimate_bytes(parameter_tensors)
    all_bytes, all_unknown_dtypes = estimate_bytes(tensors)

    dtype_numel = defaultdict(int)
    for tensor in parameter_tensors:
        dtype_numel[tensor["dtype"]] += tensor["numel"]

    print()
    print_numel_row("estimated parameters", parameter_numel)
    print_numel_row("all tensor elements", all_numel)
    print()
    print("by dtype")
    for dtype, numel in sorted(dtype_numel.items()):
        print_numel_row(dtype, numel)

    print()
    print("estimated disk size")
    print_size_row("estimated parameters", parameter_bytes)
    print_size_row("all tensor elements", all_bytes)

    unknown_dtypes = parameter_unknown_dtypes | all_unknown_dtypes
    if unknown_dtypes:
        print()
        print("unknown dtype byte sizes: " + ", ".join(sorted(unknown_dtypes)))


def main(cmd_arguments):
    model_dir = cmd_arguments.d

    tensors = []
    for file_name in sorted(os.listdir(model_dir)):
        if not file_name.endswith(".safetensors"):
            continue

        path = os.path.join(model_dir, file_name)
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                shape = tuple(tensor.shape)
                tensors.append(
                    {
                        "file": file_name,
                        "key": key,
                        "shape": shape,
                        "dtype": str(tensor.dtype),
                        "numel": element_count(shape),
                    }
                )

    rows = [
        (tensor["file"], tensor["key"], shape_text(tensor["shape"]), tensor["dtype"])
        for tensor in sorted(tensors, key=lambda item: (item["file"], item["key"]))
    ]
    print_table(("file", "key", "shape", "dtype"), rows)
    print_summary(tensors, cmd_arguments.include_quant_meta)


if __name__ == "__main__":
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument("--d", type=str, help="directory", required=True)
    cmd_parser.add_argument(
        "--include-quant-meta",
        action="store_true",
        help="include quantization metadata tensors in the main estimate",
    )
    cmd_arguments = cmd_parser.parse_args()
    main(cmd_arguments)
