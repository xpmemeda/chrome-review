from __future__ import annotations

import collections
import onnx
from onnx import TensorProto
from typing import Dict, Any


def _dtype_name(t: int) -> str:
    return TensorProto.DataType.Name(t) if hasattr(TensorProto.DataType, "Name") else str(t)


def inspect_onnx_fp_precision(onnx_path: str) -> Dict[str, Any]:
    """
    Inspect ONNX model precision:
      - initializer (weights/constants) dtypes (most important)
      - graph inputs/outputs/value_info dtypes (may be missing for intermediates)

    Returns a dict with:
      - summary: "fp16" / "fp32" / "mixed" / "unknown"
      - counts: per-section dtype histogram
      - notes: hints about interpretation
    """
    model = onnx.load(onnx_path)

    def collect_from_value_infos(vinfos) -> collections.Counter:
        c = collections.Counter()
        for vi in vinfos:
            t = vi.type
            if not t.HasField("tensor_type"):
                continue
            tt = t.tensor_type
            if not tt.HasField("elem_type"):
                continue
            c[int(tt.elem_type)] += 1
        return c

    # 1) Weights / constants (initializers)
    init_counter = collections.Counter()
    init_bytes = collections.Counter()
    for w in model.graph.initializer:
        dt = int(w.data_type)
        init_counter[dt] += 1
        # Rough size estimate (may be 0 if stored in external data or not raw_data)
        # If external data is used, this will undercount; still OK for dtype detection.
        if w.raw_data:
            init_bytes[dt] += len(w.raw_data)

    # 2) Inputs/Outputs/ValueInfos (types can be absent for intermediates)
    input_counter = collect_from_value_infos(model.graph.input)
    output_counter = collect_from_value_infos(model.graph.output)
    value_info_counter = collect_from_value_infos(model.graph.value_info)

    def has_fp16(counter: collections.Counter) -> bool:
        return counter.get(TensorProto.FLOAT16, 0) > 0

    def has_fp32(counter: collections.Counter) -> bool:
        return counter.get(TensorProto.FLOAT, 0) > 0

    # Determine overall summary primarily from initializers (best signal)
    if init_counter:
        init_has16 = has_fp16(init_counter)
        init_has32 = has_fp32(init_counter)
        if init_has16 and not init_has32:
            summary = "fp16"
        elif init_has32 and not init_has16:
            summary = "fp32"
        elif init_has16 and init_has32:
            summary = "mixed"
        else:
            # Could be BF16/INT8/etc.
            summary = "unknown"
    else:
        # Fallback: infer from IO/value_info if no initializers (rare)
        combined = input_counter + output_counter + value_info_counter
        if has_fp16(combined) and not has_fp32(combined):
            summary = "fp16"
        elif has_fp32(combined) and not has_fp16(combined):
            summary = "fp32"
        elif has_fp16(combined) and has_fp32(combined):
            summary = "mixed"
        else:
            summary = "unknown"

    def counter_to_readable(counter: collections.Counter) -> Dict[str, int]:
        return { _dtype_name(k): int(v) for k, v in counter.items() }

    return {
        "onnx_path": onnx_path,
        "ir_version": int(model.ir_version),
        "opset_import": [{ "domain": o.domain, "version": int(o.version) } for o in model.opset_import],
        "summary": summary,
        "counts": {
            "initializers": counter_to_readable(init_counter),
            "inputs": counter_to_readable(input_counter),
            "outputs": counter_to_readable(output_counter),
            "value_info": counter_to_readable(value_info_counter),
        },
        "initializer_raw_bytes": { _dtype_name(k): int(v) for k, v in init_bytes.items() },
        "notes": [
            "summary is decided primarily by initializer (weights) dtypes.",
            "value_info types may be missing; that's normal.",
            "initializer_raw_bytes may be 0 if tensors use external data; dtype counts are still valid.",
        ],
    }


# --- Example usage ---
if __name__ == "__main__":
    import json, sys
    info = inspect_onnx_fp_precision(sys.argv[1])
    print(json.dumps(info, indent=2))