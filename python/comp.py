#!/usr/bin/env python3
import numpy as np
import sys

def is_numeric_array(x):
    return isinstance(x, np.ndarray) and x.dtype != object

def compare_arrays(a, b, eps=1e-12):
    diff = a - b
    abs_err = np.abs(diff)

    max_abs = abs_err.max()
    mean_abs = abs_err.mean()

    # relative error: |a-b| / max(|a|, |b|, eps)
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), eps)
    rel_err = abs_err / denom
    max_rel = rel_err.max()

    return max_abs, mean_abs, max_rel

def main(f1, f2):
    npz1 = np.load(f1, allow_pickle=True)
    npz2 = np.load(f2, allow_pickle=True)

    keys1 = set(npz1.files)
    keys2 = set(npz2.files)

    common = sorted(keys1 & keys2)
    only1 = sorted(keys1 - keys2)
    only2 = sorted(keys2 - keys1)

    if only1:
        print(f"[Only in {f1}]: {only1}")
    if only2:
        print(f"[Only in {f2}]: {only2}")

    print("\n=== Comparing common items ===")
    for k in common:
        a = npz1[k]
        b = npz2[k]

        # unwrap object array (你之前踩过的坑)
        if isinstance(a, np.ndarray) and a.dtype == object:
            a = a.item()
        if isinstance(b, np.ndarray) and b.dtype == object:
            b = b.item()

        print(f"\n[{k}]")

        if not (is_numeric_array(a) and is_numeric_array(b)):
            print("  skipped (non-numeric or object)")
            continue

        if a.shape != b.shape:
            print(f"  shape mismatch: {a.shape} vs {b.shape}")
            continue

        if a.dtype != b.dtype:
            print(f"  dtype mismatch: {a.dtype} vs {b.dtype}")

        max_abs, mean_abs, max_rel = compare_arrays(a.astype(np.float64),
                                                     b.astype(np.float64))

        print(f"  max_abs_err : {max_abs:.6e}")
        print(f"  mean_abs_err: {mean_abs:.6e}")
        print(f"  max_rel_err : {max_rel:.6e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: compare_npz.py a.npz b.npz")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
