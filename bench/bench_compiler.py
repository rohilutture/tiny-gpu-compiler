"""Compiled (fused + autotuned) vs PyTorch eager vs the SAME graph compiled with fusion disabled.

    python -m bench.bench_compiler --csv results/compiler.csv

The 'unfused' arm runs one tinyc kernel per op, so the fused-vs-unfused gap isolates the fusion
benefit from codegen quality. The matmul here is a simple shared-memory-tiled kernel, so it will
be far slower than cuBLAS; the claim under test is fusion of memory-bound work, NOT matmul peak.
"""
from __future__ import annotations

import argparse
import csv

import torch

import tinyc as tc
from tinyc import fuse
from tinyc.api import CompiledGraph


def time_ms(fn, warm=10, reps=50):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    ts = []
    for _ in range(reps):
        flush.zero_()  # evict L2 so memory-bound kernels see DRAM
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


def torch_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


CASES = {
    "gelu_bias": (lambda x, b: tc.gelu(x + b), lambda x, b: torch_gelu(x + b),
                  {"x": (8192, 4096), "b": (1, 4096)}),
    "softmax": (tc.softmax, lambda x: torch.softmax(x, -1), {"x": (8192, 4096)}),
    "matmul_bias_gelu": (lambda x, w, b: tc.gelu(x @ w + b), lambda x, w, b: torch_gelu(x @ w + b),
                         {"x": (2048, 1024), "w": (1024, 2048), "b": (1, 2048)}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/compiler.csv")
    ap.add_argument("--no-autotune", action="store_true")
    args = ap.parse_args()
    rows = []
    for name, (fn, tfn, shapes) in CASES.items():
        feeds = {k: torch.randn(s, device="cuda") * 0.5 for k, s in shapes.items()}
        fused = tc.compile(fn, shapes)
        unfused = CompiledGraph(fused.g, fuse.unfused_kernels(fused.g))
        if not args.no_autotune:
            fused.autotune(feeds, verbose=False)
            unfused.autotune(feeds, verbose=False)
        ref = tfn(*feeds.values())
        torch.testing.assert_close(fused(**feeds), ref, rtol=1e-3, atol=1e-3)  # never time a wrong result
        torch.testing.assert_close(unfused(**feeds), ref, rtol=1e-3, atol=1e-3)
        for impl, call, ks in (("tinyc_fused", lambda: fused(**feeds), fused.kernels),
                               ("tinyc_unfused", lambda: unfused(**feeds), unfused.kernels),
                               ("torch_eager", lambda: tfn(*feeds.values()), None)):
            ms = time_ms(call)
            mb = fuse.modeled_bytes(ks) / 1e6 if ks else float("nan")
            rows.append([name, impl, f"{ms:.4f}", len(ks) if ks else "", f"{mb:.1f}"])
            print(rows[-1])
        print(fused.explain())
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "impl", "ms", "kernels", "modeled_MB"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
