# tinyc: a small GPU tensor compiler

Status: **working end to end on the RTX 5090: 27 tests pass, including GPU runs of 6 graphs (and every autotune config
of 3 of them) against the numpy interpreter at rtol/atol 2e-4. First benchmark below; single runs, not yet profiled.**

The first GPU launch found two real bugs that CPU tests could not see, both fixed (with a regression test):
reduction kernels were launched with the (M, 1) output shape as (M, N), so softmax, `matmul_reduce` and `row_scale`
silently reduced only column 0; and `autotune()` crashed on any multi-kernel graph because later kernels had no
buffer for earlier kernels' outputs.

## First benchmark (fp32, one RTX 5090, median ms; `results/compiler.csv`, `results/compiler_run.txt`)

| Graph | tinyc fused | tinyc unfused | PyTorch eager | Note |
|---|---|---|---|---|
| gelu(x + b), 8192x4096 | 0.19 ms (1 kernel) | 2.11 ms (10 kernels) | 2.08 ms | fused is ~11x less modeled traffic and ~11x faster than unfused/eager |
| softmax, 8192x4096 | 0.34 ms (3 kernels) | 0.70 ms (5) | **0.18 ms** | PyTorch's single fused kernel is ~1.9x faster than mine |
| gelu(x@w + b), 2048x2048 | 0.86 ms (1 kernel) | 0.96 ms (11) | **0.25 ms** | my SIMT tiled matmul is ~3.4x slower than cuBLAS |

Caveats: gelu_bias fused measured 0.36 ms in an earlier run before re-tuning, so expect run-to-run swings of that size
(the GPU also drives the desktop); the "fusion wins" result holds only for elementwise chains, and the compiler loses to
the library on softmax and matmul. Do not claim speedups over PyTorch except for the elementwise case.

A Python tracing DSL lowers to a graph IR, a fusion pass partitions it into kernels, each kernel
is emitted as CUDA C, compiled at runtime with NVRTC, and tuned by an autotuner.

```python
import tinyc as tc
cg = tc.compile(lambda x, w, b: tc.gelu(x @ w + b), {"x": (4096, 1024), "w": (1024, 4096), "b": (1, 4096)})
print(cg.explain())      # kernels, what each fuses, modeled memory traffic (no GPU needed)
print(cg.source(0))      # generated CUDA
cg.autotune(feeds)       # time every config per kernel on real inputs
y = cg(x=x, w=w, b=b)    # torch CUDA fp32 tensors
```

## What the compiler does

| Stage | File | Idea |
|---|---|---|
| Frontend | `tinyc/ir.py` | operator-overloading tracer; 2-D fp32; ops: unary, binary, rowreduce (sum/max), matmul; broadcast shapes (M,N) (1,N) (M,1) (1,1) |
| Reference | `tinyc/interp.py` | numpy interpreter, the ground truth for tests |
| Fusion | `tinyc/fuse.py` | partitions into kernels; fixed-point loop over a materialized set |
| Codegen | `tinyc/codegen.py` | ewise (grid-stride, N elems/thread), reduce (block per row, shuffle + smem), matmul (smem-tiled GEMM + fused epilogue) |
| Runtime | `tinyc/runtime.py` | NVRTC to cubin, driver-API load and launch via cuda-python |
| Autotune | `tinyc/api.py` | times every config per kernel on real inputs, JSON cache |

Fusion rules (see the `fuse.py` docstring): shared cheap values are **recomputed** in each consumer
instead of materialized (`remat_limit`), matmul work is never duplicated, a matmul absorbs following
elementwise ops as an epilogue, and a kernel holds at most one matmul. For softmax this gives 3 kernels
instead of 5 by recomputing `exp(x - max)` rather than writing it to memory.

Tunables: ewise `block` x `ept`; reduce `block`; matmul `tile`.

## Run

```powershell
$env:PYTHONPATH = "."
.venv\Scripts\python.exe -m pytest -q -m "not gpu"      # CPU: planner + NVRTC compile of all configs
.venv\Scripts\python.exe examples\show_plan.py          # print plans and generated CUDA

# when the GPU is free:
.venv\Scripts\python.exe -m pytest -q                   # adds device correctness (every config vs the interpreter)
.venv\Scripts\python.exe -m bench.bench_compiler --csv results/compiler.csv
```

## Honest limits (by design)

- The matmul is a simple shared-memory-tiled kernel, one output element per thread, fp32, no tensor
  cores. It will be far slower than cuBLAS. The claim this project can make is about **fusing
  memory-bound work** (elementwise, reductions, epilogues), not matmul performance.
- Modeled traffic in `explain()` is an ideal-DRAM estimate (each kernel reads inputs once, writes output
  once), not a measurement. Report measured times from `bench_compiler.py` instead, and always
  alongside the unfused arm (same codegen, no fusion) so the fusion effect is separated from codegen quality.
- Not implemented: online-softmax rewrite (softmax stays 3 kernels), common-subexpression elimination,
  vectorized loads, tensor-core matmul, dtypes other than fp32, shapes beyond 2-D.

## Next steps

1. Run on the GPU; fix whatever the first launch breaks (launch glue is untested).
2. Add CSE and an online-softmax pattern rewrite (3 kernels to 1), and measure each.
3. Vectorized `float4` loads in ewise; register-blocked matmul; then a tensor-core (WMMA) matmul path.
4. Emit Triton or PTX as a second backend and compare codegen quality.

## Phrasing rules (once there are results)

- One RTX 5090; fp32; NVRTC 12.9. Say "kernel fusion and autotuning for memory-bound ops", never "faster
  than PyTorch" without the measured table, and never compare the tiled matmul to cuBLAS as a win.
- The IR is a deliberately small subset (2-D, row reductions); do not describe it as a general ML compiler.
