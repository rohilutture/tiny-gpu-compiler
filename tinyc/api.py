"""Top-level API: trace -> fuse -> codegen -> (autotune) -> run.

    import tinyc as tc
    cg = tc.compile(lambda x, b: tc.gelu(x + b), {"x": (4096, 4096), "b": (1, 4096)})
    print(cg.explain())
    y = cg(x=x_cuda, b=b_cuda)          # torch CUDA fp32 tensors
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import codegen, fuse
from .ir import Graph, trace

CACHE_PATH = Path(".tinyc_cache.json")


class CompiledGraph:
    def __init__(self, g: Graph, kernels: list[fuse.Kernel], arch: str | None = None):
        self.g, self.kernels, self.arch = g, kernels, arch
        self._loaded: dict[int, object] = {}
        self.sources: dict[int, str] = {}

    # ---------------------------------------------------------------- introspection (no GPU)
    def explain(self) -> str:
        return fuse.explain(self.g, self.kernels)

    def source(self, i: int) -> str:
        return codegen.generate(self.kernels[i], self.kernels[i].config or None)

    # ---------------------------------------------------------------- GPU
    def _load(self, i: int):
        if i not in self._loaded:
            from . import runtime

            arch = self.arch or runtime.device_arch()
            src = self.source(i)
            self.sources[i] = src
            self._loaded[i] = runtime.LoadedKernel(runtime.compile_cubin(src, arch))
        return self._loaded[i]

    def __call__(self, **feeds):
        import torch

        from .runtime import LoadedKernel  # noqa: F401  (import error surfaces early)

        buf = {n.id: feeds[name].contiguous() for name, n in self.g.inputs.items()}
        stream = torch.cuda.current_stream().cuda_stream
        for i, k in enumerate(self.kernels):
            out = torch.empty(k.out.shape, device="cuda", dtype=torch.float32)
            buf[k.out.id] = out
            self._launch(i, k, [buf[r.id] for r in k.reads], out, stream)
        res = [buf[o.id] for o in self.g.outputs]
        return res[0] if len(res) == 1 else tuple(res)

    def _launch(self, i, k, ins, out, stream):
        cfg = k.config or codegen.default_config(k.kind)
        M, N, K = codegen.problem_dims(k)
        grid, block = codegen.launch_dims(k, cfg, M, N)
        self._load(i).launch(grid, block, [t.data_ptr() for t in ins] + [out.data_ptr()], [M, N, K], stream)

    # ---------------------------------------------------------------- autotune
    def autotune(self, feeds, warmup: int = 3, reps: int = 20, use_cache: bool = True, verbose: bool = True):
        """Pick the fastest config per kernel by timing every variant on real inputs."""
        import torch

        from . import runtime

        cache = json.loads(CACHE_PATH.read_text()) if use_cache and CACHE_PATH.exists() else {}
        dev = torch.cuda.get_device_name()
        buf = {n.id: feeds[name].contiguous() for name, n in self.g.inputs.items()}
        stream = torch.cuda.current_stream().cuda_stream
        arch = self.arch or runtime.device_arch()
        for i, k in enumerate(self.kernels):
            M, N, K = codegen.problem_dims(k)
            key = f"{dev}|{k.kind}|{M}x{N}|" \
                  f"{[(r.shape) for r in k.reads]}|{[(n.op, n.attrs.get('fn')) for n in k.body]}|{k.core.op if k.core else ''}"
            ins = [buf[r.id] for r in k.reads]
            out = torch.empty(k.out.shape, device="cuda", dtype=torch.float32)
            if key in cache:
                k.config = cache[key]["config"]
            else:
                best = None
                for cfg in codegen.configs(k.kind):
                    try:
                        lk = runtime.LoadedKernel(runtime.compile_cubin(codegen.generate(k, cfg), arch))
                    except Exception:
                        continue
                    grid, block = codegen.launch_dims(k, cfg, M, N)
                    args = ([t.data_ptr() for t in ins] + [out.data_ptr()], [M, N, K])
                    for _ in range(warmup):
                        lk.launch(grid, block, *args, stream)
                    ts = []
                    for _ in range(reps):
                        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        a.record(); lk.launch(grid, block, *args, stream); b.record(); b.synchronize()
                        ts.append(a.elapsed_time(b))
                    ms = sorted(ts)[len(ts) // 2]
                    if verbose:
                        print(f"  kernel {i} {k.kind} {cfg}: {ms:.4f} ms")
                    if best is None or ms < best[0]:
                        best = (ms, cfg)
                k.config = best[1]
                cache[key] = {"config": best[1], "ms": best[0], "time": time.time()}
            self._loaded.pop(i, None)
            # Later kernels read this kernel's output, so produce it for real (with the chosen config).
            self._launch(i, k, ins, out, stream)
            buf[k.out.id] = out
        if use_cache:
            CACHE_PATH.write_text(json.dumps(cache, indent=1))
        return self


def compile(fn, shapes: dict, remat_limit: int = 8, arch: str | None = None) -> CompiledGraph:
    g = trace(fn, shapes)
    return CompiledGraph(g, fuse.plan(g, remat_limit), arch)

