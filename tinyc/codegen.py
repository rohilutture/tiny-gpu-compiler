"""CUDA C code generation for fused kernels.

Kernel signature (all kinds):  k(const float* in0, ..., float* out, int M, int N, int K)
`in<i>` follow Kernel.reads order. Tunable config values are baked into the source text.

  ewise : grid-stride loop, EPT elements per thread, BLOCK threads
  reduce: one block per row, BLOCK threads, warp-shuffle + shared-memory reduction
  matmul: TxT shared-memory tiled GEMM (one output element per thread) + fused epilogue

Search spaces for the autotuner live in CONFIG_SPACE.
"""
from __future__ import annotations

import hashlib
import itertools

from .fuse import Kernel
from .ir import Node

CONFIG_SPACE = {
    "ewise": {"block": [128, 256, 512], "ept": [1, 2, 4]},
    "reduce": {"block": [64, 128, 256, 512, 1024]},
    "matmul": {"tile": [8, 16, 32]},
}

_UN = {
    "neg": "-({a})", "exp": "expf({a})", "relu": "fmaxf({a}, 0.0f)", "tanh": "tanhf({a})",
    "sqrt": "sqrtf({a})", "rsqrt": "rsqrtf({a})", "recip": "(1.0f / ({a}))",
}
_BIN = {"add": "({a} + {b})", "sub": "({a} - {b})", "mul": "({a} * {b})", "div": "({a} / {b})", "max": "fmaxf({a}, {b})"}


PREAMBLE = "#define TINYC_INF __int_as_float(0x7f800000)\n"


def default_config(kind: str) -> dict:
    return {k: v[0] if k != "block" else v[min(1, len(v) - 1)] for k, v in CONFIG_SPACE[kind].items()}


def configs(kind: str):
    space = CONFIG_SPACE[kind]
    keys = list(space)
    for vals in itertools.product(*(space[k] for k in keys)):
        yield dict(zip(keys, vals))


def _lit(v: float) -> str:
    return f"{float(v)!r}f" if abs(v) != float("inf") else ("TINYC_INF" if v > 0 else "(-TINYC_INF)")


def _load(k: Kernel, node: Node, i: str, j: str) -> str:
    """Expression loading `node` (a memory-resident read) at logical element (i, j), honoring broadcast."""
    idx = [r.id for r in k.reads].index(node.id)
    M, N = node.shape
    if (M, N) == (1, 1):
        off = "0"
    elif M == 1:
        off = j
    elif N == 1:
        off = i
    else:
        off = f"(size_t)({i}) * N + ({j})"
    return f"in{idx}[{off}]"


def _emit_nodes(k: Kernel, nodes: list[Node], i: str, j: str, env: dict[int, str]):
    """Emit straight-line code for `nodes`. Returns (lines, ref) where ref(node) gives its C expression.
    env maps node id -> C expression (variable or load) and is updated in place."""
    lines = []

    def ref(n: Node) -> str:
        if n.id in env:
            return env[n.id]
        if n.op == "const":
            return _lit(n.attrs["value"])
        return _load(k, n, i, j)  # memory-resident input

    for n in nodes:
        if n.op == "unary":
            expr = _UN[n.attrs["fn"]].format(a=ref(n.inputs[0]))
        elif n.op == "binary":
            expr = _BIN[n.attrs["fn"]].format(a=ref(n.inputs[0]), b=ref(n.inputs[1]))
        else:
            raise ValueError(f"cannot emit {n.op} inline")
        lines.append(f"float v{n.id} = {expr};")
        env[n.id] = f"v{n.id}"
    return lines, ref


def _sig(k: Kernel) -> str:
    ins = ", ".join(f"const float* __restrict__ in{i}" for i in range(len(k.reads)))
    return f'extern "C" __global__ void k({ins}{", " if ins else ""}float* __restrict__ out, int M, int N, int K)'


def gen_ewise(k: Kernel, cfg: dict) -> str:
    env: dict[int, str] = {}
    lines, ref = _emit_nodes(k, k.body, "i", "j", env)
    body = "\n      ".join(lines)
    ept = cfg["ept"]
    return f"""{_sig(k)} {{
  const size_t total = (size_t)M * N;
  const size_t stride = (size_t)gridDim.x * blockDim.x * {ept};
  for (size_t base = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * {ept}; base < total; base += stride) {{
    #pragma unroll
    for (int u = 0; u < {ept}; ++u) {{
      size_t e = base + u;
      if (e >= total) break;
      int i = (int)(e / N), j = (int)(e % N);
      {body}
      out[e] = {ref(k.out)};
    }}
  }}
}}
"""


def gen_reduce(k: Kernel, cfg: dict) -> str:
    fn = k.core.attrs["fn"]
    init, comb = ("0.0f", "{a} + {b}") if fn == "sum" else ("(-TINYC_INF)", "fmaxf({a}, {b})")
    env: dict[int, str] = {}
    lines, ref = _emit_nodes(k, k.body, "i", "j", env)
    body = "\n      ".join(lines)
    src = ref(k.core.inputs[0])
    block = cfg["block"]
    return f"""{_sig(k)} {{
  const int i = blockIdx.x;
  float acc = {init};
  for (int j = threadIdx.x; j < N; j += {block}) {{
      {body}
      float x = {src};
      acc = {comb.format(a="acc", b="x")};
  }}
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) {{
    float other = __shfl_xor_sync(0xffffffffu, acc, o);
    acc = {comb.format(a="acc", b="other")};
  }}
  __shared__ float part[{block // 32}];
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) part[w] = acc;
  __syncthreads();
  if (w == 0) {{
    acc = l < {block // 32} ? part[l] : {init};
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {{
      float other = __shfl_xor_sync(0xffffffffu, acc, o);
      acc = {comb.format(a="acc", b="other")};
    }}
    if (l == 0) out[i] = acc;
  }}
}}
"""


def gen_matmul(k: Kernel, cfg: dict) -> str:
    T = cfg["tile"]
    core = k.core
    env = {core.id: "acc"}
    lines, ref = _emit_nodes(k, k.body, "row", "col", env)
    epi = "\n      ".join(lines)
    return f"""{_sig(k)} {{
  __shared__ float As[{T}][{T}];
  __shared__ float Bs[{T}][{T}];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * {T} + ty, col = blockIdx.x * {T} + tx;
  float acc = 0.0f;
  for (int t = 0; t < K; t += {T}) {{
    As[ty][tx] = (row < M && t + tx < K) ? in0[(size_t)row * K + t + tx] : 0.0f;
    Bs[ty][tx] = (t + ty < K && col < N) ? in1[(size_t)(t + ty) * N + col] : 0.0f;
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < {T}; ++kk) acc += As[ty][kk] * Bs[kk][tx];
    __syncthreads();
  }}
  if (row < M && col < N) {{
      {epi}
      out[(size_t)row * N + col] = {ref(k.out)};
  }}
}}
"""


def generate(k: Kernel, cfg: dict | None = None) -> str:
    cfg = cfg or default_config(k.kind)
    # NVRTC ships without <math.h>/<cmath>, so INFINITY is undefined; build it from its bit pattern.
    return PREAMBLE + {"ewise": gen_ewise, "reduce": gen_reduce, "matmul": gen_matmul}[k.kind](k, cfg)


def problem_dims(k: Kernel) -> tuple[int, int, int]:
    """(M, N, K) kernel arguments. A row reduction's output is (M, 1) but it must loop over the INPUT's N columns;
    passing the output's N (= 1) would silently reduce only the first column."""
    if k.kind == "reduce":
        M, N = k.core.inputs[0].shape
        return M, N, N
    M, N = k.out.shape
    return M, N, (k.core.inputs[0].shape[1] if k.kind == "matmul" else N)


def launch_dims(k: Kernel, cfg: dict, M: int, N: int, sm_count: int = 170):
    """(grid, block) for cuLaunchKernel. sm_count only caps the ewise grid (grid-stride loop)."""
    if k.kind == "ewise":
        blocks = min((M * N + cfg["block"] * cfg["ept"] - 1) // (cfg["block"] * cfg["ept"]), sm_count * 32)
        return (max(blocks, 1), 1, 1), (cfg["block"], 1, 1)
    if k.kind == "reduce":
        return (M, 1, 1), (cfg["block"], 1, 1)
    T = cfg["tile"]
    return ((N + T - 1) // T, (M + T - 1) // T, 1), (T, T, 1)


def source_hash(src: str) -> str:
    return hashlib.sha1(src.encode()).hexdigest()[:12]
