"""Every generated kernel, in every autotune config, must compile with NVRTC for sm_120.
NVRTC is a pure compiler: no GPU needed."""
import pytest

import tinyc as tc
from tinyc import codegen
from tinyc.fuse import plan
from tinyc.ir import trace

runtime = pytest.importorskip("tinyc.runtime")

M, N, K = 64, 96, 48
GRAPHS = {
    "gelu_bias": (lambda x, b: tc.gelu(x + b), {"x": (M, N), "b": (1, N)}),
    "softmax": (tc.softmax, {"x": (M, N)}),
    "matmul_bias_gelu": (lambda x, w, b: tc.gelu(x @ w + b), {"x": (M, K), "w": (K, N), "b": (1, N)}),
    "mlp": (lambda x, w1, w2: (x @ w1).relu() @ w2, {"x": (M, K), "w1": (K, N), "w2": (N, K)}),
    "rownorm": (lambda x: x * (x * x).sum().recip().sqrt(), {"x": (M, N)}),
    "scalar_ops": (lambda x: (2.0 - x) / 3.0 + (-x), {"x": (M, N)}),
}


@pytest.mark.parametrize("name", GRAPHS)
def test_all_configs_compile(name):
    fn, shapes = GRAPHS[name]
    ks = plan(trace(fn, shapes))
    for k in ks:
        for cfg in codegen.configs(k.kind):
            src = codegen.generate(k, cfg)
            assert runtime.compile_cubin(src, "sm_120"), (k, cfg)


def test_compile_error_reports_nvrtc_log():
    with pytest.raises(runtime.CompileError) as e:
        runtime.compile_cubin('extern "C" __global__ void k() { undefined_symbol(); }')
    assert "undefined_symbol" in str(e.value)


def test_source_changes_with_config():
    k = plan(trace(tc.softmax, {"x": (M, N)}))[0]
    srcs = {codegen.source_hash(codegen.generate(k, c)) for c in codegen.configs("reduce")}
    assert len(srcs) == len(list(codegen.configs("reduce")))
