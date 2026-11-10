import numpy as np
import pytest

import tinyc as tc
from tinyc.fuse import modeled_bytes, plan, unfused_kernels
from tinyc.ir import broadcast_shape, trace

M, N, K = 64, 96, 48


def kinds(ks):
    return [k.kind for k in ks]


def test_broadcast_rules():
    assert broadcast_shape((4, 8), (1, 8)) == (4, 8)
    assert broadcast_shape((4, 1), (1, 8)) == (4, 8)
    with pytest.raises(ValueError):
        broadcast_shape((4, 8), (4, 7))


def test_matmul_shape_check():
    with pytest.raises(ValueError):
        trace(lambda a, b: a @ b, {"a": (4, 5), "b": (6, 7)})


def test_interpreter_matches_numpy_softmax():
    g = trace(tc.softmax, {"x": (M, N)})
    x = np.random.randn(M, N).astype(np.float32) * 3
    (y,) = tc.interpret(g, {"x": x})
    e = np.exp(x - x.max(1, keepdims=True))
    np.testing.assert_allclose(y, e / e.sum(1, keepdims=True), rtol=1e-5, atol=1e-6)


def test_elementwise_chain_fuses_to_one_kernel():
    g = trace(lambda x, b: tc.gelu(x + b), {"x": (M, N), "b": (1, N)})
    ks = plan(g)
    assert kinds(ks) == ["ewise"]
    assert len(unfused_kernels(g)) > 8  # gelu alone is ~10 primitive ops
    assert modeled_bytes(ks) < modeled_bytes(unfused_kernels(g)) / 5


def test_softmax_recomputes_exp_instead_of_materializing():
    g = trace(tc.softmax, {"x": (M, N)})
    ks = plan(g, remat_limit=8)
    assert kinds(ks) == ["reduce", "reduce", "ewise"]  # rowmax, rowsum(exp(x-m)), exp(x-m)/s
    ks0 = plan(g, remat_limit=0)
    # remat_limit=0 forbids recomputation: the shared x-m and exp(x-m) are both materialized,
    # which degenerates to the unfused plan (5 kernels)
    assert len(ks0) == 5
    assert len(ks) < len(ks0) <= len(unfused_kernels(g))
    assert modeled_bytes(ks) < modeled_bytes(ks0)


def test_matmul_bias_gelu_is_one_kernel_with_epilogue():
    g = trace(lambda x, w, b: tc.gelu(x @ w + b), {"x": (M, K), "w": (K, N), "b": (1, N)})
    ks = plan(g)
    assert kinds(ks) == ["matmul"]
    k = ks[0]
    assert k.core.op == "matmul" and len(k.body) >= 8
    assert [r.name for r in k.reads[:2]] == ["x", "w"]  # A, B first
    assert {r.name for r in k.reads} == {"x", "w", "b"}


def test_two_matmuls_materialize_the_intermediate():
    g = trace(lambda x, w1, w2: (x @ w1).relu() @ w2, {"x": (M, K), "w1": (K, N), "w2": (N, K)})
    ks = plan(g)
    assert kinds(ks) == ["matmul", "matmul"]
    assert len(ks[0].body) == 1 and ks[0].body[0].attrs["fn"] == "relu"  # relu is epilogue of the first


def test_matmul_into_reduce_is_split():
    g = trace(lambda x, w: ((x @ w) * 2.0).sum(), {"x": (M, K), "w": (K, N)})
    ks = plan(g)
    assert kinds(ks) == ["matmul", "reduce"]
    assert ks[1].body and ks[1].body[0].attrs["fn"] == "mul"  # the *2 is a reduce prologue


def test_shared_matmul_is_never_duplicated():
    def f(x, w):
        y = x @ w
        return y.exp(), y.tanh()

    ks = plan(trace(f, {"x": (M, K), "w": (K, N)}))
    assert sum(k.kind == "matmul" for k in ks) == 1 and kinds(ks).count("matmul") == 1
    assert len(ks) == 3  # matmul, exp, tanh


def test_kernel_order_respects_dependencies():
    g = trace(lambda x, w1, w2: tc.softmax((x @ w1).relu() @ w2), {"x": (M, K), "w1": (K, N), "w2": (N, K)})
    ks = plan(g)
    done = {n.id for n in g.inputs.values()}
    for k in ks:
        assert all(r.id in done for r in k.reads), k
        done.add(k.out.id)
