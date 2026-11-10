"""End-to-end: compiled kernels vs the numpy interpreter. Needs a CUDA device (briefly uses the GPU)."""
import numpy as np
import pytest
import torch

import tinyc as tc

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")]

M, N, K = 130, 200, 70  # deliberately not multiples of any tile or block size


def feeds_for(shapes, seed=0):
    r = np.random.default_rng(seed)
    return {n: (r.standard_normal(s) * 0.5).astype(np.float32) for n, s in shapes.items()}


CASES = {
    "gelu_bias": (lambda x, b: tc.gelu(x + b), {"x": (M, N), "b": (1, N)}),
    "softmax": (tc.softmax, {"x": (M, N)}),
    "matmul_bias_gelu": (lambda x, w, b: tc.gelu(x @ w + b), {"x": (M, K), "w": (K, N), "b": (1, N)}),
    "mlp": (lambda x, w1, w2: (x @ w1).relu() @ w2, {"x": (M, K), "w1": (K, N), "w2": (N, K)}),
    "matmul_reduce": (lambda x, w: ((x @ w) * 2.0).sum(), {"x": (M, K), "w": (K, N)}),
    "row_scale": (lambda x: x * (x * x).sum().recip().sqrt(), {"x": (M, N)}),
}


@pytest.mark.parametrize("name", CASES)
def test_matches_interpreter(name):
    fn, shapes = CASES[name]
    cg = tc.compile(fn, shapes)
    f = feeds_for(shapes)
    (ref,) = tc.interpret(cg.g, f)
    out = cg(**{k: torch.from_numpy(v).cuda() for k, v in f.items()})
    np.testing.assert_allclose(out.cpu().numpy(), ref, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("name", ["gelu_bias", "softmax", "matmul_bias_gelu"])
def test_every_config_is_correct(name):
    """Autotuning must only ever change speed, never results."""
    from tinyc import codegen

    fn, shapes = CASES[name]
    cg = tc.compile(fn, shapes)
    f = feeds_for(shapes)
    (ref,) = tc.interpret(cg.g, f)
    t = {k: torch.from_numpy(v).cuda() for k, v in f.items()}
    for k in cg.kernels:
        for cfg in codegen.configs(k.kind):
            k.config = cfg
            cg._loaded.clear()
            np.testing.assert_allclose(cg(**t).cpu().numpy(), ref, rtol=2e-4, atol=2e-4, err_msg=str(cfg))
