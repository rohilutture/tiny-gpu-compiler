"""Reference interpreter: evaluates the (unfused) graph op by op with numpy. The ground truth for tests."""
from __future__ import annotations

import numpy as np

from .ir import Graph

_UN = {
    "neg": lambda a: -a,
    "exp": np.exp,
    "relu": lambda a: np.maximum(a, 0.0),
    "tanh": np.tanh,
    "sqrt": np.sqrt,
    "rsqrt": lambda a: 1.0 / np.sqrt(a),
    "recip": lambda a: 1.0 / a,
}
_BIN = {
    "add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide, "max": np.maximum,
}


def interpret(g: Graph, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    val: dict[int, np.ndarray] = {}
    for n in g.nodes:
        if n.op == "input":
            a = np.asarray(feeds[n.name], dtype=np.float32)
            assert a.shape == n.shape, f"{n.name}: expected {n.shape}, got {a.shape}"
            val[n.id] = a
        elif n.op == "const":
            val[n.id] = np.full((1, 1), n.attrs["value"], dtype=np.float32)
        elif n.op == "unary":
            val[n.id] = _UN[n.attrs["fn"]](val[n.inputs[0].id]).astype(np.float32)
        elif n.op == "binary":
            val[n.id] = _BIN[n.attrs["fn"]](val[n.inputs[0].id], val[n.inputs[1].id]).astype(np.float32)
        elif n.op == "rowreduce":
            f = np.sum if n.attrs["fn"] == "sum" else np.max
            val[n.id] = f(val[n.inputs[0].id], axis=1, keepdims=True).astype(np.float32)
        elif n.op == "matmul":
            val[n.id] = (val[n.inputs[0].id] @ val[n.inputs[1].id]).astype(np.float32)
        else:
            raise ValueError(n.op)
    return [val[o.id] for o in g.outputs]
