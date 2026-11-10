"""Graph IR and tracing frontend.

All tensors are 2-D fp32, row-major. Broadcasting is limited to the four shapes
(M,N), (1,N), (M,1), (1,1), which is enough for biases, row statistics and scalars.

Ops:
  input                      graph input (leaf)
  const     attrs.value      scalar literal (leaf, shape (1,1))
  unary     attrs.fn         neg exp relu tanh sqrt rsqrt recip
  binary    attrs.fn         add sub mul div max
  rowreduce attrs.fn         sum | max over the last axis -> (M,1)
  matmul                     (M,K) @ (K,N) -> (M,N)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Callable

UNARY = ("neg", "exp", "relu", "tanh", "sqrt", "rsqrt", "recip")
BINARY = ("add", "sub", "mul", "div", "max")
Shape = tuple[int, int]


@dataclass(eq=False)
class Node:
    id: int
    op: str
    inputs: list["Node"]
    shape: Shape
    attrs: dict = field(default_factory=dict)
    name: str | None = None

    def __repr__(self):  # compact, for debugging and explain()
        tag = self.attrs.get("fn") or self.name or (self.attrs.get("value") if self.op == "const" else "")
        return f"%{self.id}:{self.op}{('.' + str(tag)) if tag != '' else ''}{self.shape}"

    @property
    def is_leaf(self) -> bool:
        return self.op in ("input", "const")

    @property
    def is_ewise(self) -> bool:
        return self.op in ("unary", "binary")


def broadcast_shape(a: Shape, b: Shape) -> Shape:
    out = []
    for x, y in zip(a, b):
        if x == y or y == 1:
            out.append(x)
        elif x == 1:
            out.append(y)
        else:
            raise ValueError(f"cannot broadcast {a} with {b}")
    return (out[0], out[1])


class Graph:
    def __init__(self):
        self.nodes: list[Node] = []
        self.inputs: dict[str, Node] = {}
        self.outputs: list[Node] = []
        self._ids = count()

    def add(self, op, inputs, shape, **attrs) -> Node:
        n = Node(next(self._ids), op, list(inputs), tuple(shape), attrs)
        self.nodes.append(n)  # creation order is a valid topological order
        return n

    def consumers(self) -> dict[int, list[Node]]:
        c: dict[int, list[Node]] = {n.id: [] for n in self.nodes}
        for n in self.nodes:
            for i in n.inputs:
                c[i.id].append(n)
        return c

    def dump(self) -> str:
        return "\n".join(f"{n!r} <- {[i.id for i in n.inputs]}" for n in self.nodes)


class Tensor:
    """Tracing handle: operators append nodes to the graph."""

    def __init__(self, g: Graph, node: Node):
        self.g, self.node = g, node

    @property
    def shape(self) -> Shape:
        return self.node.shape

    def _wrap(self, other) -> "Tensor":
        if isinstance(other, Tensor):
            return other
        return Tensor(self.g, self.g.add("const", [], (1, 1), value=float(other)))

    def _bin(self, fn, other, reverse=False):
        o = self._wrap(other)
        a, b = (o, self) if reverse else (self, o)
        return Tensor(self.g, self.g.add("binary", [a.node, b.node], broadcast_shape(a.shape, b.shape), fn=fn))

    def _un(self, fn):
        return Tensor(self.g, self.g.add("unary", [self.node], self.shape, fn=fn))

    __add__ = lambda s, o: s._bin("add", o)
    __radd__ = lambda s, o: s._bin("add", o, True)
    __sub__ = lambda s, o: s._bin("sub", o)
    __rsub__ = lambda s, o: s._bin("sub", o, True)
    __mul__ = lambda s, o: s._bin("mul", o)
    __rmul__ = lambda s, o: s._bin("mul", o, True)
    __truediv__ = lambda s, o: s._bin("div", o)
    __rtruediv__ = lambda s, o: s._bin("div", o, True)
    __neg__ = lambda s: s._un("neg")

    def __matmul__(self, other: "Tensor") -> "Tensor":
        (m, k), (k2, n) = self.shape, other.shape
        if k != k2:
            raise ValueError(f"matmul shape mismatch {self.shape} @ {other.shape}")
        return Tensor(self.g, self.g.add("matmul", [self.node, other.node], (m, n)))

    def exp(self): return self._un("exp")
    def relu(self): return self._un("relu")
    def tanh(self): return self._un("tanh")
    def sqrt(self): return self._un("sqrt")
    def rsqrt(self): return self._un("rsqrt")
    def recip(self): return self._un("recip")

    def sum(self): return Tensor(self.g, self.g.add("rowreduce", [self.node], (self.shape[0], 1), fn="sum"))
    def max(self): return Tensor(self.g, self.g.add("rowreduce", [self.node], (self.shape[0], 1), fn="max"))


def gelu(x: Tensor) -> Tensor:
    """tanh approximation, built from primitives so the fuser sees every op."""
    return 0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh())


def softmax(x: Tensor) -> Tensor:
    e = (x - x.max()).exp()
    return e / e.sum()


def trace(fn: Callable, shapes: dict[str, Shape]) -> Graph:
    g = Graph()
    args = {}
    for name, shp in shapes.items():
        n = g.add("input", [], shp)
        n.name = name
        g.inputs[name] = n
        args[name] = Tensor(g, n)
    out = fn(**args)
    outs = out if isinstance(out, (tuple, list)) else (out,)
    g.outputs = [o.node for o in outs]
    return g
