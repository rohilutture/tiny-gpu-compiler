"""Fusion planner: partitions the graph into kernels.

Every node in the *materialized set* S is written to memory by exactly one kernel; that kernel's
body is the backward cone of S-free nodes feeding it, computed in registers. Rules:

  * S starts as {graph outputs, all rowreduce nodes}.
  * A non-S node needed by two or more kernels is either RECOMPUTED in each (cheap cone, no
    matmul) or MATERIALIZED (cone size > remat_limit, or its cone contains a matmul: matmul
    work is never duplicated).
  * A kernel contains at most one matmul; matmul operands must be in memory; a rowreduce
    prologue may not contain a matmul. Violations materialize the offending matmul/operand.
  * A matmul followed by elementwise ops fuses them as an epilogue (bias, activation).

The loop runs to a fixed point. Kernel kinds: "ewise", "reduce", "matmul" (with epilogue).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .ir import Graph, Node


@dataclass(eq=False)
class Kernel:
    kind: str
    out: Node                      # node written to memory
    body: list[Node]               # nodes computed in registers, topological; includes `out` for ewise/matmul
    reads: list[Node]              # memory-resident nodes read (matmul: A, B first)
    core: Node | None = None       # the matmul / rowreduce node
    config: dict = field(default_factory=dict)

    def __repr__(self):
        return f"Kernel({self.kind}, out={self.out!r}, fused={[n.id for n in self.body]}, reads={[r.id for r in self.reads]})"


def _size(n: Node) -> int:
    return n.shape[0] * n.shape[1]


def plan(g: Graph, remat_limit: int = 8) -> list[Kernel]:
    S: set[int] = {o.id for o in g.outputs} | {n.id for n in g.nodes if n.op == "rowreduce"}
    by_id = {n.id: n for n in g.nodes}

    def cone(m: Node):
        nodes: set[int] = set()
        reads: dict[int, Node] = {}

        def visit(n: Node):
            if n.op == "const" or n.id in nodes:
                return
            if n.op == "input" or (n.id in S and n is not m):
                reads.setdefault(n.id, n)
                return
            nodes.add(n.id)
            for i in n.inputs:
                if n.op == "matmul":  # operands are always read from memory
                    if i.op != "const":
                        reads.setdefault(i.id, i)
                else:
                    visit(i)

        visit(m)
        return nodes, reads

    def has_matmul(n: Node) -> bool:
        nodes, _ = cone(n)
        return any(by_id[i].op == "matmul" for i in nodes)

    while True:
        changed = False
        cones = {i: cone(by_id[i]) for i in sorted(S) if by_id[i].op != "input"}

        # matmul operands must be materialized
        for i, (nodes, _) in cones.items():
            for nid in nodes:
                n = by_id[nid]
                if n.op == "matmul":
                    for op in n.inputs:
                        if op.op not in ("input", "const") and op.id not in S:
                            S.add(op.id); changed = True
        if changed:
            continue

        # one matmul per kernel; none inside a reduce prologue
        for i, (nodes, _) in cones.items():
            mms = sorted(nid for nid in nodes if by_id[nid].op == "matmul")
            m = by_id[i]
            if m.op == "rowreduce" and mms:
                S.update(mms); changed = True
            elif len(mms) > 1:
                S.update(mms[:-1]); changed = True
        if changed:
            continue

        # shared non-S nodes: recompute (cheap) or materialize
        member: dict[int, int] = {}
        for i, (nodes, _) in cones.items():
            for nid in nodes:
                if nid != i:
                    member[nid] = member.get(nid, 0) + 1
        for nid, cnt in member.items():
            if cnt >= 2 and nid not in S:
                n = by_id[nid]
                nodes, _ = cone(n)
                if n.op == "matmul" or any(by_id[x].op == "matmul" for x in nodes) or len(nodes) > remat_limit:
                    S.add(nid); changed = True
        if not changed:
            break

    kernels = []
    for i in sorted(S):
        m = by_id[i]
        if m.op == "input":
            continue
        nodes, reads = cone(m)
        body_nodes = sorted(nodes)
        mm = [x for x in body_nodes if by_id[x].op == "matmul"]
        if m.op == "rowreduce":
            kind, core = "reduce", m
            body = [by_id[x] for x in body_nodes if x != m.id]
            rd = list(reads.values())
        elif mm:
            kind, core = "matmul", by_id[mm[0]]
            body = [by_id[x] for x in body_nodes if x != core.id]
            a, b = core.inputs
            rd = [a, b] + [r for r in reads.values() if r.id not in (a.id, b.id)]
        else:
            kind, core = "ewise", None
            body = [by_id[x] for x in body_nodes]
            rd = list(reads.values())
        kernels.append(Kernel(kind, m, body, rd, core))
    return kernels


def modeled_bytes(kernels: list[Kernel]) -> int:
    """Modeled DRAM traffic in bytes (fp32): every kernel reads its inputs once and writes its output once."""
    return 4 * sum(sum(_size(r) for r in k.reads) + _size(k.out) for k in kernels)


def unfused_kernels(g: Graph) -> list[Kernel]:
    """Baseline: one kernel per op, every intermediate round-trips through memory."""
    out = []
    for n in g.nodes:
        if n.is_leaf:
            continue
        reads = [i for i in n.inputs if i.op != "const"]
        kind = {"rowreduce": "reduce", "matmul": "matmul"}.get(n.op, "ewise")
        out.append(Kernel(kind, n, [] if kind != "ewise" else [n], reads, n if kind != "ewise" else None))
    return out


def explain(g: Graph, kernels: list[Kernel]) -> str:
    base = unfused_kernels(g)
    lines = [f"{len(kernels)} kernels (unfused: {len(base)}), modeled traffic "
             f"{modeled_bytes(kernels) / 1e6:.2f} MB (unfused {modeled_bytes(base) / 1e6:.2f} MB)"]
    for k in kernels:
        fused = ", ".join(f"{n.op}.{n.attrs.get('fn', '')}" if n.attrs.get("fn") else n.op for n in k.body)
        lines.append(f"  [{k.kind:6}] -> %{k.out.id}{k.out.shape} reads {[f'%{r.id}' for r in k.reads]} fuses: {fused or '-'}"
                     + (f"  cfg={k.config}" if k.config else ""))
    return "\n".join(lines)
