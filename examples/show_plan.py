"""Print the fusion plan and generated CUDA for a few graphs. CPU only (no GPU needed).

    python examples/show_plan.py
"""
import tinyc as tc

M, N, K = 4096, 4096, 1024

print("=== gelu(x + b) ===")
cg = tc.compile(lambda x, b: tc.gelu(x + b), {"x": (M, N), "b": (1, N)})
print(cg.explain()); print(cg.source(0))

print("\n=== softmax ===")
cg = tc.compile(tc.softmax, {"x": (M, N)})
print(cg.explain())
for i in range(len(cg.kernels)):
    print(f"--- kernel {i} ---"); print(cg.source(i))

print("\n=== gelu(x @ w + b)  (matmul with fused bias+gelu epilogue) ===")
cg = tc.compile(lambda x, w, b: tc.gelu(x @ w + b), {"x": (M, K), "w": (K, N), "b": (1, N)})
print(cg.explain()); print(cg.source(0))
