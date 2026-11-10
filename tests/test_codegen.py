

def test_reduce_kernels_loop_over_the_input_columns_not_the_output_width():
    """Regression: the first GPU run reduced only column 0 because the (M, 1) output shape was passed as (M, N)."""
    import tinyc as tc
    from tinyc import codegen

    cg = tc.compile(tc.softmax, {"x": (4, 200)})
    reduces = [k for k in cg.kernels if k.kind == "reduce"]
    assert reduces and all(codegen.problem_dims(k) == (4, 200, 200) for k in reduces)
