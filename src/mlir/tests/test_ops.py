"""
Parametrized tests for all registered AIPUGC ops.

Test flow per case:
    1. ``aipuir <layer_type> -p '<params>' -o graph.def``  → generate IR (cached)
    2. ``aipugc graph.def --mlir-print-debuginfo``          → compile IR

Generated ``graph.def`` files are stored under ``--graph-dir`` (default
``tests/generated_graphs/``) and reused across runs.  Pass ``--regenerate``
to force a fresh generation.

Run examples::

    # first run generates all graph.def and runs compile tests
    pytest tests/ -v

    # subsequent runs reuses cached graph.def, much faster
    pytest tests/ -v

    # force re-generation
    pytest tests/ -v --regenerate

    # only convolution ops
    pytest tests/ -v -m conv

    # only int8 cases
    pytest tests/ -v -k int8

    # generate IR only (skip compilation)
    pytest tests/ --skip-compile

    # specify tool paths
    pytest tests/ --aipuir-path=/path/to/aipuir --aipugc-path=/path/to/aipugc
"""

import os
import pytest

from conftest import run_aipugc, ensure_graph
from op_configs import ALL_TEST_CASES


# ---------------------------------------------------------------------------
# Parametrized test: IR Generation (with caching)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "test_id, op_name, param_str, category",
    ALL_TEST_CASES,
    ids=[tc[0] for tc in ALL_TEST_CASES],
)
def test_ir_generation(test_id, op_name, param_str, category,
                       aipuir_bin, graph_dir, regenerate):
    """aipuir must produce a valid graph.def for the given op + params."""
    graph_def, result = ensure_graph(
        aipuir_bin, graph_dir, test_id, op_name, param_str,
        regenerate=regenerate,
    )

    if result is not None:
        assert result.returncode == 0, (
            f"aipuir failed for {op_name} with params '{param_str}'\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )

    assert os.path.isfile(graph_def), (
        f"graph.def not found at {graph_def}"
    )


# ---------------------------------------------------------------------------
# Parametrized test: aipugc Compilation (reads cached graph.def)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "test_id, op_name, param_str, category",
    ALL_TEST_CASES,
    ids=[tc[0] for tc in ALL_TEST_CASES],
)
def test_aipugc_compile(test_id, op_name, param_str, category,
                        aipuir_bin, aipugc_bin, skip_compile,
                        graph_dir, regenerate):
    """aipugc must compile the cached graph.def without errors."""
    if skip_compile:
        pytest.skip("--skip-compile flag set")
    if aipugc_bin is None:
        pytest.skip("aipugc executable not found; "
                    "set --aipugc-path or AIPUGC_PATH")

    graph_def, gen_result = ensure_graph(
        aipuir_bin, graph_dir, test_id, op_name, param_str,
        regenerate=regenerate,
    )

    if gen_result is not None and gen_result.returncode != 0:
        pytest.skip(
            f"aipuir failed (pre-condition); skipping compile.\n"
            f"stderr: {gen_result.stderr[:500]}"
        )
    if not os.path.isfile(graph_def):
        pytest.skip(f"graph.def missing for {test_id}; run IR gen first")

    comp = run_aipugc(aipugc_bin, graph_def)
    assert comp.returncode == 0, (
        f"aipugc compilation failed for {op_name} ({test_id})\n"
        f"params: {param_str}\n"
        f"--- stdout ---\n{comp.stdout[-2000:]}\n"
        f"--- stderr ---\n{comp.stderr[-2000:]}"
    )


# ---------------------------------------------------------------------------
# Smoke tests: default IR (no params) for every op
# ---------------------------------------------------------------------------

_DEFAULT_OPS = [
    "Abs", "Negative", "Square", "Sqrt", "Rsqrt", "Cosine", "Exp", "Log",
    "Quantize",
    "Add", "Sub", "Mul", "Div", "Eltwise", "Pow", "SquaredDifference",
    "Convolution", "DepthwiseConvolution", "Pooling",
    "Activation", "Cast",
    "BatchNorm", "LayerNorm", "RMSNorm",
    "FullyConnected", "MatMul",
    "Softmax", "Reduce", "ArgMinMax",
    "Reshape", "Transpose", "Tile", "Concat", "Split", "Slice", "Pad",
    "Crop", "Gather", "ChannelShuffle",
    "DepthToSpace", "SpaceToDepth", "BatchToSpace", "SpaceToBatch",
    "Resize", "GridSample",
    "ReverseSequence", "BasicLSTM",
]


@pytest.mark.parametrize("op_name", _DEFAULT_OPS, ids=_DEFAULT_OPS)
def test_default_ir(op_name, aipuir_bin, graph_dir, regenerate):
    """aipuir should generate a valid default IR for every supported op."""
    test_id = f"default_{op_name}"
    graph_def, result = ensure_graph(
        aipuir_bin, graph_dir, test_id, op_name, param_str=None,
        regenerate=regenerate,
    )

    if result is not None:
        assert result.returncode == 0, (
            f"aipuir default IR generation failed for '{op_name}'\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    assert os.path.isfile(graph_def), f"graph.def missing for {op_name}"


@pytest.mark.parametrize("op_name", _DEFAULT_OPS, ids=_DEFAULT_OPS)
def test_default_compile(op_name, aipuir_bin, aipugc_bin, skip_compile,
                         graph_dir, regenerate):
    """aipugc should compile the default IR for every supported op."""
    if skip_compile:
        pytest.skip("--skip-compile flag set")
    if aipugc_bin is None:
        pytest.skip("aipugc not found")

    test_id = f"default_{op_name}"
    graph_def, gen_result = ensure_graph(
        aipuir_bin, graph_dir, test_id, op_name, param_str=None,
        regenerate=regenerate,
    )
    if gen_result is not None and gen_result.returncode != 0:
        pytest.skip(f"aipuir failed for default {op_name}")
    if not os.path.isfile(graph_def):
        pytest.skip(f"graph.def missing for default {op_name}")

    comp = run_aipugc(aipugc_bin, graph_def)
    assert comp.returncode == 0, (
        f"aipugc failed for default {op_name}\n"
        f"--- stderr ---\n{comp.stderr[-2000:]}"
    )


# ---------------------------------------------------------------------------
# Multi-layer combo tests
# ---------------------------------------------------------------------------

_COMBO_CASES = [
    ("Conv_Pool_Eltwise",
     "Convolution_Pooling_Eltwise",
     "is=[1,20,20,8]"),
    ("Conv_Eltwise_Pool_Abs",
     "Convolution_Eltwise_Pooling_Abs",
     "is=[1,16,16,3]"),
    ("Conv_BatchNorm",
     "Convolution_BatchNorm",
     "is=[1,10,10,8]"),
    ("Conv_Activation",
     "Convolution_Activation",
     "is=[1,10,10,3]"),
    ("Add_Softmax",
     "Add_Softmax",
     "is=[1,10,10,8]"),
    ("Cast_Eltwise",
     "Cast_Eltwise",
     "is=[1,10,10,8]"),
]


@pytest.mark.parametrize(
    "label, layer_spec, params",
    _COMBO_CASES,
    ids=[c[0] for c in _COMBO_CASES],
)
def test_multi_layer_ir(label, layer_spec, params,
                        aipuir_bin, graph_dir, regenerate):
    """Multi-layer IR generation via underscore-connected layer names."""
    test_id = f"combo_{label}"
    graph_def, result = ensure_graph(
        aipuir_bin, graph_dir, test_id, layer_spec, params,
        regenerate=regenerate,
    )
    if result is not None:
        assert result.returncode == 0, (
            f"Multi-layer IR gen failed: {layer_spec}\n"
            f"stderr: {result.stderr}"
        )
    assert os.path.isfile(graph_def)


@pytest.mark.parametrize(
    "label, layer_spec, params",
    _COMBO_CASES,
    ids=[c[0] for c in _COMBO_CASES],
)
def test_multi_layer_compile(label, layer_spec, params,
                             aipuir_bin, aipugc_bin, skip_compile,
                             graph_dir, regenerate):
    """aipugc should compile multi-layer IR."""
    if skip_compile:
        pytest.skip("--skip-compile flag set")
    if aipugc_bin is None:
        pytest.skip("aipugc not found")

    test_id = f"combo_{label}"
    graph_def, gen_result = ensure_graph(
        aipuir_bin, graph_dir, test_id, layer_spec, params,
        regenerate=regenerate,
    )
    if gen_result is not None and gen_result.returncode != 0:
        pytest.skip(f"aipuir failed for {layer_spec}")
    if not os.path.isfile(graph_def):
        pytest.skip(f"graph.def missing for combo {label}")

    comp = run_aipugc(aipugc_bin, graph_def)
    assert comp.returncode == 0, (
        f"aipugc multi-layer compile failed: {layer_spec}\n"
        f"stderr: {comp.stderr[-2000:]}"
    )


# ---------------------------------------------------------------------------
# Float-specific tests
# ---------------------------------------------------------------------------

_FLOAT_OPS = [
    ("Softmax_f16", "Softmax", "is=[1,20,20,30]; float_ir=true"),
    ("Softmax_bf16", "Softmax",
     "is=[1,20,20,30]; float_ir=true; it=bfloat16"),
    ("Conv_f16", "Convolution", "is=[1,10,10,8]; float_ir=true; kx=3"),
    ("Add_f16", "Add", "is=[1,10,10,8]; float_ir=true"),
    ("MatMul_f16", "MatMul", "is=[1,8,16]; float_ir=true"),
    ("Reduce_f16", "Reduce", "is=[1,10,10,8]; float_ir=true"),
    ("LayerNorm_f16", "LayerNorm", "is=[1,10,10,8]; float_ir=true"),
]


@pytest.mark.float_only
@pytest.mark.parametrize(
    "label, op_name, params",
    _FLOAT_OPS,
    ids=[c[0] for c in _FLOAT_OPS],
)
def test_float_ir(label, op_name, params,
                  aipuir_bin, graph_dir, regenerate):
    """Float IR variants must generate successfully."""
    test_id = f"float_{label}"
    graph_def, result = ensure_graph(
        aipuir_bin, graph_dir, test_id, op_name, params,
        regenerate=regenerate,
    )
    if result is not None:
        assert result.returncode == 0, (
            f"Float IR generation failed for {op_name}\n"
            f"stderr: {result.stderr}"
        )
    assert os.path.isfile(graph_def)
