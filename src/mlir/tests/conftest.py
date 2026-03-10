"""
AIPUGC MLIR compiler test infrastructure.

Provides fixtures, CLI options, and helper functions for testing
all registered ops with aipuir IR generation and aipugc compilation.

Generated graph.def files are cached under ``--graph-dir`` (default:
``tests/generated_graphs/``) so that expensive ``aipuir`` calls are
only run once.  Pass ``--regenerate`` to force re-generation.
"""

import os
import subprocess
import pytest


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--aipugc-path",
        default=None,
        help="Path to aipugc executable. Falls back to AIPUGC_PATH env var, "
             "then tries build/tool/aipugc/aipugc relative to project root.",
    )
    parser.addoption(
        "--aipuir-path",
        default=None,
        help="Path to aipuir executable. Falls back to AIPUIR_PATH env var, "
             "then assumes 'aipuir' is on PATH.",
    )
    parser.addoption(
        "--skip-compile",
        action="store_true",
        default=False,
        help="Only test IR generation (skip aipugc compilation step).",
    )
    parser.addoption(
        "--graph-dir",
        default=None,
        help="Directory to cache generated graph.def files. "
             "Default: tests/generated_graphs/",
    )
    parser.addoption(
        "--regenerate",
        action="store_true",
        default=False,
        help="Force re-generation of all graph.def files even if cached.",
    )


@pytest.fixture(scope="session")
def aipugc_bin(request):
    """Resolve the aipugc executable path."""
    path = request.config.getoption("--aipugc-path")
    if path is None:
        path = os.environ.get("AIPUGC_PATH")
    if path is None:
        candidate = os.path.normpath(os.path.join(
            os.path.dirname(__file__), "..", "build",
            "tool", "aipugc", "aipugc",
        ))
        if os.path.isfile(candidate):
            path = candidate
    return path


@pytest.fixture(scope="session")
def aipuir_bin(request):
    """Resolve the aipuir executable path."""
    path = request.config.getoption("--aipuir-path")
    if path is None:
        path = os.environ.get("AIPUIR_PATH")
    if path is None:
        path = "aipuir"
    return path


@pytest.fixture(scope="session")
def skip_compile(request):
    return request.config.getoption("--skip-compile")


@pytest.fixture(scope="session")
def graph_dir(request):
    """Persistent directory that caches generated graph.def files."""
    path = request.config.getoption("--graph-dir")
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "generated_graphs")
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def regenerate(request):
    return request.config.getoption("--regenerate")


def run_aipuir(aipuir_bin, layer_type, params=None, output_path="graph.def",
               cwd=None, timeout=120):
    """
    Run ``aipuir`` to generate an IR file.

    Returns
    -------
    subprocess.CompletedProcess
    """
    cmd = [aipuir_bin, layer_type]
    if params:
        cmd.extend(["-p", params])
    cmd.extend(["-o", output_path])
    return subprocess.run(
        cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout,
    )


def run_aipugc(aipugc_bin, graph_def_path, extra_args=None, timeout=180):
    """
    Run ``aipugc`` to compile a graph.def file.

    Returns
    -------
    subprocess.CompletedProcess
    """
    cmd = [aipugc_bin, graph_def_path, "--mlir-print-debuginfo"]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
    )


def ensure_graph(aipuir_bin, graph_dir, test_id, op_name, param_str,
                 regenerate=False):
    """
    Return the path to a cached ``graph.def``, generating it if necessary.

    File layout::

        <graph_dir>/<test_id>/graph.def

    Returns ``(graph_def_path, gen_result_or_None)``.
    ``gen_result`` is *None* when the file was served from cache.
    """
    case_dir = os.path.join(graph_dir, test_id)
    graph_def = os.path.join(case_dir, "graph.def")

    if os.path.isfile(graph_def) and not regenerate:
        return graph_def, None

    os.makedirs(case_dir, exist_ok=True)
    result = run_aipuir(
        aipuir_bin, op_name, params=param_str,
        output_path=graph_def, cwd=case_dir,
    )
    return graph_def, result


def build_param_string(parts):
    """
    Merge a list of param fragments into a single ``-p`` value.

    Example::

        build_param_string(["is=[1,10,10,8]", "it=int16", "kx=3"])
        # => "is=[1,10,10,8]; it=int16; kx=3"

    Empty / None fragments are silently skipped.
    """
    return "; ".join(p for p in parts if p)
