#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class DTypeProfile:
    name: str
    param_suffix: str


@dataclass(frozen=True)
class ShapeProfile:
    name: str
    shape_expr: str
    rank: int


@dataclass
class CaseTiming:
    name: str
    op: str
    generate_sec: float
    compile_sec: float
    total_sec: float
    success: bool


OP_ALIASES: Dict[str, Sequence[str]] = {
    "Activation": ("Activation",),
    "Add": ("Add",),
    "Argminmax": ("ArgMinMax", "Argminmax"),
    "BasicLSTM": ("BasicLSTM",),
    "BatchNorm": ("BatchNorm",),
    "BatchToSpace": ("BatchToSpace",),
    "Cast": ("Cast",),
    "ChannelShuffle": ("ChannelShuffle",),
    "Concat": ("Concat",),
    "Constant": ("Constant",),
    "Convolution": ("Convolution",),
    "Crop": ("Crop",),
    "DepthToSpace": ("DepthToSpace",),
    "Div": ("Div",),
    "Eltwise": ("Eltwise",),
    "FullyConnected": ("FullyConnected",),
    "Gather": ("Gather",),
    "GridSample": ("GridSample",),
    "Input": ("Input",),
    "LayerNorm": ("LayerNorm",),
    "Matmul": ("MatMul", "Matmul"),
    "Mul": ("Mul",),
    "Pad": ("Pad",),
    "Pool2D": ("Pooling", "Pool2D"),
    "Pow": ("Pow",),
    "RMSNorm": ("RMSNorm",),
    "Reduce": ("Reduce",),
    "Reshape": ("Reshape",),
    "Resize": ("Resize",),
    "ReverseSequance": ("ReverseSequence", "ReverseSequance"),
    "Slice": ("Slice",),
    "Softmax": ("Softmax",),
    "SpaceToBatch": ("SpaceToBatch",),
    "SpaceToDepth": ("SpaceToDepth",),
    "Split": ("Split",),
    "SquareDifference": ("SquaredDifference", "SquareDifference"),
    "Sub": ("Sub",),
    "Tile": ("Tile",),
    "Transpose": ("Transpose",),
    "UnaryWithoutParameterOp": (),
}


DTYPE_PROFILES = (
    DTypeProfile("int8", "it=int8"),
    DTypeProfile("uint8", "it=uint8"),
    DTypeProfile("float16", "float_ir=true"),
    DTypeProfile("bfloat16", "float_ir=true; it=bfloat16"),
)

SHAPE_PROFILES = (
    ShapeProfile("2d_small", "is=[1,16]", 2),
    ShapeProfile("3d_small", "is=[1,8,8]", 3),
    ShapeProfile("4d_small", "is=[1,8,8,8]", 4),
)


def run_cmd(cmd: Sequence[str], cwd: Path) -> Tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout


def has_aipugc_dialect_ir(output: str) -> bool:
    markers = ("aipugc.func", "aipugc.", "AIPUDialect")
    return any(marker in output for marker in markers)


def discover_ops() -> List[str]:
    mlirgen_dir = Path(__file__).resolve().parents[3] / "mlir" / "aipugc" / "mlirgen"
    ops: List[str] = []
    for cpp in sorted(mlirgen_dir.glob("*.cpp")):
        stem = cpp.stem
        if stem in {"MLIRGenRegister", "MLIRGen", "utils"}:
            continue
        ops.append(stem)
    return ops


def generate_graph(
    aipuir: str,
    layer_name: str,
    params: str,
    out_graph: Path,
    work_dir: Path,
) -> Tuple[bool, str]:
    cmd = [aipuir, layer_name, "-p", params, "-o", str(out_graph)]
    code, out = run_cmd(cmd, work_dir)
    return code == 0 and out_graph.exists(), out


def compile_to_aipugc(
    aipugc_bin: str, graph_path: Path, work_dir: Path
) -> Tuple[bool, str]:
    cmd = [aipugc_bin, str(graph_path), "--mlir-print-debuginfo"]
    code, out = run_cmd(cmd, work_dir)
    if code != 0:
        return False, out
    return has_aipugc_dialect_ir(out), out


def fmt_sec(seconds: float) -> str:
    return f"{seconds:.3f}s"


def main() -> int:
    default_aipugc_path = (
        Path(__file__).resolve().parents[3] / "build" / "tool" / "aipugc" / "aipugc"
    )
    default_work_dir = (
        Path(__file__).resolve().parents[3]
        / "build"
        / "test"
        / "aipugc"
        / "tmp_work"
    )

    default_aipuir = os.getenv("AIPUIR_BIN", "aipuir")
    default_aipugc = os.getenv("AIPUGC_BIN", str(default_aipugc_path))
    default_work = os.getenv("AIPUGC_TEST_WORK_DIR", str(default_work_dir))

    parser = argparse.ArgumentParser(
        description=(
            "Generate single-op IR by aipuir with shape/dim/dtype matrix, then "
            "verify graph.def can be translated into aipugc dialect IR."
        )
    )
    parser.add_argument(
        "--aipuir",
        default=default_aipuir,
        help="Path or command of aipuir. Default from $AIPUIR_BIN or 'aipuir'.",
    )
    parser.add_argument(
        "--aipugc",
        default=default_aipugc,
        help=(
            "Path or command of aipugc. Default from $AIPUGC_BIN or "
            "'<repo>/src/mlir/build/tool/aipugc/aipugc'."
        ),
    )
    parser.add_argument(
        "--work-dir",
        default=default_work,
        help=(
            "Temporary working directory. Default from $AIPUGC_TEST_WORK_DIR or "
            "'<repo>/src/mlir/build/test/aipugc/tmp_work'."
        ),
    )
    parser.add_argument(
        "--strict-tools",
        action="store_true",
        help="Fail immediately if aipuir or aipugc is not found.",
    )
    args = parser.parse_args()

    aipuir_cmd = args.aipuir
    if shutil.which(aipuir_cmd) is None:
        msg = f"[SKIP] aipuir not found: {args.aipuir}"
        print(msg)
        return 1 if args.strict_tools else 0

    aipugc_cmd = args.aipugc
    aipugc_path = Path(aipugc_cmd)
    if aipugc_path.exists():
        aipugc_cmd = str(aipugc_path.resolve())
    else:
        which_path = shutil.which(aipugc_cmd)
        if which_path is None:
            msg = f"[SKIP] aipugc not found: {args.aipugc}"
            print(msg)
            return 1 if args.strict_tools else 0
        aipugc_cmd = which_path

    print(f"[INFO] aipuir={aipuir_cmd}")
    print(f"[INFO] aipugc={aipugc_cmd}")
    print(f"[INFO] work_dir={args.work_dir}")

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if not work_dir.exists():
        msg = f"[SKIP] unable to create work dir: {work_dir}"
        print(msg)
        return 1 if args.strict_tools else 0

    if not Path(aipugc_cmd).exists() and shutil.which(aipugc_cmd) is None:
        msg = f"[SKIP] aipugc not found: {args.aipugc}"
        print(msg)
        return 1 if args.strict_tools else 0

    run_start = time.perf_counter()
    ops = discover_ops()
    total = 0
    failed_cases: List[str] = []
    case_timings: List[CaseTiming] = []
    op_stage_time: Dict[str, Dict[str, float]] = {
        op: {"generate": 0.0, "compile": 0.0, "total": 0.0, "count": 0.0}
        for op in ops
    }

    for op_file_stem in ops:
        aliases = OP_ALIASES.get(op_file_stem, (op_file_stem,))
        if not aliases:
            continue

        per_op_success = 0
        per_op_ranks = set()
        per_op_dtype = set()

        for dtype in DTYPE_PROFILES:
            dtype_done = False
            for shape in SHAPE_PROFILES:
                case_start = time.perf_counter()
                total += 1
                case_name = f"{op_file_stem}:{dtype.name}:{shape.name}"
                case_dir = work_dir / op_file_stem / dtype.name / shape.name
                case_dir.mkdir(parents=True, exist_ok=True)
                graph_path = case_dir / "graph.def"

                params = f"{shape.shape_expr}; {dtype.param_suffix}"
                generated = False
                last_generate_log = ""
                generate_sec = 0.0
                for layer_name in aliases:
                    t0 = time.perf_counter()
                    ok, out = generate_graph(
                        aipuir_cmd, layer_name, params, graph_path, case_dir
                    )
                    generate_sec += time.perf_counter() - t0
                    last_generate_log = out
                    if ok:
                        generated = True
                        break

                if not generated:
                    total_sec = time.perf_counter() - case_start
                    case_timings.append(
                        CaseTiming(
                            name=case_name,
                            op=op_file_stem,
                            generate_sec=generate_sec,
                            compile_sec=0.0,
                            total_sec=total_sec,
                            success=False,
                        )
                    )
                    op_stage_time[op_file_stem]["generate"] += generate_sec
                    op_stage_time[op_file_stem]["total"] += total_sec
                    op_stage_time[op_file_stem]["count"] += 1
                    failed_cases.append(
                        f"{case_name} generate failed.\n{last_generate_log}"
                    )
                    continue

                t1 = time.perf_counter()
                ok, out = compile_to_aipugc(aipugc_cmd, graph_path, case_dir)
                compile_sec = time.perf_counter() - t1
                total_sec = time.perf_counter() - case_start
                case_timings.append(
                    CaseTiming(
                        name=case_name,
                        op=op_file_stem,
                        generate_sec=generate_sec,
                        compile_sec=compile_sec,
                        total_sec=total_sec,
                        success=ok,
                    )
                )
                op_stage_time[op_file_stem]["generate"] += generate_sec
                op_stage_time[op_file_stem]["compile"] += compile_sec
                op_stage_time[op_file_stem]["total"] += total_sec
                op_stage_time[op_file_stem]["count"] += 1
                if not ok:
                    failed_cases.append(
                        f"{case_name} convert to aipugc dialect failed.\n{out}"
                    )
                    continue

                per_op_success += 1
                per_op_ranks.add(shape.rank)
                per_op_dtype.add(dtype.name)
                dtype_done = True
                print(
                    f"[PASS] {case_name} "
                    f"(gen={fmt_sec(generate_sec)}, lower={fmt_sec(compile_sec)}, "
                    f"total={fmt_sec(total_sec)})"
                )
                break

            if not dtype_done:
                failed_cases.append(
                    f"{op_file_stem}: no valid case for dtype profile '{dtype.name}'"
                )

        if len(per_op_ranks) < 2:
            failed_cases.append(
                f"{op_file_stem}: dim coverage too low, got ranks={sorted(per_op_ranks)}"
            )
        if len(per_op_dtype) < 2:
            failed_cases.append(
                f"{op_file_stem}: dtype coverage too low, got dtypes={sorted(per_op_dtype)}"
            )
        if per_op_success == 0:
            failed_cases.append(f"{op_file_stem}: no successful case")

    run_sec = time.perf_counter() - run_start
    total_gen_sec = sum(c.generate_sec for c in case_timings)
    total_compile_sec = sum(c.compile_sec for c in case_timings)
    total_case_sec = sum(c.total_sec for c in case_timings)
    infra_overhead_sec = max(0.0, run_sec - total_case_sec)
    print(
        "[TIME] run="
        f"{fmt_sec(run_sec)}, generate={fmt_sec(total_gen_sec)}, "
        f"lower={fmt_sec(total_compile_sec)}, case_total={fmt_sec(total_case_sec)}, "
        f"infra_overhead={fmt_sec(infra_overhead_sec)}"
    )

    if case_timings:
        slow_cases = sorted(case_timings, key=lambda c: c.total_sec, reverse=True)[:10]
        print("[TIME] top slow cases:")
        for idx, c in enumerate(slow_cases, start=1):
            status = "PASS" if c.success else "FAIL"
            print(
                f"  {idx:02d}. {c.name} [{status}] "
                f"gen={fmt_sec(c.generate_sec)}, lower={fmt_sec(c.compile_sec)}, "
                f"total={fmt_sec(c.total_sec)}"
            )

    op_rows = []
    for op_name, stat in op_stage_time.items():
        count = int(stat["count"])
        if count == 0:
            continue
        op_rows.append(
            (
                op_name,
                count,
                stat["generate"],
                stat["compile"],
                stat["total"],
                stat["total"] / count,
            )
        )
    if op_rows:
        op_rows.sort(key=lambda x: x[5], reverse=True)
        print("[TIME] top slow ops (avg case time):")
        for idx, (op_name, count, gen_s, comp_s, total_s, avg_s) in enumerate(
            op_rows[:10], start=1
        ):
            print(
                f"  {idx:02d}. {op_name} "
                f"count={count}, avg={fmt_sec(avg_s)}, "
                f"gen={fmt_sec(gen_s)}, lower={fmt_sec(comp_s)}, "
                f"total={fmt_sec(total_s)}"
            )

    print(f"[INFO] total attempted cases: {total}")
    if failed_cases:
        print(f"[FAIL] failed checks: {len(failed_cases)}")
        for item in failed_cases:
            print("-" * 80)
            print(item)
        return 1

    print("[RESULT] all op shape/dim/dtype matrix checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
