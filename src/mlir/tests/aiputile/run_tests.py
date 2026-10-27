#!/usr/bin/env python3
"""Compile and execute real AIPUTile kernels using LLVM/MLIR 19 tools.

No device results are simulated. The portable harness does not link the private
graph builder; only source AIPUGC ops are parsed with allow-unregistered-dialect.
All generated AIPUTile, SCF, MemRef and LLVM IR is registered and verified.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile


def run(command, text=None, expected=None):
    result = subprocess.run([str(x) for x in command], input=text, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if expected is not None:
        if result.returncode == 0 or expected not in result.stderr:
            raise RuntimeError(f"Expected diagnostic {expected!r}: {command}\n{result.stderr}")
    elif result.returncode:
        raise RuntimeError(f"Failed ({result.returncode}): {command}\n{result.stderr}")
    return result.stdout


def source(shape, dtype="f32", reverse=False):
    tensor = "tensor<" + "x".join([*(str(d) for d in shape), dtype]) + ">"
    operands = "%a, %t" if reverse else "%t, %a"
    return f"""module {{
  func.func @kernel(%x: {tensor}, %b: {tensor}, %a: {tensor}) -> {tensor} {{
    %t = "aipugc.add"(%x, %b) : ({tensor}, {tensor}) -> {tensor}
    %y = "aipugc.mul"({operands}) : ({tensor}, {tensor}) -> {tensor}
    return %y : {tensor}
  }}
}}
"""


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    temporary = tempfile.TemporaryDirectory(prefix="aiputile-tests-")
    out = args.artifacts or Path(temporary.name)
    out.mkdir(parents=True, exist_ok=True)
    opt = [args.opt, "--allow-unregistered-dialect"]
    imported_pass = "lower-aipugc-to-aiputile"
    tests = []

    def optimize(text, pipeline, expected=None):
        return run([*opt, f"--pass-pipeline=builtin.module({pipeline})"], text, expected)

    def record(name):
        tests.append(name)
        print(f"PASS {name}", flush=True)

    def stages(text, tile, device=False):
        imported = optimize(text, imported_pass)
        assert "aipugc." not in imported and "aiputile.add" in imported
        fused = optimize(imported, f"aiputile-fuse-add-mul{{tile-size={tile}}}")
        assert "aiputile.fused_add_mul" in fused
        assert "aiputile.add " not in fused and "aiputile.mul " not in fused
        # Parser/printer round trip with the actual registered dialect.
        run([args.opt], fused)
        loops = optimize(fused, f"aiputile-lower-tiles{{device-address-space={int(device)}}}")
        assert loops.count("scf.for") == 2, loops
        assert loops.count("memref.load") == 3 and loops.count("memref.store") == 1
        assert "memref.alloc" not in loops and "aiputile.fused_add_mul" not in loops
        assert "arith.addf" in loops and "arith.mulf" in loops
        assert "arith.minui" in loops
        assert "tosa." not in loops and "linalg." not in loops
        lowered = optimize(loops, f"aiputile-to-llvm{{index-bitwidth={32 if device else 64}}},reconcile-unrealized-casts")
        assert "aiputile.fused_add_mul" not in lowered
        for prefix in ("scf.", "memref.", "arith.", "func.func", "unrealized_conversion_cast"):
            assert prefix not in lowered, (prefix, lowered)
        ll = run([args.llvm_bin / "mlir-translate", "--mlir-to-llvmir"], lowered)
        run([args.llvm_bin / "opt", "-passes=verify", "-disable-output"], ll)
        assert "alloca " not in ll and "@malloc" not in ll
        assert "fadd fast" not in ll and "fmul fast" not in ll
        if device:
            assert "ptr addrspace(1)" in ll and "addrspacecast" not in ll
        return imported, fused, loops, lowered, ll

    # Positive matching, both Mul operand orders, rank 0 and multidimensional.
    cases = [(shape, tile, reverse) for shape in [(), (1,), (3,), (255,), (256,),
              (257,), (1024,), (2, 3, 5)] for tile in [1, 7, 256, 513]
             for reverse in [False]]
    cases += [((257,), 256, True), ((1, 64, 256, 256), 256, False),
              ((1, 64, 256, 256), 513, False)]
    runner = Path(__file__).with_name("host_runner.c")
    for number, (shape, tile, reverse) in enumerate(cases):
        text = source(shape, reverse=reverse)
        results = stages(text, tile)
        prefix = out / f"case-{number:02}"
        prefix.with_suffix(".input.mlir").write_text(text)
        for suffix, data in zip([".imported.mlir", ".fused.mlir", ".loops.mlir", ".llvm.mlir", ".ll"], results):
            prefix.with_suffix(suffix).write_text(data)
        ll = prefix.with_suffix(".ll")
        executable = prefix.with_suffix(".exe")
        elements = math.prod(shape)
        for level in [0, 3]:
            command = [args.llvm_bin / "clang", ll, runner, f"-DELEMENTS={elements}",
                       f"-O{level}", "-ffp-contract=off", "-o", executable]
            if reverse:
                command.append("-DADD_ON_RHS")
            run(command)
            run([executable])
        # Retain real host assembly for storage/loop inspection.
        run([args.llvm_bin / "clang", ll, "-O3", "-S", "-o", prefix.with_suffix(".s")])
        record(f"numeric shape={shape} tile={tile} reversed={reverse}, O0+O3")

    valid = source((257,))
    # Exercise the function/return importer used by graph-generated AIPUGC IR.
    # Generic spelling avoids inventing a second parser for the source dialect.
    ty = "tensor<257xf32>"
    generic_func = f'''module {{
      "aipugc.func"() ({{
      ^bb0(%x: {ty}, %b: {ty}, %a: {ty}):
        %t = "aipugc.add"(%x, %b) : ({ty}, {ty}) -> {ty}
        %y = "aipugc.mul"(%t, %a) : ({ty}, {ty}) -> {ty}
        "aipugc.return"(%y) : ({ty}) -> ()
      }}) {{sym_name = "kernel", function_type = ({ty}, {ty}, {ty}) -> {ty}}} : () -> ()
    }}'''
    stages(generic_func, 256)
    record("import generic aipugc.func and aipugc.return")
    for size in [0, -1, 2147483648]:
        optimize(valid, f"{imported_pass},aiputile-fuse-add-mul{{tile-size={size}}}", "tile-size must")
        record(f"reject tile-size={size}")
    for dtype in ["f16", "f64", "i32"]:
        optimize(source((257,), dtype), imported_pass, "static, unencoded, dense FP32")
        record(f"reject {dtype}")
    optimize(source(("?",)), imported_pass, "static, unencoded, dense FP32")
    record("reject dynamic shape")
    for shape in [(0,), (65536, 65536)]:
        optimize(source(shape), imported_pass, "nonempty shapes")
        record(f"reject shape={shape}")
    mismatch = valid.replace("%b: tensor<257xf32>", "%b: tensor<1xf32>")
    mismatch = mismatch.replace('"aipugc.add"(%x, %b) : (tensor<257xf32>, tensor<257xf32>)',
                                '"aipugc.add"(%x, %b) : (tensor<257xf32>, tensor<1xf32>)')
    optimize(mismatch, imported_pass, "identical input/output shapes")
    record("reject broadcast")
    multi = valid.replace(
        ') -> tensor<257xf32> {', ') -> (tensor<257xf32>, tensor<257xf32>) {')
    multi = multi.replace("    return %y : tensor<257xf32>",
        '    %extra = "aipugc.mul"(%t, %b) : (tensor<257xf32>, tensor<257xf32>) -> tensor<257xf32>\n'
        '    return %y, %extra : tensor<257xf32>, tensor<257xf32>')
    optimize(multi, f"{imported_pass},aiputile-fuse-add-mul", "single-consumer Add result")
    record("reject shared Add producer")
    dead_consumer = valid.replace("    return %y", '    %extra = "aipugc.mul"(%t, %b) : (tensor<257xf32>, tensor<257xf32>) -> tensor<257xf32>\n    return %y')
    optimize(dead_consumer, f"{imported_pass},aiputile-fuse-add-mul", "single-consumer Add result")
    record("reject shared producer before implicit DCE")
    noadd = valid.replace('"aipugc.add"', '"aipugc.mul"')
    optimize(noadd, f"{imported_pass},aiputile-fuse-add-mul", "single-consumer Add result")
    record("reject absent Add producer")
    attrs = valid.replace('"aipugc.add"(%x, %b)', '"aipugc.add"(%x, %b) {with_activation = "RELU"}')
    optimize(attrs, imported_pass, "does not support source parameter")
    record("reject fused activation")
    unsupported = valid.replace('"aipugc.add"', '"aipugc.sub"')
    optimize(unsupported, imported_pass, "unsupported source IR remains")
    record("reject unsupported source op")
    fused = stages(valid, 256)[1]
    encoded = valid.replace('tensor<257xf32>', 'tensor<257xf32, "blocked">')
    optimize(encoded, imported_pass, "static, unencoded, dense FP32")
    record("reject encoded layout")
    effect = fused.replace("    return %", '    "test.side_effect"() : () -> ()\n    return %')
    optimize(effect, "aiputile-lower-tiles", "one fused operation and one return")
    record("reject extra side effects")
    constant = fused.replace("    %0 =", '    %c = arith.constant dense<1.0> : tensor<257xf32>\n    %0 =')
    constant = constant.replace("aiputile.fused_add_mul %arg0", "aiputile.fused_add_mul %c")
    optimize(constant, "aiputile-lower-tiles", "one fused operation and one return")
    record("reject constant in runtime-input-only kernel")
    caller = '  func.func @caller(%x: tensor<257xf32>, %b: tensor<257xf32>, %a: tensor<257xf32>) -> tensor<257xf32> {\n' \
        '    %v = call @kernel(%x, %b, %a) : (tensor<257xf32>, tensor<257xf32>, tensor<257xf32>) -> tensor<257xf32>\n'
    caller += '    return %v : tensor<257xf32>\n  }\n'
    calls = fused.rsplit("}", 1)[0] + caller + "}\n"
    optimize(calls, "aiputile-lower-tiles", "no in-module callers")
    record("reject in-module callers when changing entry ABI")
    optimize(fused.replace("tile_size = 256", "tile_size = 0"), "", "tile_size must")
    record("verifier rejects invalid fused op")
    optimize(fused, "aiputile-lower-tiles{device-address-space=3}", "supported ABI address spaces")
    record("reject unknown memory space")
    loops = stages(valid, 256)[2]
    optimize(loops, "aiputile-to-llvm{index-bitwidth=16}", "index-bitwidth must")
    record("reject unsupported index width")
    device = stages(valid, 256, device=True)
    (out / "device-abi.ll").write_text(device[-1])
    record("AS1 pointer ABI and 32-bit indices: LLVM verifier, not NPU compilation")

    report = {"passed": len(tests), "tests": tests,
              "npu_compilation": "unrun: private backend unavailable",
              "npu_execution": "unrun: toolchain/runtime/hardware unavailable",
              "original_baseline": "unrun: private graph-builder dependencies unavailable",
              "physical_sram_dma": "unimplemented: interface not confirmed",
              "device_performance": "unrun: no device"}
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(tests)} checks passed; {len(cases) * 2} native kernel executions; NPU checks UNRUN", flush=True)
    if args.artifacts:
        print(f"Artifacts: {out}")


if __name__ == "__main__":
    main()
