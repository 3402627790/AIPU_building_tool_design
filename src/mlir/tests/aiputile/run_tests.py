#!/usr/bin/env python3
"""Validate the Tile-first pipeline with real MLIR/LLVM and CPU execution.

The portable driver does not link the private graph builder. Only source AIPUGC
operations use allow-unregistered-dialect; every generated Tile operation/type
is registered and verified. Neither CPU control is the original AIPUGC baseline.
"""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile


def run(command, text=None, expected=None):
    result = subprocess.run([str(x) for x in command], input=text, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=120)
    if expected is not None:
        messages = (expected,) if isinstance(expected, str) else expected
        if result.returncode == 0 or not any(x in result.stderr for x in messages):
            raise RuntimeError(f"Expected diagnostic {messages!r}: {command}\n{result.stderr}")
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


def loop_module(n=257, producer_size=256, consumer_size=256, shared=False,
                capture=False, extra_effect=False):
    """Build valid generic Tile IR, allowing individual contract violations."""
    full = f"!aiputile.tile<{n}xf32>"

    def tile_loop(result, inputs, size, operation, capture=False, effect=False):
        local = f"!aiputile.tile<{min(size, n)}xf32>"
        lhs = "%x" if capture else "%l"
        effect_op = '        "test.side_effect"() : () -> ()\n' if effect else ""
        return f'''    %{result} = "aiputile.tile_loop"({inputs}) ({{
      ^bb0(%l: {local}, %r: {local}, %offset: index, %valid: index):
{effect_op}        %value = aiputile.{operation} {lhs}, %r : ({local}, {local}) -> {local}
        "aiputile.yield"(%value) : ({local}) -> ()
    }}) {{tile_size = {size} : i64}} : ({full}, {full}) -> {full}
'''

    result_type = f"({full}, {full})" if shared else full
    text = f"module {{\n  func.func @kernel(%x: {full}, %b: {full}, %a: {full}) -> {result_type} {{\n"
    text += tile_loop("t", "%x, %b", producer_size, "add", capture, extra_effect)
    text += tile_loop("y", "%t, %a", consumer_size, "mul")
    if shared:
        text += tile_loop("extra", "%t, %b", consumer_size, "mul")
        text += f"    return %y, %extra : {full}, {full}\n"
    else:
        text += f"    return %y : {full}\n"
    return text + "  }\n}\n"


def tile_loops(text):
    return len(re.findall(r'(?:"aiputile\.tile_loop"|aiputile\.tile_loop\b)', text))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    temporary = tempfile.TemporaryDirectory(prefix="aiputile-tests-")
    out = args.artifacts or Path(temporary.name)
    out.mkdir(parents=True, exist_ok=True)
    tests = []
    native_executions = 0
    imported_pass = "lower-aipugc-to-aiputile"

    def optimize(text, pipeline, expected=None):
        return run([args.opt, "--allow-unregistered-dialect",
                    f"--pass-pipeline=builtin.module({pipeline})"], text, expected)

    def record(name):
        tests.append(name)
        print(f"PASS {name}", flush=True)

    def stages(text, size):
        imported = optimize(text, imported_pass)
        assert "aipugc." not in imported and "!aiputile.tile<" in imported
        assert imported.count("aiputile.add") == 1 and imported.count("aiputile.mul") == 1
        assert imported.count("aiputile.from_tensor") == 3, imported
        assert imported.count("aiputile.to_tensor") == 1, imported
        for line in imported.splitlines():
            if "aiputile.add " in line or "aiputile.mul " in line:
                assert "tensor<" not in line, line
        tiled = optimize(imported, f"aiputile-tile{{tile-size={size}}}")
        assert tile_loops(tiled) == 2, tiled
        assert tiled.count("aiputile.add") == 1 and tiled.count("aiputile.mul") == 1
        assert "aiputile.yield" in tiled and "index" in tiled
        fused = optimize(tiled, "aiputile-fuse-tiles")
        assert tile_loops(fused) == 1, fused
        assert fused.count("aiputile.add") == 1 and fused.count("aiputile.mul") == 1
        assert "aiputile.fused_add_mul" not in fused
        # No source dialect escape hatch is needed for the generated registered IR.
        for ir in [imported, tiled, fused]:
            run([args.opt], ir)
        return imported, tiled, fused

    def lower(tile_ir, fused, device=False):
        loops = optimize(tile_ir, f"aiputile-lower-tiles{{device-address-space={int(device)}}}")
        assert loops.count("scf.for") == (2 if fused else 4), loops
        assert loops.count("memref.load") == (3 if fused else 4), loops
        assert loops.count("memref.store") == (1 if fused else 2), loops
        assert ("memref.alloc" in loops) == (not fused), loops
        assert ("memref.dealloc" in loops) == (not fused), loops
        for spelling in ["!aiputile.tile", "aiputile.tile_loop", "aiputile.from_tensor", "aiputile.to_tensor"]:
            assert spelling not in loops, (spelling, loops)
        assert "arith.addf" in loops and "arith.mulf" in loops and "arith.minui" in loops
        assert "tosa." not in loops and "linalg." not in loops
        lowered = optimize(loops, f"aiputile-to-llvm{{index-bitwidth={32 if device else 64}}},reconcile-unrealized-casts")
        for spelling in ["!aiputile.tile", "scf.", "memref.", "arith.", "func.func", "unrealized_conversion_cast"]:
            assert spelling not in lowered, (spelling, lowered)
        ll = run([args.llvm_bin / "mlir-translate", "--mlir-to-llvmir"], lowered)
        run([args.llvm_bin / "opt", "-passes=verify", "-disable-output"], ll)
        assert ("@malloc" in ll) == (not fused), ll
        assert "fadd fast" not in ll and "fmul fast" not in ll
        if device:
            assert "ptr addrspace(1)" in ll and "addrspacecast" not in ll
        return loops, lowered, ll

    # Independent per-op tiling and subsequent fusion are observed before lowering.
    cases = [(shape, size, False) for shape in [(), (1,), (3,), (255,), (256,),
              (257,), (1024,), (2, 3, 5)] for size in [1, 7, 256, 513]]
    cases += [((257,), 256, True), ((1, 64, 256, 256), 256, False),
              ((1, 64, 256, 256), 513, False)]
    runner = Path(__file__).with_name("host_runner.c")
    for number, (shape, size, reverse) in enumerate(cases):
        text = source(shape, reverse=reverse)
        imported, tiled, fused = stages(text, size)
        prefix = out / f"case-{number:02}"
        for suffix, ir in [(".input.mlir", text), (".imported.mlir", imported),
                           (".tiled.mlir", tiled), (".fused.mlir", fused)]:
            prefix.with_suffix(suffix).write_text(ir)
        for route, ir, is_fused in [("unfused", tiled, False), ("fused", fused, True)]:
            outputs = lower(ir, is_fused)
            route_prefix = out / f"case-{number:02}-{route}"
            for suffix, data in zip([".loops.mlir", ".llvm.mlir", ".ll"], outputs):
                route_prefix.with_suffix(suffix).write_text(data)
            ll = route_prefix.with_suffix(".ll")
            for level in [0, 3]:
                executable = route_prefix.with_suffix(f".O{level}.exe")
                command = [args.llvm_bin / "clang", ll, runner,
                           f"-DELEMENTS={math.prod(shape)}", f"-O{level}",
                           "-ffp-contract=off", "-o", executable]
                if reverse:
                    command.append("-DADD_ON_RHS")
                run(command)
                run([executable])
                native_executions += 1
            run([args.llvm_bin / "clang", ll, "-O3", "-S", "-o", route_prefix.with_suffix(".s")])
        record(f"fused/unfused numeric shape={shape} tile={size} reversed={reverse}, O0+O3")

    valid = source((257,))
    # Two independent single-use producers exercise repeated region merging
    # and input deduplication, not just the original Add -> Mul special case.
    branched = valid.replace(
        '    %y = "aipugc.mul"(%t, %a) : (tensor<257xf32>, tensor<257xf32>) -> tensor<257xf32>',
        '    %q = "aipugc.mul"(%b, %a) : (tensor<257xf32>, tensor<257xf32>) -> tensor<257xf32>\n'
        '    %y = "aipugc.mul"(%t, %q) : (tensor<257xf32>, tensor<257xf32>) -> tensor<257xf32>')
    branch_imported = optimize(branched, imported_pass)
    assert branch_imported.count("aiputile.from_tensor") == 3
    assert branch_imported.count("aiputile.to_tensor") == 1
    branch_tiled = optimize(branch_imported, "aiputile-tile{tile-size=256}")
    assert tile_loops(branch_tiled) == 3
    branch_fused = optimize(branch_tiled, "aiputile-fuse-tiles")
    assert tile_loops(branch_fused) == 1
    assert branch_fused.count("aiputile.add") == 1
    assert branch_fused.count("aiputile.mul") == 2
    (out / "branched.tiled.mlir").write_text(branch_tiled)
    (out / "branched.fused.mlir").write_text(branch_fused)
    record("merge two independent producer regions and deduplicate shared runtime inputs")
    branch_runner = out / "branched-runner.c"
    branch_runner.write_text(r'''#include <stdio.h>
extern void kernel(const float *, const float *, const float *, float *);
int main(void) {
  float x[257], b[257], a[257], y[257];
  for (int i = 0; i < 257; ++i) {
    x[i] = (i - 140) / 7.0f;
    b[i] = (i % 31 - 15) / 13.0f;
    a[i] = (i % 43 - 21) / 19.0f;
    y[i] = -99999.0f;
  }
  kernel(x, b, a, y);
  for (int i = 0; i < 257; ++i) {
    volatile float p = x[i] + b[i];
    volatile float q = b[i] * a[i];
    float reference = p * q;
    if (y[i] != reference) {
      fprintf(stderr, "branched mismatch i=%d got=%g expected=%g\n", i, y[i], reference);
      return 1;
    }
  }
  puts("PASS: branched producer fusion, 257 runtime FP32 elements");
  return 0;
}
''')
    for route, ir, is_fused in [("unfused", branch_tiled, False), ("fused", branch_fused, True)]:
        branch_loops = optimize(ir, "aiputile-lower-tiles")
        assert branch_loops.count("scf.for") == (2 if is_fused else 6)
        assert branch_loops.count("memref.alloc") == (0 if is_fused else 2)
        branch_llvm = optimize(branch_loops, "aiputile-to-llvm,reconcile-unrealized-casts")
        branch_ll = run([args.llvm_bin / "mlir-translate", "--mlir-to-llvmir"], branch_llvm)
        run([args.llvm_bin / "opt", "-passes=verify", "-disable-output"], branch_ll)
        ll_path = out / f"branched-{route}.ll"
        ll_path.write_text(branch_ll)
        for level in [0, 3]:
            executable = out / f"branched-{route}.O{level}.exe"
            run([args.llvm_bin / "clang", ll_path, branch_runner, f"-O{level}",
                 "-ffp-contract=off", "-o", executable])
            run([executable])
            native_executions += 1
    record("branched producer graph: fused/unfused LLVM verification and native O0+O3 execution")
    returned_producer = valid.replace(
        ') -> tensor<257xf32> {', ') -> (tensor<257xf32>, tensor<257xf32>) {')
    returned_producer = returned_producer.replace(
        '    return %y : tensor<257xf32>',
        '    return %t, %y : tensor<257xf32>, tensor<257xf32>')
    returned_tiled = optimize(returned_producer,
        "lower-aipugc-to-aiputile,aiputile-tile{tile-size=256}")
    assert returned_tiled.count("aiputile.to_tensor") == 2
    optimize(returned_tiled, "aiputile-fuse-tiles", ("single", "consumer", "shared"))
    record("reject producer shared by a consumer and an external tensor result")
    imported, tiled, fused = stages(valid, 256)
    assert "!aiputile.tile<256xf32>" in tiled
    record("257 elements use 256-capacity tiles with offset/valid-length region arguments")
    large = stages(source((1, 64, 256, 256)), 256)[1]
    assert tile_loops(large) == tile_loops(tiled) == 2
    assert len(large) < 2 * len(tiled)
    record("IR size remains bounded for 16,384 logical tiles")

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
        optimize(imported, f"aiputile-tile{{tile-size={size}}}", "tile-size")
        record(f"reject tile-size={size}")
    for dtype in ["f16", "f64", "i32"]:
        optimize(source((257,), dtype), imported_pass, "FP32")
        record(f"reject {dtype}")
    optimize(source(("?",)), imported_pass, ("static", "dynamic"))
    record("reject dynamic shape")
    for shape in [(0,), (65536, 65536)]:
        optimize(source(shape), imported_pass, ("nonempty", "INT32_MAX", "overflow"))
        record(f"reject shape={shape}")
    mismatch = valid.replace("%b: tensor<257xf32>", "%b: tensor<1xf32>")
    mismatch = mismatch.replace('"aipugc.add"(%x, %b) : (tensor<257xf32>, tensor<257xf32>)',
                                '"aipugc.add"(%x, %b) : (tensor<257xf32>, tensor<1xf32>)')
    optimize(mismatch, imported_pass, ("identical", "match", "shape"))
    record("reject broadcast")
    attrs = valid.replace('"aipugc.add"(%x, %b)', '"aipugc.add"(%x, %b) {with_activation = "RELU"}')
    optimize(attrs, imported_pass, "source parameter")
    record("reject fused activation")
    optimize(valid.replace('"aipugc.add"', '"aipugc.sub"'), imported_pass,
             ("unsupported source", "failed to legalize"))
    record("reject unsupported source op")
    encoded = valid.replace('tensor<257xf32>', 'tensor<257xf32, "blocked">')
    optimize(encoded, imported_pass, ("unencoded", "encoding", "dense"))
    record("reject encoded layout")

    # Well-typed but incompatible producer/consumer tile partitions must not fuse.
    optimize(loop_module(producer_size=128, consumer_size=256), "aiputile-fuse-tiles",
             ("domain", "partition", "tile_size", "tile size"))
    record("reject fusion of different tile domains")
    optimize(loop_module(shared=True), "aiputile-fuse-tiles",
             ("single", "consumer", "shared"))
    record("reject shared tile-loop producer")
    run([args.opt], loop_module())
    record("parse/print registered tile-loop IR")
    invalid_partition = loop_module().replace("tile_size = 256", "tile_size = 0", 1)
    run([args.opt], invalid_partition, "tile_size must")
    record("verifier rejects invalid tile-loop partition")
    invalid_offset = loop_module().replace("%offset: index", "%offset: f32", 1)
    run([args.opt], invalid_offset, ("offset", "index"))
    record("verifier requires index-valued tile offsets")
    tensor_arithmetic = valid.replace('"aipugc.add"(%x, %b)', '"aiputile.add"(%x, %b)')
    run([args.opt, "--allow-unregistered-dialect"], tensor_arithmetic,
        ("Tile", "tile", "operand"))
    record("registered tile arithmetic rejects Tensor operands")
    run([args.opt, "--allow-unregistered-dialect"], loop_module(extra_effect=True),
        ("unsupported", "side effect", "body", "only"))
    record("reject side effects inside a tile computation")
    run([args.opt], loop_module(n=256, capture=True),
        ("outside", "isolated", "capture"))
    record("reject tile-loop region captures")
    wrong_body = loop_module().replace("!aiputile.tile<256xf32>", "!aiputile.tile<255xf32>")
    run([args.opt], wrong_body, ("body", "argument", "capacity"))
    record("reject tile body capacity inconsistent with partition")
    for bad_type in ["!aiputile.tile<0xf32>", "!aiputile.tile<256xf16>"]:
        invalid = loop_module().replace("!aiputile.tile<256xf32>", bad_type)
        run([args.opt], invalid, ("capacity", "positive", "FP32", "f32"))
        record(f"reject invalid tile type {bad_type}")
    optimize(imported, "aiputile-fuse-tiles", ("tile", "tiling"))
    record("fusion requires the explicit tiling stage")
    optimize(tiled, "aiputile-lower-tiles{device-address-space=3}", "address space")
    record("reject unknown memory space")
    optimize(tiled, "aiputile-lower-tiles{device-address-space=1}",
             ("fuse", "temporary", "intermediate", "allocation"))
    record("reject unfused temporary device allocation")
    loops = lower(fused, True)[0]
    optimize(loops, "aiputile-to-llvm{index-bitwidth=16}", "index-bitwidth")
    record("reject unsupported index width")
    device = lower(fused, True, device=True)
    (out / "device-abi.ll").write_text(device[-1])
    record("fused AS1/32-bit ABI passes LLVM verifier; device codegen remains unrun")

    report = {"passed": len(tests), "native_kernel_executions": native_executions,
              "tests": tests, "pipeline": "Tile types -> per-op tile_loop -> tile-loop fusion -> LLVM",
              "npu_compilation": "unrun: private backend unavailable",
              "npu_execution": "unrun: toolchain/runtime/hardware unavailable",
              "original_baseline": "unrun: private graph-builder dependencies unavailable",
              "physical_sram_dma": "unimplemented: interface not confirmed",
              "device_performance": "unrun: no device"}
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(tests)} checks passed; {native_executions} native kernel executions; NPU checks UNRUN", flush=True)
    if args.artifacts:
        print(f"Artifacts: {out}")


if __name__ == "__main__":
    main()
