# AGENTS.md

## What This Is

AIPU Building Tool (aipubuildtool) — AIPU hardware compiler backend. Compiles neural network computation graphs to AIPU ELF/.bin binaries. Has two paths: MC (multi-core, ZY ISA) and MQ (multi-queue, Compass).

## Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release   # or Debug
make -j$(nproc)
```

Key CMake options:
- `-DCMAKE_BUILD_TYPE=Release|Debug` — Release enables -O3 and strips; Debug enables -O0
- `-DWITH_TEST=ON` — builds test binaries
- `-DCODE_COVERAGE=ON` — enables gcov instrumentation
- `-DSTATIC_STDLIB=ON` — static link libgcc/libstdc++

Requires: C++17, GCC, Linux x86_64 (MLIR only builds on x86_64). Android ARM cross-compilation is supported but skips MLIR and VM modules.

## Test

```bash
cd build && make -j$(nproc)
./test/testcase
```

Single test executable `testcase` linked against `libaipubuildtool.so`. No test framework — uses plain C++ assertions.

## Formatting

`.clang-format` at root: LLVM style, 4-space indent, Allman braces, no tabs. Run `clang-format -i` on changed files.

## Code Coverage

```bash
# 1. Build with: cmake .. -DCODE_COVERAGE=ON
# 2. Run testcase
# 3. From repo root:
./code_coverage.sh "src/builder,src/core"  # optional: specific dirs
```

Output: `code_coverage_report/coverage.html`. Requires `gcovr`.

## Environment Setup

For MLIR/AIPU dialect development:

```bash
source env_setup.sh   # tcsh
# or source env_setup.csh
```

Sets `PATH`, `LD_LIBRARY_PATH`, `PYTHONPATH` for `3rdparty/llvm_mlir_sdk/`.

## Architecture Quick Facts

- **MC path** (builder.cpp, mcbuilder.cpp): full lowering pipeline, 50+ passes, tiling, multi-core scheduling, ZY ISA assembly. Outputs .bin/ELF.
- **MQ path** (mqbuilder.cpp): simpler, GOP plugins only, no tiling. Links pre-compiled kernel `.o` files. Outputs Compass ELF.
- `optimizer.cpp` is the entry for both paths; MC calls `optimize()` (full lowering), MQ calls `optimize_mq()` (GOP plugins only).
- `lowering/lowering.cpp` controls MC pass order.
- `src/mlir/` contains the AIPU MLIR dialect (aipugc). Only builds on x86_64.
- `src/vm/` (virtual machine) only builds on x86_64, skipped on Android.

## Key Source Layout

| Dir | Purpose |
|-----|---------|
| `src/builder/` | Compiler backend core (MC+MQ paths) |
| `src/core/` | Graph/Node/Tensor data structures |
| `src/mlir/` | AIPU MLIR dialect |
| `src/transform/` | Graph optimization passes |
| `src/engine/` | Execution engine |
| `src/gsim/` | Graph simulator |
| `test/` | Tests (flat `testcase` binary) |
| `tools/` | CLI tools (aipugb, aipurun, aipudumper) |
| `plugin/` | User-facing plugins |
| `python/` | Python bindings |
| `lib/` | Layer libraries (pre-compiled kernels) |

## Conventions

- C++17, no exceptions (`-fno-exceptions`)
- Strict warnings: `-Wall -Werror` (plus several `-Werror=...` flags)
- Shared library output: `libaipubuildtool.so`
- Embed binary resources via `ld --format=binary` + objcopy (see `embedfiles` macro in src/CMakeLists.txt)
