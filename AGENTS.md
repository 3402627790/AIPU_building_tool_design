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
| `src/gsim/` | Graph simplify |
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

## Architecture Overview (架构总览)

### 项目定位

把前端解析好的**神经网络计算图（Graph）**编译成能在 AIPU 芯片上运行的**二进制可执行文件**。整体工具链是 Python 前端（解析/量化）+ C++ 后端（编译）的组合：

```
3rd-party model → Parser(解析) → Quantization(量化) → Builder(编译) → AIPU ELF/.bin
                                                              (本仓库 = 这里)
```

### 两条编译路径（核心架构）

内部有**两条完全独立的编译流水线**，对应不同硬件目标：

| 维度 | MC 路径 | MQ 路径 |
|------|---------|---------|
| 目标 | Multi-Core, ZY ISA (X1/X2/X3/Z1-Z5) | Multi-Queue, Compass |
| 入口文件 | `src/builder/builder.cpp` + `mcbuilder.cpp` | `src/builder/mqbuilder.cpp` |
| 优化入口 | `optimizer.cpp::optimize()` (完整 lowering) | `optimizer.cpp::optimize_mq()` (仅 GOP 插件) |
| Lowering | 50+ pass, tiling, 图分割 | 无 tiling |
| 算子实现 | 预编译 kernel 库 + 手写 ZY 汇编 (DMA/Cache/中断) | 预编译 kernel `.o` 链接 (cpscc) |
| 调度器 | `scheduler.cpp` + `mc_scheduler*` + `load_balancer.cpp` | `soms_scheduler.cpp` (单核) |
| 内存分配 | AMAT / MMU allocator | SOMS stream slot |
| 代码生成 | `text_generator.cpp` / `ro_generator.cpp` / `ds_code_generator*` | `code_generator.cpp` + `assembler.cpp` + `object_generator.cpp` |
| 链接器 | `linker/linker_v1.cpp`(ZY汇编) / `linker_v2.cpp`(ELF) | `linker/linker_cps.cpp` |
| 输出 | AIPU ELF / .bin | Compass ELF |

### MC 路径编译流水线（optimizer.cpp）

```
init_symbol_shape → infer_symbol_shape → lower_before_all → lower_before_tiling
→ tiling(切片) → divide_graph(图分割) → lower_before_optimize → unfold_graph
→ lower_after_unfold → lower_before_plugin → apply_plugins(应用插件)
→ 调度 → 内存分配 → 代码生成 → 链接 → 输出 bin/ELF
```

### 图优化 Pass（两个目录）

- `src/transform/`：**通用图简化/融合**，100+ 个 pass（`fuse_*.cpp`、`eliminate_*.cpp`、`lower_*.cpp`）
- `src/builder/lowering/`：**MC 专用 lowering**，50+ 个 pass（`tiling.cpp`、`datalayout_schedule.cpp`、`divide_graph.cpp`、硬件约束适配 `hwa_*_adaptor.cpp`）

### 插件机制

`doc/GBuilder/cpp_builder_plugin_design.md` 定义了完整插件接口（`BuilderOpPlugin`）：
`get_score()` → 排序 → `get_graph_pattern()` → 子图匹配 → `check_params()` → `setup()` → `generate_params()` → `generate_code_name()`

- **外部插件**：客户自定义算子（`plugin/` 目录，.so 动态加载）
- **内置插件**：opt_builtin（conv/eltwise/lut/quantize/switch 等）
- **GOP 插件**（`gop_plugin.h`）：MQ 路径专用，`apply_gopplugins()` 应用

### MQ 路径完整流程（src/builder/mqbuilder.md 有逐步讲解）

```
初始化全局 Tensor(stack/print/param/stream0-3)
→ optimizer.optimize_mq() → PerfEstimator → LoadBalancer 分层
→ KernelAdapter(映射算子到 kernel 库) → SOMSScheduler(内存分配)
→ CodeGenerator(链接 kernel .o → Compass ELF)
→ Assembler(生成 task_t 描述符/stream) → ObjectGenerator(写入最终 ELF)
```

### 辅助模块

| 目录 | 作用 |
|------|------|
| `src/mlir/` | AIPU MLIR dialect（aipugc），含 `Target/Compass/TranslateToCompass.cpp`，仅 x86_64 构建 |
| `src/vm/` | 虚拟机（asmtranscode/tec），仅 x86_64 |
| `src/engine/` | 执行引擎（本地运行验证） |
| `src/gsim/` | 图简化 |
| `src/checker/` | IR 检查 |
| `src/quantization_transform/` | 量化（quantizer/observer/qinfo） |
| `src/dtypes/` | 数据类型与 LUT 生成 |
| `src/tongue/` | 符号 shape 表达式引擎 |
| `src/statistic/` | 性能模型（cost model，通过 ld --format=binary 嵌入） |
| `src/asm/` | ZY ISA 汇编（asm.h/asmopcode.h） |

### 对外接口与工具

- **tools/**：`aipugb`(编译)、`aipurun`(运行)、`aipudumper`、`aipuchecker`、`aipuestimator`、`aipuperfinfo`、`aipubinmerge`、`aipubinutils`
- **python/**：pybind11 绑定 → `_C` 扩展，Python 侧 AIPUBuilder 包
- **modules/**：`btrt`(板端运行时，加载 ELF 到设备)、`runtime`、`toolchain`(外部工具链封装 cpscc/cpshostcc/aipuocc)
- **lib/**：预编译 kernel 库（`lib/opt/x3/*.cl` OpenCL kernel 等）
- **include/builder/ibuilder.h**：抽象接口 `IBuilder{build/get_graph/get_elf/get_header}`

### 建议的阅读路径

1. `src/builder/README.md` — 两条路径的全景图
2. `src/builder/mqbuilder.md` — MQ 路径逐步讲解（代码简单清晰，新人首选）
3. `doc/AIPUBuilder_CPP_design.md` — 设计动机与插件机制
4. `src/builder/optimizer.cpp` — MC 路径优化流水线入口
5. `include/builder/ibuilder.h` + `include/builder/builder.h` — 顶层接口
