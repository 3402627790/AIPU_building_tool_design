# 项目长期记忆：AIPU_building_tool_design

AIPU 神经网络编译器后端（aipubuildtool）。双编译路径，编译器命令与路径严格对应：

- **MC 路径（multi-core，ZY ISA）**：完整 lowering 流水线。`src/builder/emitter/code_emitter.cpp` 用 **aipuocc**（基于 LLVM/Clang 后端，`-target aipu -O2 -mcpu=<target> -c`）编译设备代码；动态 shape 由 `ds_code_generator(_cx).cpp`（ZY 汇编）生成。输出 AIPU ELF/.bin。linker_v2 也用 aipuocc。
- **MQ 路径（multi-queue，Compass）**：只走 GOP 插件、无 tiling。`src/builder/code_generator.cpp`（被 mqbuilder.cpp 实例化）用 **cpscc**（Compass toolkit 的 C 编译器，`-dc -O2 --cps-path`）编译 `.cps` 设备代码，cpshostcc 编译 host 侧 `.cpp`；LinkerCps 链接成 Compass ELF。

工具链命令 ↔ 路径对应（已代码核实）：
- `aipuocc` → MC 设备编译器（LLVM/Clang 后端定制；默认 `/project/ai/scratch01/toolchain/llvm/binary/bin/aipuocc`）
- `cpscc`   → MQ 设备编译器（Compass toolkit 提供；默认 `/project/ai/scratch01/compass-toolkit/compiler/latest/bin/cpscc`）
- `cpshostcc` → host 侧编译器（默认 g++，链接 libaipubtrt）
- 兄弟：`aipuas`/`aipuld`（AIPU 汇编/链接）、`cpsld`（Compass 链接）

关键目录：src/builder/{mcbuilder,mqbuilder,code_generator,code_emitter,linker}, src/mlir（aipugc dialect，仅 x86_64）, python/src/AIPUBuilder。
