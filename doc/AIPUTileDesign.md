# AIPUTile：Add + Mul 逻辑 Tile Fusion

## 当前交付边界

已实现并实际编译、执行的是 **逻辑 Tile Loop Fusion 的可移植部分**。新路径不使用
TOSA、Linalg、它们的 tiling/fusion pass，也不分配完整中间 Tensor。

当前机器没有用户私有的 Zhouyi 后端、AIPU 工具链、kernel 库或可用 NPU 运行条件。
因此 **P0 的 Zhouyi 编译和原始 baseline 回归尚未完成，P1 物理 SRAM/DMA、P2 设备性能尚未完成**。
不能将下述 CPU 验证解释为 NPU 验证。

## 工程检查结果

仓库：`/workspace/AIPU_building_tool_design`。

| 内容 | 真实位置 / 检查结果 |
| --- | --- |
| 原主入口、pipeline | `src/mlir/tool/aipugc/main.cpp` |
| 原 ODS / 注册 | `src/mlir/include/aipugc/AipuOps.td`、`src/mlir/mlir/aipugc/Dialect.cpp` |
| 原 TOSA / Linalg / LLVM 转换 | `LoweringAipugcToTOSA.cpp`、`LowerTosaToLinalg.cpp`、`LowerAffineToLLVM.cpp` |
| 原输出参数 ABI | `src/mlir/mlir/aipugc/KernelABIPass.cpp` |
| LLVM 导出 / TargetMachine | `src/mlir/tool/aipugc/utils.cpp` |
| Zhouyi 汇编 / 对象输出 | `src/mlir/mlir/aipugc/LLIRTool.cpp`、`tool/aipugc/llir_main.cpp` |
| 设备 kernel 示例 | `src/mlir/isNanOp/kernel/isnan_kernel.cps` |
| 原测试 | `src/mlir/tests/test_ops.py`，依赖 graph builder / aipuir |
| 新增隔离 SDK | `/workspace/.aipu-onboarding/llvm19`，**官方 LLVM/MLIR/Clang 19.1.6** |
| 通用工具 | CMake 3.31.6、Ninja 1.11.1、GCC 14.2.0 |

隔离 SDK 来自 LLVM 官方发布的 `LLVM-19.1.6-Linux-X64.tar.xz`，使用官方 release
attestation 中的 SHA256 校验：

```
d55dcbb309de7ade4e3073ec3ac3fac4d3ff236d54df3c4de04464fe68bec531
```

没有替换现有工具链，没有将上游 LLVM 冒充 Zhouyi SDK。
历史 `/project/ai/...` 路径在当前机器不存在。

完整工程还有这些既有阻塞：

- `3rdparty/llvm_mlir_sdk`、其他 third-party 依赖、内部 AIPU SDK / 库缺失。
- `modules/compassLLM/CMakeLists.txt`、`src/statistic/cost_model_x3p.cbin` 不在当前 checkout。
- `include/core/compute_tensor.h` 的 HEAD 版本有重复头部：两次同名 `#ifndef` 只有一个
  `#endif`，而且第二段模板后直接进入残缺的 `Slice` 实现。主驱动的真实语法检查首先报
  `unterminated conditional directive`。该文件没有在本次修改中改动，不能凭空恢复缺失实现。
- 原工具使用私有 `llvm::CallingConv::Zhouyi_Kernel` 和 `LLVMInitializeZhouyi*`，上游 SDK
  不提供这些接口。

`lib/opt/x3/eltwise.cl` 可见旧 X3 示例使用 `__lsram`、`__dma` 和 32 KiB 常量，
但所需 `aipu.h` / `dma.h` 及工具链缺失，不能据此认定当前 Compass 目标的容量、地址空间或
DMA ABI。`isnan_kernel.cps` 还依赖未提供的 toolkit 头文件。设备数据类型与指令支持未获实际验证。

## 文件与核心实现

| 文件 | 用途 |
| --- | --- |
| `src/mlir/include/aiputile/AIPUTile.td` | dialect 名称、C++ namespace |
| `src/mlir/include/aiputile/AIPUTileOps.td` | Add、Mul、FusedAddMul 的 ODS |
| `src/mlir/include/aiputile/AIPUTileDialect.h`、`AIPUTileOps.h` | 生成声明和验证接口 |
| `src/mlir/include/aiputile/Passes.h` | Pass factories |
| `src/mlir/include/aiputile/CMakeLists.txt` | TableGen |
| `src/mlir/aiputile/AIPUTileDialect.cpp` | 注册真实 dialect 和全部 operations |
| `src/mlir/aiputile/AIPUTileOps.cpp` | `verifyTensorTypes` 和 op verifiers |
| `src/mlir/aiputile/Passes.cpp` | source conversion、fusion、tile loops、LLVM conversion |
| `src/mlir/aiputile/aiputile-opt.cpp` | 无私有后端依赖的 MLIR 测试驱动 |
| `src/mlir/aiputile/CMakeLists.txt` | 共享静态库、独立构建、CTest |
| `src/mlir/tests/aiputile/add_mul.mlir` | 257 元素 tail 示例 |
| `src/mlir/tests/aiputile/host_runner.c` | 运行时输入、CPU reference、非融合对照和越界哨兵 |
| `src/mlir/tests/aiputile/run_tests.py` | IR、失败诊断、LLVM verifier、真实 host codegen / 执行 |
| `src/mlir/CMakeLists.txt`、`tool/aipugc/CMakeLists.txt` | 接入现有构建 |
| `src/mlir/tool/aipugc/main.cpp` | pipeline 选择、MLIR 输入、阶段输出 |
| `src/mlir/tool/aipugc/utils.h`、`utils.cpp` | 复用导出，Tile 路径跳过不安全指针改写并验证 LLVM IR |

核心函数：`ImportBinary::matchAndRewrite`、`ImportFunction::matchAndRewrite`、
`FuseAddMul::matchAndRewrite`、`checkKernel`、`lowerKernel`、`TileToLLVMPass::runOnOperation`。

## Operation 语义与限制

所有算子只支持 shape 完全一致的、静态、非空、无 encoding 的 dense FP32 Tensor。
支持 rank 0；不支持 broadcast、动态 shape、量化、激活、额外计算或内存 view。
FP32 是当前 CPU 验证类型，不代表 Zhouyi 原生 FP32 支持已确认。

| Op | Operand / Result / Attribute | 语义 |
| --- | --- | --- |
| `aiputile.add` | lhs、rhs → output | FP32 逐元素加法，pure，无 region |
| `aiputile.mul` | lhs、rhs → output | FP32 逐元素乘法，pure，无 region |
| `aiputile.fused_add_mul` | x、b、a → output；`tile_size: i64`、`add_on_lhs: bool` | 严格保留 `(x+b)*a` 的两步运算；若源 Mul 的 Add 在右边，则保留 `a*(x+b)` 的 operand 顺序；pure，无 region |

Verifier 检查类型、shape、encoding、元素总数和 Tile Size，Parser/Printer 由 ODS 生成。
Tensor 元素数上限为 INT32_MAX；这仅限制索引计算，不是设备内存容量保证。
Tile Size 必须在 `[1, INT32_MAX]`；执行时取 `min(tile_size, N)`。

`lower-aipugc-to-aiputile` 使用 Dialect Conversion，按现有公开 op 名匹配
`aipugc.add/mul/func/return`，从而避免新库依赖整个私有 graph builder。
不支持的 source op、未知参数、激活和非空 `dictAttr` 明确失败。
只接受空参数或 `with_activation = "NONE"`，不会悄悄丢弃量化/舍入语义。

`aiputile-fuse-add-mul` 在 greedy DCE 之前检查 Add 的单消费者约束；保守拒绝共享 Add，
包括目前看似无用的第二个消费者。匹配同一 block 的生产/消费关系，并保留 Mul operand 顺序。
融合与展开是独立 Pass，不能把一个黑盒 fused op 当作最终结果。

## 实际 Pipeline

```
AIPUGC Add/Mul (+ aipugc.func/return 或 func.func/return)
  → lower-aipugc-to-aiputile
AIPUTile Add / Mul
  → aiputile-fuse-add-mul{tile-size=256}
AIPUTile FusedAddMul
  → aiputile-lower-tiles
Func + SCF + Arith + MemRef
  → aiputile-to-llvm
SCF → CF；Arith / MemRef / CF / Func → LLVM
  → reconcile-unrealized-casts
LLVM Dialect
  → 现有 dumpLLVMIR 或 mlir-translate
LLVM IR
```

Tile 路径没有使用 TOSA/Linalg 类型转换、bufferization 或它们的 fusion pass。
原默认 graph 输入和 baseline 分支保留；完整 baseline 回归受上述依赖和源文件问题阻塞。

`lowerKernel` 先验证整个 entry function：三个运行时输入、一个 fused op、一个直接 return、
无额外副作用、常量、未知函数属性或 module 内调用者，然后转换成四个静态连续 MemRef 参数：

```
void kernel(X, B, A, Y)
```

调用方必须提供与三个输入均不重叠的输出 allocation，输出参数带 `llvm.noalias`；
三个输入之间可以互相 alias。该 ABI 不支持部分重叠/原地输出，不能传入任意 stride 或 offset。
变更函数签名前检查 symbol uses，避免遗留不匹配调用。

实际循环使用 Tile 编号，避免最后一次 `offset += tile_size` 的 32 位溢出：

```
for tile_id = 0 .. ceil(N/tile_size):
    offset = tile_id * tile_size
    length = min(tile_size, N-offset)
    for local = 0 .. length:
        i = offset + local
        tmp = X[i] + B[i]     // 标量 SSA
        Y[i] = tmp * A[i]
```

多维 Tensor 的 flat index 被还原为各维索引；rank 0 使用空索引。没有完整 T 的 alloc/store/load。
Tile IR 显式可见两个 `scf.for`、tail `arith.minui`、三个 `memref.load` 和一个 `memref.store`。

## 内存与 ABI 安全

CPU 使用 address space 0、64 位索引。设备候选路径使用 address space 1、32 位索引、bare pointer。
memory space 在 MemRef 阶段指定，经 LLVMTypeConverter 同时转换参数、GEP、load/store 类型。
AS1 只是从现有代码得出的设备 ABI 候选，**不是 SRAM**，还需私有后端确认。

原 `utils.cpp::setZhouyiAddrSpace1` 直接 `mutateType`，可能让参数类型与 FunctionType、
指令类型不一致。新 Tile 分支不调用该函数；它在 MLIR 内完成类型转换，并在共享导出函数中
于优化前后运行 LLVM verifier。保留 baseline 的历史实现，未把这个未验证的全局行为重写混入新路径。
因此旧路径该风险仍待独立修复/回归。

`aiputile.staging_bytes = 5 * min(tile_size, N) * 4` 仅估计 X/B/A/T/Y 五组逻辑 payload。
不含对齐、DMA descriptor、runtime 或双缓冲开销，不据此判断 SRAM 是否足够。
当前没有物理 SRAM allocation、DMA 指令或容量调度。`tmp` 是 SSA 标量；host 汇编可检查真实算术，
backend 仍可能产生少量 spill，不能声称 NPU 上的寄存器/SRAM 驻留。

## 已实际使用的构建命令

在当前云机器执行成功，不改全局工具链：

```bash
cd /workspace/AIPU_building_tool_design
export PATH=/workspace/.aipu-onboarding/venv/bin:$PATH
sdk=/workspace/.aipu-onboarding/llvm19
cmake -S src/mlir/aiputile -B /workspace/.aipu-onboarding/tile-build -G Ninja \
  -DMLIR_DIR="$sdk/lib/cmake/mlir" -DLLVM_DIR="$sdk/lib/cmake/llvm" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$sdk/bin/clang" -DCMAKE_CXX_COMPILER="$sdk/bin/clang++" \
  -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld
cmake --build /workspace/.aipu-onboarding/tile-build --parallel 4
ctest --test-dir /workspace/.aipu-onboarding/tile-build --output-on-failure
```

独立构建使用真实同一套 Tile C++ 源码，而非模拟后端。私有完整工程构建不能通过，
也没有将隔离上游 SDK 链接到 `3rdparty/llvm_mlir_sdk` 冒充私有实现。

## 测试和示例命令

保留本次运行生成的各阶段 IR、LLVM IR、真实 x86 汇编、可执行文件和结果 JSON：

```bash
python3 src/mlir/tests/aiputile/run_tests.py \
  --opt /workspace/.aipu-onboarding/tile-build/aiputile-opt \
  --llvm-bin /workspace/.aipu-onboarding/llvm19/bin \
  --artifacts /workspace/.aipu-onboarding/tile-results
```

单独编译示例，输出保存在仓库外：

```bash
opt=/workspace/.aipu-onboarding/tile-build/aiputile-opt
"$opt" src/mlir/tests/aiputile/add_mul.mlir --allow-unregistered-dialect \
  --pass-pipeline='builtin.module(lower-aipugc-to-aiputile,aiputile-fuse-add-mul{tile-size=256},aiputile-lower-tiles,aiputile-to-llvm,reconcile-unrealized-casts)' \
  -o /workspace/.aipu-onboarding/example.llvm.mlir
sdk=/workspace/.aipu-onboarding/llvm19
"$sdk/bin/mlir-translate" --mlir-to-llvmir /workspace/.aipu-onboarding/example.llvm.mlir \
  -o /workspace/.aipu-onboarding/example.ll
"$sdk/bin/opt" -passes=verify -disable-output /workspace/.aipu-onboarding/example.ll
"$sdk/bin/clang" /workspace/.aipu-onboarding/example.ll \
  src/mlir/tests/aiputile/host_runner.c -DELEMENTS=257 -O3 -ffp-contract=off \
  -o /workspace/.aipu-onboarding/example
/workspace/.aipu-onboarding/example
```

可移植驱动没有链接原 AIPUGC dialect，它只对 **source** 的 generic `aipugc.*` 使用
`--allow-unregistered-dialect`；转换自己的 verifier 仍检查类型、shape 和参数。
全部生成的 AIPUTile/SCF/MemRef/LLVM op 均真实注册、验证，并进行 parse/print round trip。
这不等于原 AIPUGC custom parser 或整个原驱动已验证。完整驱动接入的是原有真实 AIPUGC 注册。

完整私有构建恢复后，以下接入命令需要在用户服务器另行验证，**当前未执行成功**：

```bash
aipugc graph.def --pipeline=tile --tile-size=256 --emit=aiputile --dump=tile.mlir
aipugc graph.def --pipeline=tile --tile-size=256 --emit=tile-fused --dump=fused.mlir
aipugc graph.def --pipeline=tile --tile-size=256 --emit=tile-loops --dump=loops.mlir
aipugc graph.def --pipeline=tile --tile-size=256 --llvm-target=zhouyi --emit=ll --dump=tile.ll
aipugc-llir tile.ll --emit=tile.o --target-triple=zhouyi-unknown-compass --compile-opt-level=3
```

`--input-format=mlir` 可让完整驱动直接接收 MLIR。原 graph 输入和默认 `--pipeline=baseline`
保持默认。MLIR 的 `--mlir-print-ir-after-all` 等现有打印选项仍可观察每个 Pass。
对象文件后的链接/加载/执行需要真实 toolkit/runtime，尚未实现验证，不能编造调用接口。

## 数值、IR 和测试结果

本次测试覆盖 rank 0、1 元素、3、255、256、257、1024、`[2,3,5]`、
`[1,64,256,256]`，Tile Size 1/7/256/513，包含刚好一 Tile、tail、多 Tile、反向 Mul operand。
35 组数值配置分别用 `-O0` 和 `-O3` 编译执行，共 **70 次真实 CPU kernel 执行**。

输入由 host 在运行时生成，包含正数、负数、分数、±0、Inf、NaN。
reference 强制 Add 的 FP32 中间舍入；对普通数值逐项相等，对 NaN 检查 NaN，
对零检查符号。不开 fast-math，关闭 contraction，不改变运算顺序。
另执行独立 noinline Add/Mul 的非融合 CPU 对照，真实写读完整 T，检查输出两侧哨兵。
该对照不是原 AIPUGC baseline，也不证明 CPU 的 T 必然经过物理 DDR。

负向测试覆盖非法 Tile Size、动态/空/溢出 shape、类型、broadcast、encoding、多消费者、
未知 source op/参数、额外副作用、常量输入、module 内调用、未知地址空间和索引宽度。
当前完整测试为 **59 项检查通过、70 次 native kernel 执行通过，CTest 1/1 通过**。
计数以运行生成的 `results.json` 和 CTest 为准；预期拒绝属于通过的负向检查，
不计为成功编译的 kernel。设备相关检查标记为 unrun。

| 验证项 | 状态 |
| --- | --- |
| ODS / 新库 / portable driver 编译 | 通过，LLVM/MLIR 19.1.6 |
| Tile IR、tail 和 LLVM conversion | 通过 |
| LLVM IR verifier、host assembly / executable | 通过 |
| CPU 数值 / 非融合 CPU 对照 / 边界哨兵 | 70 次执行通过 |
| AS1 + 32-bit index 类型一致性 | LLVM verifier 通过；不含私有 calling convention 验证 |
| 原 AIPUGC parser / baseline / 完整驱动 | 未通过构建条件，未回归 |
| Zhouyi NPU 编译、Kernel 执行 | 未运行，私有依赖和硬件缺失 |
| 物理 SRAM / DMA / 性能计数器 | 未实现或未验证 |

## 性能分析与后续设备 benchmark

**仅理论流量，不是实测性能：** 当 baseline 确实物化 T、每元素 S=4 字节时，
非融合为 `6*N*S`，融合为 `4*N*S`，理论少 33.3%。
`[1,64,256,256]` 的 N=4,194,304，对应 96 MiB → 64 MiB，少 32 MiB。
目前 LLVM IR 和 host 输出证明没有完整 T 的显式存储；NPU 上的物理访问尚未验证。

没有设备 benchmark 命令或延迟/加速比数据，因为 runtime ABI 和设备访问尚不可用。
不能把上面的 CPU 数值测试时长或理论流量换算成 NPU speedup。
真实环境可用后，应先检查 baseline 优化后的 LLVM IR/汇编是否已融合；若已融合，
另建合法的独立 Add/Mul 非融合对照，并明确区分它与原 baseline。
随后在相同目标、CPU/features、优化等级、数据、ABI/runtime 下预热并多次计时、报告中位数；
区分 launch、loop/vectorization、T 的存储、SRAM/DMA 等因素，采集可用的 DDR/DMA/周期计数器。

后续优先级：恢复完整私有构建和原 baseline 回归 → 真正 Zhouyi codegen/ABI →
确认 SRAM/DMA 头文件和容量 → 映射物理 staging 与对齐/descriptor 预算 →
设备数值与 benchmark。尚未开展 double buffering、自动 tile 搜索或其他算子扩展。
