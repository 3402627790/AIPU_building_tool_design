# AIPUTile：先分块，再按 Tile 融合

## 本次修改的目标

AIPUTile 现在把 **Tile 作为 IR 中的一等值**，不再用 Tensor 作为
`aiputile.add/mul` 的输入输出，也不再先构造整个 Tensor 的 `FusedAddMulOp`。
实际顺序是：

```
AIPUGC
  → 自定义 Tile 类型和边界转换
  → 各个 Add/Mul 独立分块
  → 对相同分块域的生产者/消费者进行 Region 融合
  → SCF + Arith + MemRef + Func
  → LLVM Dialect → LLVM IR
```

独立分块后的 IR 有两个 `aiputile.tile_loop`，分别表达 Add 和 Mul 的 Tile 序列；
融合后有一个 `tile_loop`，其 body 中保留 Add 和 Mul，临时结果在 body 内以 SSA 值传递。
没有使用 TOSA、Linalg、它们的 tiling/fusion pass 或 bufferization。

当前交付仍是 **逻辑 Tile 的可移植实现**。私有 Zhouyi 后端、AIPU SDK、kernel 库和
可用设备不在当前云环境中。设备编译、设备执行、SRAM/DMA 和设备性能不属于已验证结果。

## Tile 的具体语义

类型由 ODS TypeDef 定义：

```mlir
!aiputile.tile<256xf32>
```

- `256` 是这块不可变逻辑值的元素容量，`f32` 是元素类型。
- 类型不承诺它已加载到 SRAM，不包含硬编码的 SRAM 地址空间。
- Tile 是计算值，不是可任意读写的 MemRef 指针；输入数据的视图边界由
  `from_tensor`、`tile_loop` 的 partition 语义定义，实际 buffer 生命周期由后续 Lowering 决定。
- 仅支持 FP32、正容量、最多 INT32_MAX 个元素。原输入 Tensor 要求静态、非空、
  无 encoding、相同 Shape。支持 rank 0 和多维连续布局，计算域按行主序展平。

函数边界仍可使用原 Tensor ABI，便于接入 AIPUGC：`from_tensor` 把静态 Tensor 的
完整展平范围表示成 Tile 值，`to_tensor` 恢复原 Shape。Tile 算术的 Operand/Result
全部是自定义 Tile 类型，Tensor 只存在于这些显式边界。

在分块之前，`!aiputile.tile<257xf32>` 表示长度257的完整逻辑计算域；分块后，
循环 body 内是 `!aiputile.tile<256xf32>`，每次只处理该域的一块。
**不能把循环外的完整逻辑域理解为一次必须载入片上的257元素物理块。**

## Operation 集合

| Op | 输入 / 输出 | 含义 |
| --- | --- | --- |
| `aiputile.from_tensor` | 静态 Tensor → Tile | 从 Tensor 建立只读的完整逻辑数据域，验证展平元素数和 FP32 |
| `aiputile.to_tensor` | Tile → 静态 Tensor | 恢复边界结果 Shape，验证展平元素数一致 |
| `aiputile.add` | Tile、Tile → Tile | 对当前有效范围执行逐元素 Add，类型必须一致 |
| `aiputile.mul` | Tile、Tile → Tile | 对当前有效范围执行逐元素 Mul，保留左右输入次序 |
| `aiputile.tile_loop` | 多个完整域 Tile → 一个完整域 Tile | 按 `tile_size` 同步切取所有输入，在 body 中执行每块计算，并将各块结果按原位置组合 |
| `aiputile.yield` | 一个当前块 Tile | 把 body 的当前块结果交给所属 `tile_loop` |

`FusedAddMulOp` 已从新路径移除。融合通过合并计算 Region 表达，仍可观察 Tile Add/Mul，
不需要为每一种算子组合定义一个专用融合 Op。

## 用循环表达 Tile 序列，不静态复制所有块

设完整域长度为 N，请求分块大小为 T：

```
capacity = min(T, N)
count = ceil(N / capacity)
for tile_id in [0, count):
    offset = tile_id * capacity
    valid_length = min(capacity, N - offset)
    执行 body
```

`tile_loop` body 的 Block Argument 依次为：

1. 每个输入的一块 `!aiputile.tile<capacityxf32>`；
2. 当前块在完整域中的 `offset : index`；
3. 当前块的 `valid_length : index`。

N=257、T=256 时，IR 内只有一个循环 body，但它有两次逻辑执行：

| tile_id | offset | capacity | valid_length | 有效范围 |
| --- | --- | --- | --- | --- |
| 0 | 0 | 256 | 256 | `[0, 256)` |
| 1 | 256 | 256 | 1 | `[256, 257)` |

尾部块维持256的类型容量，有效长度为1。只有有效前缀可以被计算和写出；
Lowering 的内层循环上界是 `valid_length`，不会读取或写入无效元素。

对于 `[1,64,256,256]`，T=256 时有16,384次逻辑迭代，但不会静态产生16,384份 Add/Mul。
每个算子仍只有一个 `tile_loop` body。测试检查大 N 时 IR 的 TileLoop 和算术操作数量不增长。

## 独立分块 Pass

`aiputile-tile{tile-size=256}` 在各个未分块的 Add/Mul 上分别建立 TileLoop。
对于当前示例，生成的关系为：

```
完整域 X/B → Add 的 tile_loop → 完整域 T
完整域 T/A → Mul 的 tile_loop → 完整域 Y
```

Add 和 Mul 的两个循环采用同一个分块参数。body 中的运算输入输出已经变为当前块容量，
并携带显式 `offset`、`valid_length`，所以可以独立检查分块策略及尾部。

Tile Size 必须在 `[1, INT32_MAX]`。若请求大于 N，则有效容量取 N。
这里的约束只保证逻辑分块和索引计算，不是硬件 SRAM 容量判断。

## 按 Tile 域融合

`aiputile-fuse-tiles` 在分块完成后匹配生产者/消费者 TileLoop，检查：

- 同一 block 中存在直接生产/消费关系，生产者结果只有一个消费者；
- 完整输入输出域、元素类型、有效分块容量 `min(tile_size,N)` 和循环内当前块类型一致；
- body 只包含受支持的 Tile Add/Mul 和 Yield，没有副作用或非法捕获；
- 生产者 body 的结果可以映射为消费者当前 Tile 的输入。

对于 Add 和 Mul，融合后一个 TileLoop body 中先执行 Add，再执行 Mul：

```
当前 X_tile、B_tile → Add → T_tile → Mul ← 当前 A_tile
                                      ↓
                                    Y_tile
```

`T_tile` 成为同一 body 内的 SSA 结果。有效分块容量不同的两个域不能直接融合，当前版本保守报错，不自动重新分块。
若两个请求大小都大于 N，得到相同的有效分块容量，则允许融合。共享生产者也保守拒绝。

这一步融合的是相同范围的 Tile 计算，而不是把完整 Tensor 算子合成一个黑盒。
原先的 `aiputile-fuse-add-mul{tile-size=...}` 命令需要更新，Tile Size 已移到分块 Pass；
即使保留旧融合名称作为别名，它也只执行融合，不会自动分块。新命令应明确分成 `aiputile-tile{tile-size=...}` 和 `aiputile-fuse-tiles` 两个阶段。

## Lowering、存储与对照版本

`aiputile-lower-tiles` 处理 TileLoop body，并把函数改为输出参数 ABI：

```
void kernel(X, B, A, Y)
```

三个输入和 Y 是静态连续 MemRef。调用方必须为 Y 提供与输入不重叠的存储；
输出参数带 `llvm.noalias`，输入之间允许互相 alias。模块内有调用者时拒绝改变 entry ABI。

Lowering 把外层 Tile 循环、内层有效元素循环展开为 SCF/Arith/MemRef，
多维 flat index 被还原为各维下标，rank 0 使用空下标。
Tile 类型和边界转换在此被消除。

可以选择 Lowering **未融合** 或 **已融合** 的 TileLoop，用于真实的同路径对照：

| 内容 | 未融合 TileLoop | 融合 TileLoop |
| --- | --- | --- |
| Add/Mul 所在循环 | 两套 Tile/元素循环 | 一套 Tile/元素循环 |
| T 的存储 | 全 N 元素临时 MemRef，Add 写出、Mul 读回 | 同一 body 内的 SSA 结果 |
| 临时分配生命周期 | `memref.alloc` → 两个循环 → `memref.dealloc` | 没有完整 T 的临时分配 |
| 最终输出 | 写入调用方的 Y | 写入调用方的 Y |

该未融合对照是 **Tile 路径自己的对照版本**，不是原有 TOSA/Linalg baseline。
下游 LLVM 仍有权优化 IR，物理 DDR 流量必须根据最终目标代码/硬件行为确认。

CPU 使用 address space 0、64位索引。候选设备路径使用 AS1、32位索引、bare pointer，
通过 LLVMTypeConverter 一致转换函数参数和内存操作。AS1 不代表 SRAM。
当前设备 Lowering 只支持完全融合、无需动态临时分配的路径；未融合的 AS1 临时存储明确拒绝。

原 baseline 的 `setZhouyiAddrSpace1` 有直接 `mutateType` 的历史风险。Tile 路径继续避开该函数，
在 MLIR 类型转换中处理地址空间，并在共享 LLVM 导出阶段进行 verifier 检查。
尚未实现真实 SRAM 分配、DMA、对齐/descriptor 预算、双缓冲或异步执行。

## 主要修改文件

| 文件 | 用途 |
| --- | --- |
| `src/mlir/include/aiputile/AIPUTile.td` | dialect 定义 |
| `src/mlir/include/aiputile/AIPUTileTypes.td`、`AIPUTileTypes.h` | Tile 类型 ODS、生成声明 |
| `src/mlir/aiputile/AIPUTileTypes.cpp` | Tile 类型验证和解析/打印实现 |
| `src/mlir/include/aiputile/AIPUTileOps.td`、`AIPUTileOps.h` | Tile 算术、边界、循环和 Yield 定义 |
| `src/mlir/aiputile/AIPUTileOps.cpp` | Tile 类型、范围、body 和 Yield 验证 |
| `src/mlir/aiputile/AIPUTileDialect.cpp` | 类型及操作注册 |
| `src/mlir/include/aiputile/CMakeLists.txt` | TypeDef/Operation TableGen |
| `src/mlir/aiputile/Passes.cpp`、`include/aiputile/Passes.h` | Import、独立分块、TileLoop 融合、Lowering、LLVM 转换 |
| `src/mlir/aiputile/CMakeLists.txt` | 独立构建及可移植测试驱动 |
| `src/mlir/tool/aipugc/main.cpp` | Tile pipeline 的新阶段顺序 |
| `src/mlir/tests/aiputile/run_tests.py` | 阶段 IR、拒绝用例、融合/未融合数值和 LLVM 检查 |
| `src/mlir/tests/aiputile/host_runner.c` | 运行时数据、CPU reference、独立 Add/Mul 对照和输出哨兵 |

## 构建与测试

独立构建使用当前已隔离安装的官方 LLVM/MLIR/Clang 19.1.6：

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

保存各阶段 IR、LLVM IR、真实 CPU 汇编和执行结果：

```bash
python3 src/mlir/tests/aiputile/run_tests.py \
  --opt /workspace/.aipu-onboarding/tile-build/aiputile-opt \
  --llvm-bin /workspace/.aipu-onboarding/llvm19/bin \
  --artifacts /workspace/.aipu-onboarding/tile-first-results
```

数值矩阵覆盖 rank 0、1、3、255、256、257、1024、`[2,3,5]` 和
`[1,64,256,256]`，Tile Size 1/7/256/513，并检查 Mul 左右输入顺序。
35组 Add/Mul 配置分别编译执行融合和未融合两个版本，各用 O0/O3，完成140次 native 执行。
另对两个独立生产者汇合的三计算图 `P=Add(X,B), Q=Mul(B,A), Y=Mul(P,Q)`，
分别执行融合/未融合和 O0/O3，完成4次额外执行，验证多次 Region 合并、输入去重和 SSA 映射。
本次完整脚本实际通过 **70项检查、144次真实 CPU Kernel 执行**，结果保存在
`/workspace/.aipu-onboarding/tile-first-results/results.json`；NPU 相关检查明确标记 unrun。

Host 在运行时生成输入，检查正负数、分数、±0、Inf/NaN，要求 FP32 运算顺序与舍入一致，
禁用 contraction，不启用 fast-math。两条生成路径均与 CPU reference 和独立 noinline Add/Mul
对照比较，检查 Y 两侧哨兵。自动测试还检查非法类型、容量、分块、尾部、域不一致、
共享生产者（包括结果还需通过 Tensor 边界返回）、副作用、Region 捕获、IR 大小和 AS1 类型转换。

## 逐阶段观察命令

以下在完整工程的 `src/mlir/build` 中执行。独立构建时把 `./aiputile/aiputile-opt`
替换为 `/workspace/.aipu-onboarding/tile-build/aiputile-opt`。

```bash
# 1. Tensor 只保留在边界，算术已经使用 !aiputile.tile
./aiputile/aiputile-opt ../tests/aiputile/add_mul.mlir \
  --allow-unregistered-dialect \
  --pass-pipeline='builtin.module(lower-aipugc-to-aiputile)' \
  -o add_mul_imported.mlir

# 2. 两个独立 TileLoop，分别对应 Add/Mul 的块序列
./aiputile/aiputile-opt add_mul_imported.mlir \
  --pass-pipeline='builtin.module(aiputile-tile{tile-size=256})' \
  -o add_mul_tiled.mlir

# 3. 合并对应 Tile 的 body，仍能看到 Tile Add/Mul
./aiputile/aiputile-opt add_mul_tiled.mlir \
  --pass-pipeline='builtin.module(aiputile-fuse-tiles)' \
  -o add_mul_fused.mlir

# 4. 生成有效元素循环，完整中间 T 不再分配
./aiputile/aiputile-opt add_mul_fused.mlir \
  --pass-pipeline='builtin.module(aiputile-lower-tiles)' \
  -o add_mul_fused_loops.mlir

# 未融合对照：跳过步骤3，直接下沉分块后的两个 TileLoop
./aiputile/aiputile-opt add_mul_tiled.mlir \
  --pass-pipeline='builtin.module(aiputile-lower-tiles)' \
  -o add_mul_unfused_loops.mlir

# 5. LLVM Dialect
./aiputile/aiputile-opt add_mul_fused_loops.mlir \
  --pass-pipeline='builtin.module(aiputile-to-llvm,reconcile-unrealized-casts)' \
  -o add_mul_llvm.mlir
```

也可以一次运行：

```bash
./aiputile/aiputile-opt ../tests/aiputile/add_mul.mlir \
  --allow-unregistered-dialect \
  --pass-pipeline='builtin.module(lower-aipugc-to-aiputile,aiputile-tile{tile-size=256},aiputile-fuse-tiles,aiputile-lower-tiles,aiputile-to-llvm,reconcile-unrealized-casts)' \
  -o add_mul_llvm.mlir
```

完整主驱动的 Tile 分支也按相同顺序组装 Pass，可选择：

```bash
aipugc graph.def --pipeline=tile --emit=aiputile --dump=imported.mlir
aipugc graph.def --pipeline=tile --tile-size=256 --emit=tile-partitioned --dump=tiled.mlir
aipugc graph.def --pipeline=tile --tile-size=256 --emit=tile-fused --dump=fused.mlir
aipugc graph.def --pipeline=tile --tile-size=256 --emit=tile-loops --dump=loops.mlir
```

`--input-format=mlir` 可让完整驱动接收 MLIR 而不是 graph.def。完整私有工程受环境依赖限制，
上面这些主驱动接入命令需要在用户服务器验证；portable 测试不能替代这项回归。

后续使用匹配 SDK 的 `mlir-translate --mlir-to-llvmir` 输出 `.ll`，
`opt -passes=verify` 验证，再用 host Clang 或 `llc` + C 编译器与 `host_runner.c` 链接。
`ELEMENTS=257` 必须匹配静态 Shape；设备 LLVM IR 不能当作 CPU IR 直接链接执行。

## 当前环境与未完成事项

- 当前可移植 SDK 是官方19.1.6，安装在仓库外，未改全局工具链。
  新的 Region 融合直接克隆和映射 body，不依赖已经更名的 Greedy Rewrite API。
- portable harness 只对源 AIPUGC generic spelling 使用 `--allow-unregistered-dialect`；
  新 Tile 类型与操作全部注册。这不代表原 AIPUGC custom parser 或完整原驱动已回归。
- 原默认 baseline 分支继续保留。云环境缺少私有 SDK、第三方依赖、`modules/compassLLM`、
  二进制 cost model；HEAD 的 `compute_tensor.h` 还有重复头部和残缺声明，阻止完整驱动构建。
- Zhouyi calling convention、设备 FP32/指令、实际 SRAM 容量、DMA API、对象链接/加载和
  NPU runtime 启动没有得到当前环境验证，不将 AS1 或逻辑 Tile 当作物理 SRAM。
- 类型目前是1D静态容量；源多维 Shape 在边界保留，计算使用展平域。动态 Shape、broadcast、
  更复杂布局、自动重分块、reduction/halo 和跨块依赖不在本版支持范围。

## 性能解释

未融合 Tile 路径在 Lowering 后显式写读完整 T，融合路径在同一 Tile body 内直接传递 SSA 值。
这提供了可复现的结构对照，不需要假定原 TOSA/Linalg baseline 一定没有融合。

若最终设备代码确实物化 T，且忽略缓存、对齐和元数据，FP32 非融合总流量为 `6*N*4`，
融合为 `4*N*4`，理论少33.3%；N=4,194,304时为96 MiB → 64 MiB。
这是理论模型，不能直接当作 DDR 实测数据或加速比。LLVM 优化可能进一步改变循环和访存，
应检查最终 IR/汇编，并在相同 target、features、优化等级、runtime 和输入下进行设备计时与计数器采样。
物理 SRAM/DMA 和设备 benchmark 仍待真实工具链/硬件接口可用后实现。
