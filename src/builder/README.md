# Builder 模块新人指南

## 1. Builder 是什么

Builder 是 AIPU 编译工具链的后端核心模块。它接收前端解析好的**神经网络计算图（Graph）**，经过优化、调度、内存分配、代码生成等一系列步骤，最终输出可以在 AIPU 硬件上运行的**二进制文件（.bin / ELF）**。

---

## 2. 两条编译路径

Builder 内部有两条独立的编译路径，对应不同的硬件目标：

```
                        ┌─────────────────────────────────────────────┐
                        │              输入：Graph（计算图）            │
                        └────────────────────┬────────────────────────┘
                                             │
                        ┌────────────────────┴────────────────────┐
                        ▼                                         ▼
              ┌─────────────────┐                       ┌─────────────────┐
              │   MC 路径        │                       │   MQ 路径        │
              │  (Multi-Core)   │                       │  (Multi-Queue)  │
              │  ZY-ISA 目标     │                       │  Compass 目标    │
              │  X1 / X2 / X3   │                       │  新架构          │
              └─────────────────┘                       └─────────────────┘
              │                                         │
              │ 完整 lowering (50+ pass)                 │ 仅 GOP 插件
              │ Tiling / 图分割                          │ 无 Tiling
              │ AMAT/MMU 内存分配                        │ SOMS 内存分配
              │ 手写 ZY 汇编（DMA/Cache/中断）            │ 链接预编译 kernel .o
              │ 多核调度 (mc_scheduler)                  │ 单核
              │                                         │
              ▼                                         ▼
         AIPU ELF / .bin                           Compass ELF
```

---

## 3. MC 路径编译流程（mcbuilder.cpp / builder.cpp）

```
GraphBuilder::build_graph()
  │
  ├─ 1. 常量分配 ── 为权重分配 RO 内存偏移
  │
  ├─ 2. 优化器 (optimizer.cpp)
  │     └─ 完整 lowering 流水线 (lowering/lowering.cpp)
  │         ├─ passes_before_all        ── 记录原始参数
  │         ├─ passes_before_optimize   ── 算子分解/适配/融合（150+ pass）
  │         ├─ tiling                   ── 分片，处理超出 SRAM 的算子
  │         ├─ divide_graph             ── 图分割为子图
  │         ├─ unfold_graph             ── 展开子图
  │         ├─ passes_after_optimize    ── 数据排布调度、profiler 标注
  │         └─ apply_plugins            ── 应用用户/内置插件
  │
  ├─ 3. 调度器 (scheduler.cpp)
  │     ├─ 拓扑排序，分层
  │     └─ 内存分配 (AMAT 或 MMU 分配器)
  │
  ├─ 4. 负载均衡 (load_balancer.cpp)
  │     ├─ 计算密度分析 (compute-bound vs memory-bound)
  │     └─ 多核分割策略
  │
  ├─ 5. 多核调度 (mc_scheduler) ── 多 TEC 任务分配
  │
  ├─ 6. 代码生成
  │     ├─ text_generator   ── 从 aipulib.a 加载预编译 kernel 指令
  │     ├─ ro_generator     ── 生成每个算子的 RO 参数/描述符
  │     └─ ds_code_generator── 动态 Shape 汇编补丁
  │
  ├─ 7. 链接 (linker/)
  │     ├─ LinkerV1 ── 内联 ZY ISA 汇编（DMA/Cache/中断处理）
  │     └─ LinkerV2 ── 调用外部工具链，生成 ELF
  │
  └─ 8. 输出
        ├─ aipu_binfile.cpp ── .bin 格式
        └─ aipu_elf.cpp     ── ELF 格式
```

---

## 4. MQ 路径编译流程（mqbuilder.cpp）

```
MqBuilder::build()
  │
  ├─ 1. 初始化全局 Tensor（stack / print / param / stream0-3）
  │
  ├─ 2. 优化 ── optimizer.optimize_mq()（仅 GOP 插件，不做 tiling）
  │
  ├─ 3. 性能估算 ── PerfEstimator
  │
  ├─ 4. 负载均衡 ── LoadBalancer::schedule()（拓扑分层）
  │
  ├─ 5. Kernel 适配 ── KernelAdapter::kernel_adapt()
  │     └─ 将每个算子映射到对应的 kernel 库（.o / .a）
  │
  ├─ 6. 内存分配 ── SOMSScheduler::mem_allocate()
  │     └─ 分配 Tensor 到 stream slot
  │
  ├─ 7. 代码生成 ── CodeGenerator::code_gen()
  │     └─ 收集所有 kernel 库，调用 LinkerCps 链接成 Compass ELF
  │
  ├─ 8. 汇编 ── Assembler::assembly()
  │     └─ 将图节点组装成 task 描述符（task_t / launch_info_t）
  │
  └─ 9. 输出 ── ObjectGenerator::gen()
        └─ 写入最终 Compass ELF（包含 stream、参数表、常量、tensor 索引）
```

---

## 5. 文件清单与功能说明

### 5.1 顶层入口文件

| 文件 | 说明 |
|------|------|
| `builder.cpp` | `GraphBuilder` 实现，MC 路径的主编译流程入口 |
| `mcbuilder.cpp` | `GraphELFBuilder`，继承 `GraphBuilder`，输出 AIPU ELF，支持多核和 i-cache prefetch |
| `mqbuilder.cpp` | `MqBuilder`，MQ 路径的独立实现，流程清晰，**推荐新人首先阅读** |
| `tools.cpp` | 顶层工具入口（`aipugb`、`aipurun`、`aipudumper` 等命令行工具的调用入口） |
| `target.cpp` | 硬件目标定义，包含芯片 ID → ISA / TEC 数 / SRAM 大小等参数的映射表 |

### 5.2 优化与图变换

| 文件 | 说明 |
|------|------|
| `optimizer.cpp` | `BuilderOptimizer`，优化入口。MC 路径调用完整 lowering；MQ 路径仅调用 GOP 插件 |
| `reorder_optimizer.cpp` | 层内节点重排序优化，支持从 JSON 加载自定义重排策略 |
| `lowering/lowering.cpp` | **Lowering Pass 管理器**，控制 50+ 图变换 pass 的执行顺序 |
| `lowering/tiling.cpp` | Tiling 分片，将超出 SRAM 容量的算子切分为多个子任务 |
| `lowering/datalayout_schedule.cpp` | 数据排布调度（NHWC/NCHW 等），决定每个节点的 tensor layout |

`lowering/` 目录下的其他文件按功能分为：

- **算子分解**：`decompose_rmsnorm`、`decompose_layernorm`、`decompose_groupnorm` 等
- **硬件约束适配**：`hwa_conv_inc_adaptor`、`hwa_conv_outc_adaptor`、`hwa_fc_inc_adaptor` 等
- **算子融合**：`fuse_resize_crop`、`fuse_transpose_reshape_matmul`、`add_sub_mul_to_eltwise` 等
- **图结构调整**：`divide_graph`、`unfold_graph`、`remove_split`、`remove_pad` 等
- **MatMul 链路**：`adjust_matmul_inputs` → `split_matmul_for_large_outc` → `matmul_c_adaptor` → `matmul_to_fc`

### 5.3 调度与内存分配

| 文件 | 说明 |
|------|------|
| `scheduler.cpp` | 基础调度器：拓扑分层、内存分配入口、AMAT/MMU 分配器派发 |
| `load_balancer.cpp` | 负载均衡：计算密度分析（`ops / constant_size`），判断 compute-bound 还是 memory-bound，决定多核分割策略 |
| `soms_scheduler.cpp` | SOMS 调度器，MQ 路径使用，按 stream slot 分配 tensor |
| `mc_scheduler.h` | 多核调度器头文件 |
| `mc_scheduler_gm_enhance.cpp` | 多核 GM（全局内存）区域管理 |
| `mc_scheduler_mem_alloc.cpp` | 多核内存分配 |
| `allocator.cpp / .h` | 内存分配器基类 |
| `mem_allocator.cpp / .h` | 内存分配器实现，管理 slot 列表，SOC SRAM 多级区域，512 字节对齐 |
| `amat_allocator.cpp / .h` | AMAT 分配器：基于 tensor 活跃区间的内存复用，4096 字节页对齐 |
| `mmu_allocator.cpp / .h` | MMU 分配器：模拟硬件 MMU 段映射，固定地址（LSRAM `0xf0000000`、GSRAM `0xf8x00000`） |
| `sram_allocator.h` | SRAM 分配器接口 |
| `sram_optimize.cpp / .h` | SRAM 优化：判断哪些节点的输出 tensor 可以放入片上 SRAM |
| `mem_remapping.cpp / .h` | 内存地址重映射 |
| `x2_SM_opt.cpp / .h` | X2 共享内存优化 |

### 5.4 代码生成与链接

| 文件 | 说明 |
|------|------|
| `code_generator.cpp / .h` | MQ 路径代码生成：收集 kernel `.o` 文件，调用 LinkerCps 链接 |
| `ds_code_generator.cpp / .h` | 动态 Shape 代码生成：生成运行时描述符补丁的 ZY 汇编 |
| `ds_code_generator_cx.cpp / .h` | 动态 Shape 代码生成（CX 变体），MCBuilder 使用 |
| `text_generator.cpp / .h` | 加载预编译 kernel 指令（从 `aipulib.a` 或 `AIPU_LIB_PATH`），提供按算子类型缓存访问 |
| `ro_generator.cpp / .h` | RO 参数/描述符生成器注册表，每个算子注册自己的 RO 布局函数 |
| `assembler.cpp / .h` | MQ 路径汇编器：将图节点组装成 task 描述符（`task_t` / `launch_info_t`），构建 stream 任务列表 |
| `object_generator.cpp / .h` | MQ 路径对象生成器：将 stream、参数表、常量、tensor 索引写入 Compass ELF |
| `readonly_param.cpp` | 只读参数封装（`ComputeTensor` 或 `NodeParamValue`） |
| `kernel_adapter.cpp / .h` | Kernel 适配器：将算子映射到对应的预编译 kernel 库 |

### 5.5 链接器子系统（linker/）

| 文件 | 说明 |
|------|------|
| `linker_v1.cpp / .h` | V1 链接器：MC 路径使用，直接生成 ZY ISA 汇编（DMA memcpy、Cache 控制、中断处理） |
| `linker_v2.cpp` | V2 链接器：基于 ELF 的链接，调用外部工具链 |
| `linker_cps.cpp` | Compass 链接器：MQ 路径使用，调用外部 `ld` 链接 kernel `.o` 为 Compass ELF |
| `codegen.cpp / .h` | ZY ISA 汇编 DSL（`CodeBlock`、`MOV`、`ST`、`PUSH` 等） |
| `elfreader.cpp` | ELF 文件读取器，链接时读取预编译 kernel |
| `toolchain.cpp / .h` | 外部工具链调用封装（`dlopen` / `fork+exec`） |
| `ZY.h` | ZY ISA 指令编码常量和辅助函数 |

链接器输出的 text 段布局（V1）：

```
┌────────────────────┐
│   init_context     │  ← 初始化上下文
│   call main        │
├────────────────────┤
│   op lib (conv)    │  ← 各算子的预编译指令
│   op lib (pool)    │
│   op lib (...)     │
├────────────────────┤
│   main:            │  ← 主函数，按顺序 call 各算子
│     call op1       │
│     call op2       │
│     ...            │
└────────────────────┘
```

### 5.6 输出文件生成

| 文件 | 说明 |
|------|------|
| `aipu_binfile.cpp` | AIPU .bin 格式输出，包含 text / data / RO section |
| `aipu_elf.cpp` | AIPU ELF 格式输出，使用 ELFIO 库 |

### 5.7 子目录

#### ops/ — 内置算子实现

用于图变换过程中插入的虚拟/内部算子：

| 文件 | 说明 |
|------|------|
| `internal_concat/crop/pad/slice/spliter/switch.cpp` | 内部图重写算子 |
| `layoutconvertor.cpp` | 数据排布转换 |
| `reshape.cpp` | Reshape |
| `subgraphop.cpp` | 子图算子（包含完整 builder 流程） |
| `user_defined.cpp` | 用户自定义算子 |
| `forrange_header/ender.cpp` | 循环控制流算子 |
| `if_header/ender.cpp` | 条件分支控制流算子 |
| `cacheupdate.cpp` | Cache 一致性算子 |

#### emitter/ — C 代码发射器

为 DS（Dynamic Shape）kernel 生成 C 代码：

| 文件 | 说明 |
|------|------|
| `code_emitter.cpp / .h` | 发射器基类（dtype 映射、参数查找） |
| `activation/cast/eltwise/gather/logical/reshape/constant.cpp` | 各算子的 C kernel 发射 |

#### profiler/ — 性能分析

| 文件 | 说明 |
|------|------|
| `profiler.cpp` | Profiler 基础实现 |
| `profiler_x1.cpp` | X1 目标 profiler |
| `profiler_x2.cpp` | X2 目标 profiler |

#### plugin/ — 内置插件

基于 `GraphMatcherPattern` 的子图匹配替换插件：

| 文件 | 说明 |
|------|------|
| `internal_layoutreshapelayout.cpp` | Layout→Reshape→Layout 融合 |
| `internal_layoutreshapetrans.cpp` | Layout→Reshape→Transpose 融合 |
| `internal_transreshapelayout.cpp` | Transpose→Reshape→Layout 融合 |
| `multi_scalars.cpp` | 多标量融合 |

#### ar/ — 归档文件读取

| 文件 | 说明 |
|------|------|
| `ar_reader.cpp / .h` | 读取标准 `.a` 归档文件，从中提取预编译 kernel |

#### dumper/ — 调试 Dump

| 文件 | 说明 |
|------|------|
| `dumper.h` | Dumper 抽象接口 |
| `config_manager.cpp / .h` | Dump 配置管理 |
| `dump_manager.cpp / .h` | Dump 信息管理（名称、地址、大小） |
| `mem_manager.cpp / .h` | Dump 用内存分配跟踪 |
| `tcb_manager.cpp / .h` | TCB（Task Control Block）管理 |

---

## 6. 类继承关系

```
IBuilder（接口）
  └─ MqBuilder ── MQ 路径，独立实现

GraphBuilder（MC 路径基类，输出 .bin）
  └─ GraphELFBuilder ── MC 路径，输出 ELF
```

---

## 7. 推荐阅读顺序

如果你是新人，建议按以下顺序阅读代码：

1. **`mqbuilder.cpp`** — 从最简洁的 MQ 路径入手，~200 行就能看懂完整编译链路
2. **`optimizer.cpp`** — 理解优化入口，MC 和 MQ 两条路径的差异
3. **`lowering/lowering.cpp`** — 理解 50+ lowering pass 的执行顺序（MC 路径核心）
4. **`scheduler.cpp`** — 理解图调度和内存分配的派发逻辑
5. **`load_balancer.cpp`** — 理解 compute-bound / memory-bound 分类和多核分割
6. **`assembler.cpp` + `object_generator.cpp`** — MQ 路径的输出如何生成
7. **`linker/linker_v1.cpp`** — MC 路径的 ZY ISA 汇编生成
8. **`builder.cpp` + `mcbuilder.cpp`** — MC 路径完整流程（较复杂，最后看）

---

## 8. 关键概念速查

| 概念 | 含义 |
|------|------|
| **ZY ISA** | AIPU X1/X2/X3 的指令集架构 |
| **CPS** | Compass，新一代架构格式 |
| **TEC** | Thread Execution Core，AIPU 的执行核心 |
| **AMAT** | 基于活跃区间的内存分配器 |
| **MMU** | 模拟硬件 MMU 段映射的内存分配器 |
| **SOMS** | MQ 路径的内存调度器，按 stream slot 分配 |
| **GM** | Global Memory，多核共享的全局内存 |
| **LSRAM / GSRAM** | 片上 Local SRAM / Global SRAM |
| **RO** | Read-Only 参数区（权重、描述符等） |
| **GOP** | Graph Op Plugin，图级别算子插件 |
| **Tiling** | 将超出 SRAM 容量的算子切分为多个子任务 |
| **Lowering** | 将高层算子逐步变换为硬件可执行的低层表示 |

---

## 9. AIPU .bin 文件结构

### 9.1 整体布局

文件按顺序排列为 6 个部分：

```
┌──────────────────────────────────┐  offset 0
│          ABHeader (96B)          │  ← 文件头，包含各 section 的 offset/size
├──────────────────────────────────┤  header.text_offset
│          Text Section            │  ← 可执行指令（ZY ISA 机器码）
├──────────────────────────────────┤  header.rodata_offset
│          RO Section              │  ← 只读参数（算子描述符、地址引用）
├──────────────────────────────────┤  header.dcr_offset
│          Descriptor Section      │  ← 控制描述符
├──────────────────────────────────┤  header.data_offset
│          Data Section            │  ← 常量数据（权重、bias 等）
├──────────────────────────────────┤  header.bss_offset
│          BSS Section             │  ← 运行时内存分配描述（不含实际数据）
├──────────────────────────────────┤
│          Remap Section           │  ← 地址重定位表
└──────────────────────────────────┘
```

### 9.2 ABHeader（96 字节）

| 字段 | 大小 | 含义 |
|------|------|------|
| `ident` | 16B | Magic：`"AIPU BIN"` |
| `arch` | 4B | 目标架构 |
| `version` | 4B | 图版本（高 8 位，当前为 5） |
| `build_version` | 4B | 编译工具版本号 |
| `hdrsize` | 4B | Header 自身大小 |
| `filesize` | 4B | 文件总大小 |
| `type` | 4B | 主函数偏移 |
| `flag` | 4B | 特性标志位（bit0-3: ASID, bit4: 链表描述, bit5: SOC SRAM 预分配, bit6: 动态 Shape） |
| `entry` | 4B | 入口点 |
| `text_offset/size` | 8B | Text section 偏移和大小 |
| `rodata_offset/size` | 8B | RO section 偏移和大小 |
| `dcr_offset/size` | 8B | Descriptor section 偏移和大小 |
| `data_offset/size` | 8B | Data section 偏移和大小 |
| `bss_offset/size` | 8B | BSS section 偏移和大小 |
| `misc_offset/size` | 8B | 预留 |
| `extra_data` | 8B | 预留 |

### 9.3 各 Section 说明

- **Text Section**：AIPU 可执行指令二进制。布局为 `init_context → call main → op libs → main`。32 字节对齐。
- **RO Section**：每个算子的只读参数（描述符），包含 tensor 地址引用。4 字节对齐。运行时通过 Remap/BSS 机制填入实际地址。
- **Descriptor Section**：控制描述符。4 字节对齐。
- **Data Section**：常量数据（权重、bias、LUT 等）。32 字节对齐。

### 9.4 BSS Section（运行时内存描述）

不存储实际数据，只描述运行时需要分配的内存：

```
BSS Section
├─ stack_size / stack_align         ← 栈大小（默认 256KB）
├─ DataDescriptor[] (常量)           ← 描述 Data Section 中各常量块的位置
│   └─ SubSectionDescriptor[]       ← 每块的 type/size/dtype/scale/zp
└─ BufferDescriptor[] (可复用缓冲区)  ← 描述运行时需分配的缓冲区
    └─ SubSectionDescriptor[]       ← type: input(0)/output(1)/dump(2)/intermediate(3)/...
```

`SubSectionDescriptor.type` 枚举定义了 tensor 用途：
- 0=input, 1=output, 2=dump, 3=intermediate, 4=constant
- 10=profile_data, 12=printf_fifo, 15=zerocpy_constant, 16=output_tensor_shape
- 255=segmmu_buffer（X2+ 段式 MMU）

### 9.5 Remap Section（地址重定位表）

运行时需要把各 section 的基地址填入 RO/Descriptor/Text 中的引用位置。每条 entry：

```
{ loc_type, loc_offset, dest_type, dest_offset }
```

含义：`base[loc_type] + loc_offset` 处写入 `base[dest_type] + dest_offset`，其中 type 为 0=RO, 1=Descriptor, 2=Text。

---

## 10. AIPU .bin 实例详解

以一个极简模型 `Input → Conv → ReLU → Output`（输入 `[1,3,224,224]`，输出 `[1,64,112,112]`，Conv 权重 64KB）为例：

### 10.1 编译时：Builder 生成的 .bin 文件

```
文件偏移    内容
──────────────────────────────────────────────────────────
0x00        ┌─ ABHeader (96 字节) ──────────────────────┐
            │  ident = "AIPU BIN\0..."                  │
            │  text_offset = 0x60                       │
            │  text_size   = 0x300                      │
            │  rodata_offset = 0x360                    │
            │  rodata_size   = 0x100                    │
            │  data_offset = 0x460                      │
            │  data_size   = 0x10000  (64KB 权重)       │
            │  bss_offset  = 0x10460                    │
            │  flag: bit6=0 (静态 shape)                │
            └───────────────────────────────────────────┘

0x60        ┌─ Text Section (0x300 字节) ───────────────┐
            │  init_context:                            │
            │    设置栈指针、加载 TCB 等                   │
            │    call main                              │
            │  ────────────────────                     │
            │  conv_kernel:     (从 aipulib.a 加载)      │
            │    ZY ISA 指令... (Conv 的机器码)           │
            │  relu_kernel:                             │
            │    ZY ISA 指令... (ReLU 的机器码)           │
            │  ────────────────────                     │
            │  main:                                    │
            │    call conv_kernel                       │
            │    call relu_kernel                       │
            │    return                                 │
            └───────────────────────────────────────────┘

0x360       ┌─ RO Section (0x100 字节) ────────────────┐
            │  Conv 描述符:                             │
            │    [0x00] input_addr  = 0x00000000  ← ★  │
            │    [0x04] weight_addr = 0x00000000  ← ★  │
            │    [0x08] output_addr = 0x00000000  ← ★  │
            │    [0x0C] kernel_h=3, kernel_w=3         │
            │    [0x10] stride=2, pad=1                │
            │    ...                                    │
            │  ReLU 描述符:                             │
            │    [0x80] input_addr  = 0x00000000  ← ★  │
            │    [0x84] output_addr = 0x00000000  ← ★  │
            │    ...                                    │
            │  (★ 标记的地址是占位符，运行时通过         │
            │   Remap/BSS 机制填入实际物理地址)          │
            └───────────────────────────────────────────┘

0x460       ┌─ Data Section (0x10000 字节) ────────────┐
            │  Conv 权重二进制数据 (64KB)                │
            │  0x1A 0x3F 0x... (实际的 int8 权重值)     │
            └───────────────────────────────────────────┘

0x10460     ┌─ BSS Section ─────────────────────────────┐
            │  stack_size = 0x40000 (256KB)              │
            │  stack_align = 16                          │
            │                                            │
            │  DataDescriptor[0]:  (常量 - 权重)         │
            │    data_offset = 0x0  (在 Data Section 内) │
            │    size = 0x10000                           │
            │    SubSection[0]:                          │
            │      type = 4 (constant/weight)            │
            │      offset_in_ro = [0x04]                 │
            │      ↑ 说明 RO 的第 0x04 字节处存着        │
            │        weight 的地址，运行时要回填          │
            │                                            │
            │  BufferDescriptor[0]: (可复用缓冲区 - 输入) │
            │    size = 0x24C00 (150KB, 3×224×224)       │
            │    align = 32                              │
            │    SubSection[0]:                          │
            │      type = 0 (input)                     │
            │      id = 0                                │
            │      dtype = 3 (int8)                      │
            │      scale = 0.0078, zp = 128              │
            │      offset_in_ro = [0x00]                 │
            │      ↑ RO[0x00] 处存着 input 的地址        │
            │                                            │
            │  BufferDescriptor[1]: (中间结果)            │
            │    size = 0xE0000 (896KB, 64×112×112)      │
            │    align = 32                              │
            │    SubSection[0]:                          │
            │      type = 3 (intermediate)              │
            │      offset_in_ro = [0x08, 0x80]           │
            │      ↑ Conv 的 output 和 ReLU 的 input     │
            │        共用这块内存 (同一个 tensor)         │
            │                                            │
            │  BufferDescriptor[2]: (输出)               │
            │    size = 0xE0000                           │
            │    SubSection[0]:                          │
            │      type = 1 (output)                     │
            │      id = 0                                │
            │      offset_in_ro = [0x84]                 │
            └────────────────────────────────────────────┘

            ┌─ Remap Section ───────────────────────────┐
            │  entry_num = 2                             │
            │  [0] loc_type=0(RO), loc_offset=0x0C,      │
            │      dest_type=2(Text), dest_offset=0x80   │
            │      → 把 text 的 conv_kernel 地址          │
            │        填入 RO 的某个引用位置               │
            │  [1] loc_type=0(RO), loc_offset=0x8C,      │
            │      dest_type=2(Text), dest_offset=0x140  │
            │      → 把 relu_kernel 地址填入 RO           │
            └────────────────────────────────────────────┘
```

### 10.2 运行时：Driver 如何使用这个 .bin

```
步骤 1：Driver 读取 .bin，解析 Header
        ↓
步骤 2：根据 BSS Section 分配物理内存
        ├─ 栈：分配 256KB → 得到物理地址 0x80000000
        ├─ 常量 (权重)：分配 64KB → 得到物理地址 0x90000000
        │   └─ 把 Data Section 的权重数据拷贝进去
        ├─ Buffer[0] (输入)：分配 150KB → 得到物理地址 0xA0000000
        │   └─ 用户把输入图片数据写入此处
        ├─ Buffer[1] (中间)：分配 896KB → 得到物理地址 0xB0000000
        └─ Buffer[2] (输出)：分配 896KB → 得到物理地址 0xC0000000
        ↓
步骤 3：地址回填（核心！）
        根据 BSS 中每个 SubSection 的 offset_in_ro 信息，
        把真实物理地址写入 RO Section：

        RO[0x00] = 0xA0000000          ← input 地址
        RO[0x04] = 0x90000000          ← weight 地址
        RO[0x08] = 0xB0000000          ← conv output 地址
        RO[0x80] = 0xB0000000          ← relu input 地址（同一块内存）
        RO[0x84] = 0xC0000000          ← relu output 地址
        ↓
步骤 4：处理 Remap Section
        把 Text / RO / Desc 各 section 自身的
        加载基地址填入交叉引用的位置
        ↓
步骤 5：把 Text、RO、Data 加载到 AIPU 可访问的内存，启动执行
        AIPU 从 Text 的 init_context 开始执行：
          → 读 RO 中的 Conv 描述符 → 得到 input/weight/output 地址
          → 执行 conv_kernel
          → 读 RO 中的 ReLU 描述符 → 得到 input/output 地址
          → 执行 relu_kernel
          → 返回
        ↓
步骤 6：Driver 从 Buffer[2]（0xC0000000）读出推理结果
```

### 10.3 关键机制总结

**为什么需要这套设计？**

AIPU 是独立硬件，编译时不知道运行时的物理地址。所以：

1. **编译时**：RO 中所有地址写 0 占位，但通过 BSS 的 `offset_in_ro` 记录下"哪个地址需要填到 RO 的哪个位置"
2. **运行时**：Driver 分配好真实内存后，按 `offset_in_ro` 指引把地址逐个回填到 RO 中
3. **中间 tensor 复用**：Conv 的 output 和 ReLU 的 input 是同一个 tensor，它们的 `offset_in_ro` 指向不同 RO 位置（`[0x08, 0x80]`），但运行时填入同一个物理地址

这就是 `set_reuse_address()` 和 `set_const_address()` 函数做的事——遍历 BSS 中每个 SubSection 的 `offset_in_ro` 列表，把物理地址写入 RO 对应位置。
