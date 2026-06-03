# MqBuilder 详解

`MqBuilder::build()` 是 MQ 路径的完整编译流程，按执行顺序逐步说明：

## 第 1 步：初始化全局 Tensor（第 64-77 行）

```cpp
TensorPtr sp = std::make_shared<Tensor>("stack", TensorShape({8192}), Tensor::UINT8);
m_graph->global_tensors()["stack"] = sp;
// ... print, param, stream0~3
```

为 Compass 硬件预分配 7 个全局 tensor：
- **stack**（8KB）：kernel 执行时的栈空间
- **print**（8KB）：printf 调试输出缓冲区
- **param**（动态大小）：kernel 参数区
- **stream0~3**（动态大小）：4 个数据流通道，用于 kernel 间的数据传递

这些是 Compass 架构的固定资源，每个 kernel 运行时都需要。

## 第 2 步：图优化（第 79-83 行）

```cpp
BuilderOptimizer optimizer;
optimizer.optimize_mq(m_graph);
```

调用 `optimize_mq()`，**只执行 GOP 插件**（不做 tiling、不做完整 lowering）。然后 `check_valid_name()` 检查图中所有节点名称是否合法（不能有重复/非法字符）。

## 第 3 步：性能估算（第 85-86 行）

```cpp
PerfEstimator perf;
perf(m_graph);
```

对每个节点估算计算量和耗时，结果会附加到节点属性上，用于后续调度决策。

## 第 4 步：负载均衡 + 拓扑分层（第 88-92 行）

```cpp
aipubt::BuilderConfig::set(Constants::single_core_opt, true);
LoadBalancer load_balancer;
load_balancer.set_graph(m_graph);
load_balancer.schedule();
auto levels = load_balancer.get_levels();
```

强制设为**单核模式**（MQ 路径不做多核分割）。`LoadBalancer::schedule()` 对图做拓扑排序，把节点分成多个 level（同一 level 的节点可以并行）。`levels` 的结果是一个二维数组：`levels[i]` = 第 i 层的节点列表。

## 第 5 步：Kernel 适配（第 96-103 行）

```cpp
KernelAdapter kernel_adapter;
kernel_adapter.set_graph(m_graph);
kernel_adapter.kernel_adapt();
auto const_tensors = kernel_adapter.get_const_tensors();
auto kernel_lib_names = kernel_adapter.get_kernel_lib();
```

这是 MQ 路径的关键步骤。`KernelAdapter` 遍历图中每个算子节点，查找对应的预编译 kernel 实现：
- **`kernel_adapt()`**：为每个子图节点匹配 kernel 库文件（`.o`/`.a`）
- **`get_const_tensors()`**：收集所有常量 tensor（权重等）
- **`get_kernel_lib()`**：返回 `map<op名, 库文件路径>` 的映射

## 第 6 步：内存分配（第 105-157 行）

```cpp
SOMSScheduler scheduler;
scheduler.set_graph(m_graph);
scheduler.set_levels(levels);
scheduler.set_gm_size({0});     // 无全局共享内存
scheduler.disable_sm();          // 禁用共享内存优化
```

SOMS 调度器按 level 顺序遍历所有 tensor，决定每个 tensor 分配到哪个 stream slot。核心逻辑：

- **`not_reuse_tensor`**：全局 tensor（stack/print 等），不能被其他 tensor 复用
- **`input_tensors` / `output_tensors`**：模型的输入输出，必须独占内存
- **`dump_all_tensors` 模式**（第 122-142 行）：如果开启了 dump 调试，把所有中间节点的输出也标记为 output，防止被复用覆盖
- **`mem_allocate()`**：返回 `data_slots`，每个 slot 是一块可复用的内存区域
- **`const_allocate()`**：为常量 tensor 单独分配（常量不可复用）

每个 slot 被赋予 `buffer_id`（第 153 行），后续写入 ELF 时需要。

## 第 7 步：代码生成 — 链接 kernel（第 160-163 行）

```cpp
CodeGenerator code_gen;
code_gen.set_kernel_lib(kernel_lib_names);
code_gen.code_gen(m_elf);
```

把第 5 步收集到的所有 kernel `.o` 文件，通过 `LinkerCps` 链接成一个 Compass ELF。此时 `m_elf` 已经包含了所有 kernel 的可执行代码。

## 第 8 步：汇编 — 生成任务描述符（第 165-174 行）

```cpp
Assembler assembler;
assembler.set_graph(m_graph);
assembler.set_elf(m_elf);
assembler.assembly();
auto streams = assembler.get_streams();
auto param_offset_table = assembler.get_param_offset_table();
```

`Assembler` 遍历图的每个子图节点，为每个 kernel 生成一个 `task_t` 描述符（包含 `launch_info_t`：IB 大小、寄存器数、ISA 版本、subnormal/saturation 标志等），然后把所有 task 按执行顺序组织成 **stream 任务列表**。同时生成：
- **`param_offset_table`**：参数在 param 区的偏移表
- **`tensor_index_map`**：tensor 到索引的映射（运行时用来做地址回填）
- **`const_index_map`**：常量 tensor 的索引映射

## 第 9 步：输出 — 写入 Compass ELF（第 176-190 行）

```cpp
ObjectGenerator object_generator;
object_generator.set_streams(streams);
object_generator.set_param_offset_table(param_offset_table);
object_generator.set_params(params);
object_generator.set_tensor_index_map(m_tensor_index_map);
object_generator.set_const_index_map(const_index_map);
object_generator.set_slots(data_slots);
object_generator.set_graph(m_graph);
object_generator.set_constant(const_tensors);
object_generator.gen(m_elf);
```

把前面所有步骤的产出汇总，写入最终的 Compass ELF：
- **streams**：task 执行流（哪些 kernel 按什么顺序跑）
- **params**：kernel 参数二进制
- **slots**：内存分配方案（运行时 driver 据此分配物理内存）
- **constants**：权重等常量数据
- **tensor/const index map**：地址索引表（类似 aipu.bin 的 BSS，供 driver 做地址回填）

---

## 整体数据流

```
Graph
  → optimize_mq()         只跑 GOP 插件
  → PerfEstimator          估算每个 op 的算力
  → LoadBalancer           拓扑分层
  → KernelAdapter          每个 op → kernel .o 文件
  → SOMSScheduler          tensor → 内存 slot
  → CodeGenerator          kernel .o → 链接成 ELF 代码段
  → Assembler              op → task 描述符 → stream 任务流
  → ObjectGenerator        全部打包 → 最终 Compass ELF
```

---

## 附录：KernelAdapter 详解

### 执行流程

```
遍历 graph 的所有顶层节点
  └─ 如果是 SubGraphNode（子图节点）
       └─ 遍历子图内的每个算子节点 n
            ├─ 跳过 Input / SubGraphInput 节点
            │
            ├─ 检查节点是否有 "plugin_impl" 属性
            │   │
            │   ├─ 有 → 从 GopImplPtr 中提取：
            │   │   ├─ kernel_infos()  → 该算子需要的所有 kernel
            │   │   │   ├─ kernel_name → 存入 m_kernel_lib[name] = lib_path
            │   │   │   └─ args → 遍历每个参数：
            │   │   │       ├─ const_tensor?  → 存入 m_const_tensors（权重等常量）
            │   │   │       ├─ workspace?     → 存入 m_ws_tensors（临时工作空间）
            │   │   │       └─ surface?       → 存入 m_surface_tensors（surface tensor）
            │   │   └─ kernel_cpsbin() → 获取 kernel 库文件路径（.o/.a）
            │   │
            │   └─ 没有 → 报错（MQ 路径要求每个算子必须有插件实现）
```

### 关键点

1. **前提条件**：在 `optimize_mq()` 阶段，GOP 插件已经为每个算子节点设置了 `plugin_impl` 属性。这个属性是一个 `GopImplPtr`，包含该算子在 Compass 硬件上的 kernel 实现信息。

2. **`kernel_infos()`** 返回的是一个列表，因为一个算子可能需要多个 kernel（比如一个复杂算子被拆成多个小 kernel 串行执行）。每个 `kernel_info` 包含：
   - `kernel_name`：kernel 函数名（比如 `"conv2d_nhwc_int8"`）
   - `args`：kernel 的参数列表，每个参数关联一个 tensor

3. **`kernel_cpsbin()`** 返回 kernel 编译产物的路径（`.o` 或 `.a` 文件）。如果返回空字符串，会被设为 `"GB_DEFAULT_PATH"` 走默认搜索。

4. **四个输出**：

   | 输出 | 内容 | 下游用途 |
   |------|------|----------|
   | `m_kernel_lib` | `{kernel_name → 库文件路径}` | 传给 `CodeGenerator` 做链接 |
   | `m_const_tensors` | 常量 tensor 集合（权重等） | 传给 `SOMSScheduler` 做常量内存分配 |
   | `m_ws_tensors` | workspace tensor → 节点映射 | kernel 执行时的临时缓冲区 |
   | `m_surface_tensors` | surface tensor → 节点映射 | 硬件 surface 描述符 |

### 一句话总结

KernelAdapter 就是一个**信息收集器**——它不做任何变换，只是从每个节点的 GOP 插件实现中，把"用哪个 kernel、kernel 在哪个库文件、需要哪些常量/workspace"这些信息提取出来，供后续的内存分配和代码链接使用。

### 什么是 GOP

GOP = **Graph Op Plugin**（图算子插件），是 MQ 路径的算子实现机制：

- `GopPlugin` 定义**匹配模式**（`patterns()`）和**数据类型能力**（`dtype_cap()`），告诉 Builder "我能处理哪些算子/子图组合"
- `GopImpl` 是匹配成功后的**具体实现**，提供 `kernel_infos()`（kernel 名称、grid/block 维度、参数列表）和 `kernel_cpsbin()`（编译产物路径）

简单说：**GOP 插件 = 告诉 Builder 这个算子用哪个预编译 kernel 来跑，以及怎么配置参数**。
