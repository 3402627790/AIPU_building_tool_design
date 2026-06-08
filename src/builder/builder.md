# builder.cpp 函数速览

[builder.cpp](builder.cpp) 是 MC/GraphBuilder 路径的核心实现文件，负责把 graph 编译成 AIPU bin。它覆盖图构建、优化、调度、内存分配、assembly，以及 RO/descriptor/text/data/BSS/remap/header 等 section 的生成。

## 文件内辅助函数

### `aligned(size_t s, size_t align = 256)`

把字节数 `s` 向上对齐到 `align` 的整数倍，默认对齐到 256 字节。常用于常量、workspace、RO/descriptor 等内存区域的大小和地址对齐。

### `allocate_constant_data(...)`

为各 level 中节点使用的常量 tensor 分配常量数据空间，并生成对应的 `MemoryObject` slot。它会处理共享常量、zero-copy 常量、large-kernel 需要放入 DTCM 的权重，以及可选的随机寄存器数据。

### `dump_tune_strategy(...)`

把 load balance / GM 使用相关的调试信息写到 `tune_strategy.json`。它遍历调度 level 和 subgraph，记录每个内部节点的 level、tiling、GM 使用状态等信息，主要用于调参和性能分析。

### `set_loop_scan(GraphPtr g)`

递归遍历图和子图，为 Loop 节点的 scan output 和 dependency input 标记 `loop_scan` memory 属性。这个标记会影响后续内存对象的处理方式，避免循环 scan 相关 tensor 被错误复用或错误布局。

### `show_buffer_slots(GraphPtr m_graph, std::vector<MemoryObjectPtr> slots)`

递归打印每个 tensor 被哪个复用 buffer slot 覆盖。它主要是调试工具，用来观察 scheduler 分配后的 buffer 复用关系。

### `update_dump_tensors(...)`

根据 `dump_all_tensors` 或指定的 `dump_ids` 计算运行时需要 dump 的中间 tensor 集合。它会跳过输入、输出以及内部常量 tensor，并能处理 SubGraphNode 中 real node 的 layer id 映射。

### `check_graph_weights(GraphPtr m_graph, bool error = true)`

检查图中所有节点的 constant tensor 是否真的带有数据指针。构建 AIPU bin 时权重/常量数据是必须的，如果缺失会报错并阻止继续构建。

### `allocate_output_shape(GraphPtr g)`

在动态 shape 模式下，为需要输出 shape buffer 的节点计算 shape buffer 的 offset 和 size。它会给相关节点设置 `output_shape_offset` 和 `output_shape_size`，供后续 RO/临时参数区生成使用。

### `parse_input_shape(GraphPtr g, const std::string &opt)`

解析 `input_shape`、`max_input_shape`、`default_input_shape` 等配置项，生成输入 shape 列表。它支持直接写 shape，也支持按符号名赋值，例如把动态维度符号替换成具体数值。

### `workspace_allocate(GraphPtr g)`

递归设置节点和子图的 workspace tensor 绑定关系。它把子图 workspace 挂到上层 graph 的 workspace 基对象上，保证后续 workspace size 和 memory offset 能统一管理。

### `gen_const_layer_map(GraphPtr g, std::map<void *, int> &const_layer_map)`

建立常量数据指针到 layer id 的映射。它递归遍历未被 plugin 覆盖的 subgraph，并把每个常量 tensor 的底层数据地址关联到所属节点 layer id，供常量信息 dump 使用。

### `dump_const_info_csv(...)`

把常量 tensor 的名称、shape、layer id、size、dtype、offset、buffer id、bss id 写到 `constant_info.csv`。它依赖 `gen_const_layer_map()` 和 buffer map，主要服务调试和常量内存布局分析。

### `update_node_output_for_dynamic_gm(GraphPtr g, MemoryObjectPtr gm)`

在 dynamic GM 优化后，把部分写出到 GM 的节点输出临时替换成指向 GM 区域的新 tensor。它记录原始输出 tensor，方便后续 `restore_node_output_for_dynamic_gm()` 在汇编后恢复图连接。

### `restore_node_output_for_dynamic_gm(...)`

恢复 `update_node_output_for_dynamic_gm()` 临时替换过的节点输出，并修正 RO 参数中涉及的 tensor 引用。它用于 dynamic GM 汇编阶段之后，把 graph 关系还原到正常状态。

### `check_graph_input_shape(GraphPtr g)`

检查用户配置的 `input_shape`、`max_input_shape`、`default_input_shape` 是否与图输入一致。若维度数量、shape rank 或取值范围不合法，会发出 warning 并移除对应配置，避免错误 shape 继续影响构建。

### `reduce_constant(const std::vector<std::vector<NodePtr>> &levels)`

根据常量 hash 合并重复常量数据，减少常量内存占用。它会跳过 LoRA 相关常量，并通过共享数据指针和累加 footprint 的方式复用相同常量。

### `check_symbol_shape(GraphPtr graph)`

在生成参数前检查节点输出的符号 shape 是否能正确求值并匹配实际 shape。动态 shape 模式下如果符号表达式无法求值，会输出更详细的 warning，辅助定位 shape 推导问题。

### `update_params(GraphPtr graph, BuilderParams &params)`

对 RO 或 descriptor 参数中的符号表达式进行求值，并把表达式参数更新成最终数值或 tensor offset。动态 shape/default shape 场景下，它也会保留无法完全静态化的 `expr_offset`，供运行时或后续流程使用。

### `delete_unused_const(...)`

在单个节点的 RO/descriptor 生成后，删除没有被实际引用的常量项。这样可以减少图上保留的无用 constant，避免后续常量数据打包时带入不需要的数据。

## `GraphBuilder` 成员函数

### `GraphBuilder::GraphBuilder(GraphPtr graph)`

用已有 graph 创建 `GraphBuilder`，并初始化内部 `AIPUBin` 对象。`m_origin_graph` 保存原始图，后续 `build_graph()` 会克隆或复用它生成真正参与构建的 `m_graph`。

### `GraphBuilder::GraphBuilder(std::string file)`

从文件路径构造 `GraphBuilder` 的占位构造函数，目前只初始化 `AIPUBin`。这个构造函数没有在此处解析文件内容，实际加载 bin 由 `load()` 系列函数完成。

### `GraphBuilder::get_text_size()`

返回当前 `m_aipubin` 中 text section 的二进制大小。若 `m_aipubin` 为空，则返回 0。

### `GraphBuilder::update_subsec(BssSubSectionDescriptor &subsec, TensorPtr tensor)`

把 tensor 的 dtype、scale、zp 和 RO/descriptor 引用位置写入 BSS subsection 描述符。它依赖 `m_tensor_map`，用于描述一个 tensor 在最终 bin 中与参数表的关联。

### `GraphBuilder::desc_add_subsec(...)`

如果某个 tensor 属于指定类别列表，就把它作为对应类型的 BSS subsection 加入 buffer descriptor。常用于输入、输出、dump、profile、printf fifo、error code 等特殊 tensor 标记。

### `GraphBuilder::build_bss_section(...)`

根据 data slots、constant slots、buffer map、tensor map 和特殊 tensor 集合生成一个 BSS section。它为 feature map buffer 和 constant buffer 分别写入描述符，并处理输入/输出/dump/profile/cross-submodel 等 runtime 可见的 subsection 类型。

### `GraphBuilder::build_ro(BinSection &ro_bin, std::vector<ReadOnlyParam> &ro)`

把 `ReadOnlyParam` 列表打包成连续的 32-bit RO 或 descriptor 二进制段。tensor 参数会写入 tensor offset，立即数参数会写入 value。

### `GraphBuilder::build_remap(...)`

根据 RO 和 descriptor 中引用的 tensor，生成 remap table。它记录 RO/descriptor 项指向 RO base 或 descriptor base 的相对偏移，用于 runtime 装载后修正地址。

### `GraphBuilder::build_text()`

调用 `LinkerV1` 把生成的代码片段链接成最终 text section。它设置入口点、header type，并把链接结果写入 `m_aipubin->text`。

### `GraphBuilder::build_data(...)`

把常量 tensor 数据打包成 data section，并记录每个常量的相对 offset。它根据常量 memory object 的 base/size 计算整体数据 buffer 范围，然后把真实常量数据拷贝到对应位置。

### `GraphBuilder::load(const std::string &filename)`

从指定文件读取 AIPU bin。它打开文件流后转调 `load(std::istream &is)`，最后关闭文件并返回解析出的 `AIPUBin`。

### `GraphBuilder::load(std::istream &is)`

从输入流中加载一个 `AIPUBin` 对象。实际解析逻辑在 `AIPUBin::load()` 中完成。

### `GraphBuilder::set_build_args(const std::string key)`

设置构建开关，特殊处理包含 `profile` 的选项。profile 选项会启用 profiler 并注册默认事件，普通选项则直接写入 `BuilderConfig` 为 true。

### `GraphBuilder::build(...)`

对外的构建入口：调用 `build_aipu()` 完成内部构建，然后把 `m_aipubin` 写到目标文件。返回值同时要求构建流程成功且文件写入没有失败。

### `GraphBuilder::write(std::ostream &ofs)`

把当前 `m_aipubin` 写入给定输出流。它不负责打开或关闭文件，只负责序列化 bin 内容。

### `GraphBuilder::write(const std::string &filename)`

打开指定文件并把当前 `m_aipubin` 写入文件。它是 `write(std::ostream &ofs)` 的文件名版本。

### `GraphBuilder::show_section_size(std::string str, uint64_t size, utils::DataUnit unit)`

按十六进制字节数和人类可读单位打印 section 大小。它用于 `show_memory_info()` 输出 text、RO、desc、data、BSS 等区域的大小。

### `GraphBuilder::partition_graph()`

当前是空实现，是 graph partition 的预留扩展点。调用点在 `build_graph()` 的 topological sort 后，但目前不会改变图。

### `GraphBuilder::build_graph(...)`

从原始 graph 克隆或复用出构建用 graph，执行拓扑排序、输出 shape 分配、完整 MC optimize、性能估计、load balance、workspace 绑定和节点重排。它返回完成 lowering/plugin/scheduling 前置准备后的 graph，供 `build_aipu()` 继续做内存分配和二进制生成。

### `GraphBuilder::build_aipu(...)`

这是 MC 路径生成 AIPU bin 的主流程。它检查权重和动态 shape 配置，调用 `build_graph()`，选择 dump/keep tensor，做内存调度、GM 优化、assembly、RO/descriptor/data/BSS/header 生成，最后清理构建期临时属性。

### `GraphBuilder::show_memory_info()`

打印最终 bin 各 section 和运行时内存的大小统计。它汇总 text、RO、desc、data、extra data、BSS、stack、workspace 等信息，主要用于构建日志和内存分析。

### `GraphBuilder::assembly()`

遍历构建后的 graph，为每个节点生成 descriptor、code、RO 参数和 kernel 列表，并维护 CodeTree。它还设置 ro/descriptor offset、operator id、profiling 标记、descriptor 链关系，并处理不支持节点的错误日志。

### `GraphBuilder::get_build_text() const`

返回当前构建出的 text section 指针和大小。若 `m_aipubin` 不存在，则返回空指针和 0。

### `GraphBuilder::~GraphBuilder()`

释放 `m_aipubin`。其他大多数构建期对象由智能指针或容器管理。
