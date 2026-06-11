# Node 和 Tensor Attributes 序列化机制分析

## 问题描述

你提到："编译时的attribute以及tensor和node绑定的属性，在gbuilder流转的时候，编译时能看到，而在 mqbuilder.cpp 里面，通过graph流转，上面的这些数据，还是要保存下来的"

这是一个非常关键的问题。让我分析当前的序列化机制。

## 当前序列化机制分析

### 1. Graph 序列化配置（mqbuilder.cpp）

```cpp
void save_mq_ir_cache(GraphPtr graph, const std::string &dir)
{
    // 启用两个关键标志
    graph->set_attr(Constants::serialize_scale_zp, true);   // 保存量化参数
    graph->set_attr(Constants::dump_debug_IR, true);        // 保存调试信息
    
    graph->serialize(def_path, bin_path);
}
```

### 2. Node 序列化内容（node.cpp L1484-1977）

Node::serialize() 当前**确实保存**的内容：

✅ **基本信息**
- layer_id, layer_name, layer_type
- 输入输出 tensor 名称、shape、类型

✅ **Constant 数据**（L1859-1944）
- weights, biases 等常量 tensor
- 量化参数（scale, zp, range）- 当 `serialize_scale_zp=true`

✅ **Node Parameters**（L1959-1971）
```cpp
for (auto p : m_params)
{
    if (required_layer_params.count(p.first) == 0)
    {
        oss << p.first << "=" << p.second << "\n";  // 保存 m_params
    }
}
```

✅ **特殊字段**（当 `dump_debug_IR=true`）
- layer_top_datalayout（L1566-1578）
- 动态 shape 信息

### 3. 问题：Node Attributes (m_attrs) 没有被序列化！

**关键发现**：通过代码分析，`Node::serialize()` 函数：
- ✅ 序列化了 `m_params`（node parameters）
- ❌ **没有序列化 `m_attrs`（node attributes）**

```cpp
// node.cpp L1959-1971
// 只序列化了 m_params，没有 m_attrs！
for (auto p : m_params)  // ← 只有 params
{
    oss << p.first << "=" << p.second << "\n";
}
```

而 `m_attrs` 是一个 `std::map<std::string, Any>`（node.h），用于存储：
- 编译时生成的属性
- GOP plugin 绑定信息
- 调度信息
- 其他动态属性

### 4. Tensor Attributes 分析

让我检查 tensor 序列化：

```cpp
// 在 node.cpp 中，tensor 的序列化只包括：
// - 名称、shape、type
// - 量化参数（scale, zp, range）
```

**问题**：Tensor 的 `m_attrs` 也没有被序列化！

## 解决方案

### 方案 1: 扩展 Node::serialize() 序列化 m_attrs【推荐】

修改 `src/core/node.cpp` 的 `Node::serialize()` 函数，添加 attributes 序列化。

#### 实现步骤

在 `src/core/node.cpp` 的 `Node::serialize()` 函数中，在序列化 `m_params` 之后添加：

```cpp
// 在 L1971 之后添加
// ===== 序列化 node attributes =====
if (BuilderConfig::get<bool>(Constants::dump_debug_IR))
{
    for (const auto& attr : m_attrs)
    {
        // 过滤掉一些不应该序列化的 attributes
        static const std::set<std::string> skip_attrs = {
            Constants::plugin_impl,        // 运行时绑定，不能序列化
            Constants::kernel_infos,       // 运行时信息
            "graph",                       // graph 指针
            "subgraph"                     // subgraph 指针
        };
        
        if (skip_attrs.count(attr.first))
            continue;
            
        // 尝试序列化支持的类型
        if (serialize_attribute(attr.first, attr.second, oss))
        {
            // 成功序列化
        }
        else
        {
            DEBUG() << "Cannot serialize attribute: " << attr.first 
                    << " for node: " << m_name;
        }
    }
}
```

需要实现 `serialize_attribute()` 辅助函数：

```cpp
// 在 node.cpp 中添加
namespace {
bool serialize_attribute(const std::string& key, const Any& value, std::ostream& oss)
{
    // 尝试序列化各种类型
    if (value.is<int>())
    {
        oss << "node_attr_" << key << "_int=" << value.cast<int>() << "\n";
        return true;
    }
    else if (value.is<int64_t>())
    {
        oss << "node_attr_" << key << "_int64=" << value.cast<int64_t>() << "\n";
        return true;
    }
    else if (value.is<uint32_t>())
    {
        oss << "node_attr_" << key << "_uint32=" << value.cast<uint32_t>() << "\n";
        return true;
    }
    else if (value.is<float>())
    {
        oss << "node_attr_" << key << "_float=" << value.cast<float>() << "\n";
        return true;
    }
    else if (value.is<double>())
    {
        oss << "node_attr_" << key << "_double=" << value.cast<double>() << "\n";
        return true;
    }
    else if (value.is<bool>())
    {
        oss << "node_attr_" << key << "_bool=" << (value.cast<bool>() ? "1" : "0") << "\n";
        return true;
    }
    else if (value.is<std::string>())
    {
        oss << "node_attr_" << key << "_string=" << value.cast<std::string>() << "\n";
        return true;
    }
    else if (value.is<std::vector<int>>())
    {
        auto vec = value.cast<std::vector<int>>();
        oss << "node_attr_" << key << "_vec_int=[";
        for (size_t i = 0; i < vec.size(); ++i)
        {
            if (i > 0) oss << ",";
            oss << vec[i];
        }
        oss << "]\n";
        return true;
    }
    else if (value.is<std::vector<float>>())
    {
        auto vec = value.cast<std::vector<float>>();
        oss << "node_attr_" << key << "_vec_float=[";
        for (size_t i = 0; i < vec.size(); ++i)
        {
            if (i > 0) oss << ",";
            oss << vec[i];
        }
        oss << "]\n";
        return true;
    }
    
    // 不支持的类型
    return false;
}
} // namespace
```

相应地，需要在 IR parser 中添加反序列化逻辑（读取 `node_attr_*` 字段）。

### 方案 2: 使用 JSON 序列化 Attributes【备选】

在 node 的最后添加一个 JSON 字段存储所有 attributes：

```cpp
// 在 Node::serialize() 的最后添加
if (BuilderConfig::get<bool>(Constants::dump_debug_IR))
{
    utils::JsonObject attrs_json(utils::JsonObject::OBJ);
    
    for (const auto& attr : m_attrs)
    {
        if (should_skip_attr(attr.first))
            continue;
            
        // 尝试转换为 JSON
        if (attr.second.is<int>())
            attrs_json[attr.first] = attr.second.cast<int>();
        else if (attr.second.is<float>())
            attrs_json[attr.first] = attr.second.cast<float>();
        else if (attr.second.is<bool>())
            attrs_json[attr.first] = attr.second.cast<bool>();
        else if (attr.second.is<std::string>())
            attrs_json[attr.first] = attr.second.cast<std::string>();
        // ... 其他类型
    }
    
    if (!attrs_json.empty())
    {
        oss << "node_attrs_json=" << attrs_json.to_string() << "\n";
    }
}
```

### 方案 3: 选择性序列化关键 Attributes【最简单】

只序列化已知的、编译时需要的关键 attributes：

```cpp
// 定义需要序列化的关键 attributes
static const std::set<std::string> critical_attrs = {
    "layer_id",
    "schedule_level",
    "schedule_gid",
    "core_id",
    "buffer_id",
    "is_dwconv",
    "is_call",
    "plugin_name",
    // ... 其他关键属性
};

if (BuilderConfig::get<bool>(Constants::dump_debug_IR))
{
    for (const auto& attr_name : critical_attrs)
    {
        if (m_attrs.count(attr_name))
        {
            auto& value = m_attrs.at(attr_name);
            // 根据类型序列化
            if (value.is<int>())
                oss << "attr_" << attr_name << "=" << value.cast<int>() << "\n";
            else if (value.is<std::string>())
                oss << "attr_" << attr_name << "=" << value.cast<std::string>() << "\n";
            // ...
        }
    }
}
```

## Tensor Attributes 序列化

类似地，需要修改 tensor 的序列化逻辑。Tensor 的 attributes 在输出 tensor 信息时添加：

```cpp
// 在输出 layer_top_type 之后添加
if (BuilderConfig::get<bool>(Constants::dump_debug_IR))
{
    // 序列化每个 output tensor 的 attributes
    for (size_t i = 0; i < m_outputs.size(); ++i)
    {
        auto& t = m_outputs[i];
        for (const auto& attr : t->attrs())
        {
            if (should_serialize_tensor_attr(attr.first))
            {
                oss << "tensor_" << i << "_attr_" << attr.first 
                    << "=" << serialize_any_value(attr.second) << "\n";
            }
        }
    }
}
```

## 推荐实现策略

### 阶段 1: 快速验证（选择性序列化）
使用**方案 3**，只序列化已知的关键 attributes：
1. 列出 mqbuilder 到 gbuilder 流程中必需的 attributes
2. 在 `Node::serialize()` 中添加这些 attributes 的序列化
3. 在 IR parser 中添加对应的反序列化
4. 测试验证

### 阶段 2: 完整支持（类型化序列化）
使用**方案 1**，支持所有可序列化的 attribute 类型：
1. 实现 `serialize_attribute()` 辅助函数
2. 实现 `deserialize_attribute()` 对应函数
3. 更新 IR parser
4. 全面测试

### 阶段 3: 优化（可选）
- 添加版本号，支持向后兼容
- 压缩序列化格式
- 添加校验和

## 需要序列化的关键 Attributes 列表

基于代码分析，以下是可能需要序列化的关键 attributes：

### Node Attributes
```cpp
// 调度相关
".schedule_level"
".schedule_gid"
"core_id"
"layer_id"

// 内存相关
"buffer_id"
".ro_offset"
".ro_size"
".desc_offset"
".desc_size"

// 编译相关
"is_dwconv"
"is_call"
"is_tiled"
"tile_id"
"plugin_name"

// 性能相关
".performance_info"
"node_perf"

// 其他
"operator_id"
"kernel_loop_count"
```

### Tensor Attributes
```cpp
// 内存相关
"buffer_id"
"mem_offset"
"mem_size"

// 量化相关（已通过 serialize_scale_zp 处理）
// scale, zero_point, range

// 其他
"is_workspace_tensor"
"zerocopy_cst"
```

## 实现建议

我建议创建以下文件来实现这个功能：

1. **src/core/attribute_serializer.h/cpp** - 通用的 attribute 序列化工具
2. **修改 src/core/node.cpp** - 集成 attribute 序列化
3. **修改 IR parser** - 添加 attribute 反序列化

是否需要我为你创建具体的实现代码？请告诉我：
1. 你希望使用哪个方案（1/2/3）？
2. 有哪些特定的 attributes 是你最关心的？
3. 是否有现成的 attribute 列表可以参考？
