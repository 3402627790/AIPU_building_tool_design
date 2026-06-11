# 方案 2: JSON 序列化 - 完整实现指南

## 概述

使用 JSON 格式序列化所有 node 和 tensor 的 attributes，实现简单、扩展性好。

## 实现文件

我已经为你创建了以下代码文件：

1. **node_serialize_code.cpp** - Node attributes 序列化代码
2. **node_deserialize_code.cpp** - Node attributes 反序列化代码  
3. **tensor_serialize_code.cpp** - Tensor attributes 序列化代码
4. **attribute_serializer.patch** - Git patch 格式

## 快速实施步骤

### 步骤 1: 修改 src/core/node.cpp

#### 1.1 添加 include（文件开头）

```cpp
#include "utils/json.h"
```

在文件开头（约 L30）添加上面这行。

#### 1.2 添加 Node Attributes 序列化

在 `Node::serialize()` 函数中，找到约 **L1944**（在序列化 constants 之后），插入 **node_serialize_code.cpp** 的全部代码。

**关键位置标识**：
```cpp
// 在这段代码之后插入
if (BuilderConfig::get<bool>(Constants::serialize_scale_zp))
{
    // ... 序列化 constant 的 scale/zp
}
// ← 在这里插入 node attributes 序列化代码

const std::set<std::string> required_layer_params = {
```

#### 1.3 添加 Tensor Attributes 序列化

在 `Node::serialize()` 函数中，找到约 **L1565**（输出 layer_top_type 之后），插入 **tensor_serialize_code.cpp** 的全部代码。

**关键位置标识**：
```cpp
oss << "]";  // layer_top_type 的结尾
// ← 在这里插入 tensor attributes 序列化代码

if (BuilderConfig::get<bool>(Constants::serialize_scale_zp))
{
```

### 步骤 2: 修改 IR Parser（反序列化）

需要找到 IR parser 的实现文件，通常在 `src/core/` 或 `src/parser/` 目录下。

#### 2.1 添加解析函数

将 **node_deserialize_code.cpp** 中的 `parse_node_attributes()` 函数添加到 parser 文件中。

#### 2.2 在解析循环中调用

在解析 node 的主循环中，添加对 `node_attrs` 字段的识别：

```cpp
// 在 node 解析循环中
if (line.find("node_attrs=") == 0)
{
    std::string json_str = line.substr(11); // 跳过 "node_attrs="
    parse_node_attributes(current_node, json_str);
}
```

#### 2.3 解析 Tensor Attributes

类似地，添加对 `layer_top_N_attrs` 的解析：

```cpp
// 匹配 layer_top_0_attrs=, layer_top_1_attrs= 等
if (line.find("layer_top_") == 0 && line.find("_attrs=") != std::string::npos)
{
    // 提取索引
    size_t pos1 = line.find("_");
    size_t pos2 = line.find("_attrs=");
    int idx = std::stoi(line.substr(pos1 + 1, pos2 - pos1 - 1));
    
    // 提取 JSON
    std::string json_str = line.substr(pos2 + 7);
    
    // 解析并设置到对应的 tensor
    if (idx < current_node->outputs().size())
    {
        parse_tensor_attributes(current_node->outputs()[idx], json_str);
    }
}
```

### 步骤 3: 测试

#### 3.1 编译

```bash
cd /project/ai/scratch01/qidson01/code/AIPU_building_tool_design
make -j$(nproc)
```

#### 3.2 测试序列化

```bash
./tools/aipugb graph.def -w weight.bin --target X3U --mq_save_ir test_cache
```

检查生成的 `test_cache/lowered.def` 文件，应该能看到：

```
layer_id=5
layer_name=conv1
layer_type=Convolution
# ... 其他字段 ...
node_attrs={"layer_id":5,"schedule_level":3,"core_id":2,"buffer_id":10}
layer_top_0_attrs={"buffer_id":15,"mem_offset":2048}
```

#### 3.3 测试反序列化

```bash
./tools/aipugb graph.def -w weight.bin --target X3U --mq_load_ir test_cache
```

在编译过程中添加日志验证 attributes 是否正确加载：

```cpp
// 在反序列化后添加验证
INFO() << "Node " << node->name() << " has " << node->attrs().size() << " attributes";
for (const auto& attr : node->attrs())
{
    DEBUG() << "  attr: " << attr.first;
}
```

## 支持的数据类型

当前实现支持以下 attribute 类型：

✅ **基本类型**
- `int`, `int64_t`, `uint32_t`, `uint64_t`
- `float`, `double`
- `bool`
- `std::string`

✅ **容器类型**
- `std::vector<int>`
- `std::vector<float>`
- `std::vector<std::string>`

❌ **不支持的类型**
- 指针类型（会被跳过）
- 复杂对象（会被跳过）
- 嵌套容器

## 跳过的 Attributes

以下 attributes 会被自动跳过，不会序列化：

### Node Attributes
- `plugin_impl` - GOP plugin 运行时绑定
- `kernel_infos` - 运行时信息
- `graph` - graph 指针
- `subgraph` - subgraph 指针
- `parent_graph` - 父 graph 指针
- `codetree` - 代码树指针

### Tensor Attributes  
- `producer` - 生产者 node 指针
- `consumers` - 消费者 node 指针列表
- `graph` - graph 指针

## 输出格式示例

### Node Attributes

```
layer_id=5
layer_name=conv1
layer_type=Convolution
# ... 
node_attrs={"layer_id":5,"schedule_level":3,"core_id":2,"buffer_id":10,"is_dwconv":false,"plugin_name":"conv_opt","tile_config":[1,2,3,4]}
```

### Tensor Attributes

```
layer_top=[output1]
layer_top_shape=[[1,64,56,56]]
layer_top_type=[UINT8]
layer_top_0_attrs={"buffer_id":15,"mem_offset":2048,"mem_size":200704,"is_workspace_tensor":false}
```

## 扩展支持更多类型

如果需要支持其他类型，在序列化代码中添加：

```cpp
else if (value.is<YourType>())
{
    // 转换为 JSON 支持的类型
    attrs_json[key] = convert_to_json(value.cast<YourType>());
    has_attrs = true;
}
```

在反序列化代码中相应添加：

```cpp
case utils::JsonObject::YOUR_TYPE:
    node->set_attr(key, aipubt::Any(convert_from_json(value)));
    break;
```

## 调试技巧

### 1. 查看序列化的 attributes

```bash
# 查看生成的 IR 文件
cat test_cache/lowered.def | grep "node_attrs"

# 格式化 JSON 便于阅读
cat test_cache/lowered.def | grep "node_attrs" | sed 's/node_attrs=//' | python -m json.tool
```

### 2. 验证 attributes 是否完整

在序列化前后添加日志：

```cpp
// 序列化前
INFO() << "Serializing node " << m_name << " with " << m_attrs.size() << " attributes";

// 序列化后
INFO() << "Serialized " << attrs_json.to_map().size() << " attributes for node " << m_name;
```

### 3. 对比序列化前后

```cpp
// 保存原始 attributes
std::map<std::string, Any> original_attrs = node->attrs();

// 加载后对比
for (const auto& attr : original_attrs)
{
    if (!node->has_attr(attr.first))
    {
        ERROR() << "Lost attribute: " << attr.first;
    }
}
```

## 常见问题

### Q1: 编译错误：JsonObject 未定义

**A**: 确保添加了 `#include "utils/json.h"`

### Q2: 某些 attributes 没有被序列化

**A**: 检查该 attribute 的类型是否支持，或者是否在 `skip_attrs` 列表中

### Q3: 反序列化后 attributes 值不正确

**A**: 检查类型转换是否正确，特别是 int/int64_t/uint32_t 之间的转换

### Q4: JSON 字符串过长导致解析失败

**A**: JsonObject 支持的字符串长度有限制，如果单个 node 的 attributes 过多，考虑分批序列化

## 性能影响

- **序列化**: 每个 node 增加约 0.1-1ms（取决于 attributes 数量）
- **文件大小**: 每个 node 约增加 100-500 字节
- **反序列化**: 每个 node 增加约 0.2-2ms

对于大型模型（1000+ nodes），预计：
- 序列化增加时间: 100-1000ms
- 文件大小增加: 100-500KB
- 反序列化增加时间: 200-2000ms

这些开销相比整体的 lowering 时间（通常数秒到数分钟）是可以接受的。

## 下一步

完成基本实现后，可以考虑以下优化：

1. **压缩 JSON** - 使用更紧凑的 JSON 格式
2. **二进制序列化** - 对于大量 attributes，使用二进制格式更高效
3. **增量序列化** - 只序列化改变的 attributes
4. **版本控制** - 添加版本号支持向后兼容

## 需要帮助？

如果在实现过程中遇到问题，请提供：
1. 编译错误的完整日志
2. 你的修改位置（文件和行号）
3. 测试时的输入文件和错误信息
