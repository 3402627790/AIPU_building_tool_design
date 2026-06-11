# MQ IR Cache 功能快速参考

## 基本用法

### 保存 IR 缓存
```bash
./tools/aipugb graph.def -w weight.bin --target X3U --mq_save_ir save_ir
```
生成文件:
- `save_ir/lowered.def`
- `save_ir/lowered.weight.bin`

### 加载 IR 缓存
```bash
./tools/aipugb graph.def -w weight.bin --target X3U --mq_load_ir save_ir
```

### 运行测试脚本
```bash
# 使用默认参数
./test_mq_cache.sh graph.def weight.bin X3U

# 或指定自己的参数
./test_mq_cache.sh my_model.def my_weight.bin X2_1204
```

## 工作原理

```
正常编译流程:
  Input IR → Parse → Lowering → Scheduling → Code Gen → Binary
  
使用缓存流程:
  Cached IR → Re-apply plugins → Scheduling → Code Gen → Binary
  (跳过 Parse 和 Lowering，节省大量时间)
```

## 保存的内容

✅ Graph 结构 (nodes, edges)
✅ Tensor shapes & data types  
✅ Node attributes
✅ Tensor attributes
✅ 量化参数 (scale, zero_point)
✅ Constant 数据
✅ 编译时绑定属性

❌ GOP plugin 运行时绑定 (会自动重新应用)

## 何时重新生成缓存

当以下内容改变时需要重新生成:
- 模型结构 (graph.def)
- Weight 数据 (weight.bin)
- 目标平台 (--target)
- BuilderTool 版本
- Plugin 版本

## 故障排除

### 问题: 缓存未找到
```
WARN: mq_load_ir: cache files not found in save_ir
```
**解决**: 先运行 `--mq_save_ir` 生成缓存

### 问题: 缓存加载失败
```
ERROR: mq_load_ir: failed to parse cached IR
```
**解决**: 删除缓存目录，重新生成

### 问题: 输出不一致
**原因**: 可能是编译器的非确定性行为
**验证**: 检查模型在推理时的输出是否一致

## 实现位置

- 核心逻辑: `src/builder/mqbuilder.cpp` (L44-111, L133-185)
- 命令行参数: `include/tools/common_options.h` (L293-294)
- 配置常量: `include/core/constants.h` (L189-190)
- 参数处理: `src/builder/tools.cpp` (L760-767)

## 常见场景

### 场景 1: 快速迭代调试
```bash
# 第一次: 生成缓存
./tools/aipugb model.def -w weight.bin --target X3U --mq_save_ir cache

# 后续: 快速重编译
./tools/aipugb model.def -w weight.bin --target X3U --mq_load_ir cache
```

### 场景 2: CI/CD 流程
```bash
# 在 CI 中缓存 lowering 结果
- name: Build with cache
  run: |
    if [ -d ir_cache ]; then
      ./tools/aipugb model.def -w weight.bin --target X3U --mq_load_ir ir_cache
    else
      ./tools/aipugb model.def -w weight.bin --target X3U --mq_save_ir ir_cache
    fi
```

### 场景 3: 不同配置的快速测试
```bash
# 一次 lowering，多次编译测试不同的调度策略
./tools/aipugb model.def -w weight.bin --target X3U --mq_save_ir cache

./tools/aipugb model.def -w weight.bin --target X3U --mq_load_ir cache --load_balance_strategy 1
./tools/aipugb model.def -w weight.bin --target X3U --mq_load_ir cache --load_balance_strategy 2
```

## 性能预期

- 简单模型: 节省 20-30% 编译时间
- 中等模型: 节省 30-50% 编译时间  
- 复杂模型: 节省 40-60% 编译时间
- 超大模型: 可能节省更多

实际加速比取决于 lowering 阶段占总编译时间的比例。
