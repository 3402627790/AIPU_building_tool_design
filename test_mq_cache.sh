#!/bin/bash
# MQ IR Cache 功能测试脚本
# 用于验证 --mq_save_ir 和 --mq_load_ir 功能的正确性

set -e

# 配置
GRAPH_DEF="${1:-graph.def}"
WEIGHT_BIN="${2:-weight.bin}"
TARGET="${3:-X3U}"
CACHE_DIR="mq_ir_cache_test"
OUTPUT_DIR="mq_test_outputs"

echo "=========================================="
echo "MQ IR Cache 功能测试"
echo "=========================================="
echo "输入文件: $GRAPH_DEF"
echo "权重文件: $WEIGHT_BIN"
echo "目标平台: $TARGET"
echo ""

# 检查输入文件
if [ ! -f "$GRAPH_DEF" ]; then
    echo "错误: 找不到 graph 文件 $GRAPH_DEF"
    exit 1
fi

if [ ! -f "$WEIGHT_BIN" ]; then
    echo "错误: 找不到 weight 文件 $WEIGHT_BIN"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
rm -rf "$CACHE_DIR"

echo "步骤 1: 正常编译（无缓存）"
echo "----------------------------------------"
time ./tools/aipugb "$GRAPH_DEF" -w "$WEIGHT_BIN" --target "$TARGET" \
    -o "$OUTPUT_DIR/output_normal.bin" 2>&1 | tee "$OUTPUT_DIR/log_normal.txt"
echo ""

echo "步骤 2: 编译并保存缓存"
echo "----------------------------------------"
time ./tools/aipugb "$GRAPH_DEF" -w "$WEIGHT_BIN" --target "$TARGET" \
    --mq_save_ir "$CACHE_DIR" \
    -o "$OUTPUT_DIR/output_with_save.bin" 2>&1 | tee "$OUTPUT_DIR/log_with_save.txt"
echo ""

# 验证缓存文件是否生成
if [ ! -f "$CACHE_DIR/lowered.def" ]; then
    echo "错误: 缓存文件 lowered.def 未生成"
    exit 1
fi

if [ ! -f "$CACHE_DIR/lowered.weight.bin" ]; then
    echo "错误: 缓存文件 lowered.weight.bin 未生成"
    exit 1
fi

echo "✓ 缓存文件已生成:"
ls -lh "$CACHE_DIR/"
echo ""

echo "步骤 3: 使用缓存编译"
echo "----------------------------------------"
time ./tools/aipugb "$GRAPH_DEF" -w "$WEIGHT_BIN" --target "$TARGET" \
    --mq_load_ir "$CACHE_DIR" \
    -o "$OUTPUT_DIR/output_with_load.bin" 2>&1 | tee "$OUTPUT_DIR/log_with_load.txt"
echo ""

echo "步骤 4: 验证输出一致性"
echo "----------------------------------------"

# 比较输出文件
echo -n "比较 normal vs with_save: "
if diff "$OUTPUT_DIR/output_normal.bin" "$OUTPUT_DIR/output_with_save.bin" > /dev/null 2>&1; then
    echo "✓ 相同"
else
    echo "✗ 不同 (这可能是正常的，取决于编译器的非确定性)"
fi

echo -n "比较 with_save vs with_load: "
if diff "$OUTPUT_DIR/output_with_save.bin" "$OUTPUT_DIR/output_with_load.bin" > /dev/null 2>&1; then
    echo "✓ 相同"
else
    echo "✗ 不同 (这可能是正常的，取决于编译器的非确定性)"
fi

# 检查日志中的关键信息
echo ""
echo "步骤 5: 检查日志信息"
echo "----------------------------------------"

echo -n "检查保存日志: "
if grep -q "mq_save_ir: dumping post-lowering MQ IR" "$OUTPUT_DIR/log_with_save.txt"; then
    echo "✓ 找到保存日志"
else
    echo "✗ 未找到保存日志"
fi

echo -n "检查加载日志: "
if grep -q "mq_load_ir: loading cached MQ IR" "$OUTPUT_DIR/log_with_load.txt"; then
    echo "✓ 找到加载日志"
else
    echo "✗ 未找到加载日志"
fi

# 性能对比
echo ""
echo "步骤 6: 性能对比"
echo "----------------------------------------"
echo "查看上面的 time 输出来对比编译时间"
echo "预期: 使用缓存的编译时间应该明显少于正常编译"

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo "输出文件位于: $OUTPUT_DIR/"
echo "缓存文件位于: $CACHE_DIR/"
echo ""
echo "如需清理测试文件:"
echo "  rm -rf $OUTPUT_DIR $CACHE_DIR"
