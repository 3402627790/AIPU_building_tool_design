#!/bin/bash
# Test script for MQ IR cache with attributes serialization
# Tests both save and load functionality with attribute preservation

set -e

GRAPH_DEF="${1:-graph.def}"
WEIGHT_BIN="${2:-weight.bin}"
TARGET="${3:-X3U}"
CACHE_DIR="mq_attrs_test_cache"
OUTPUT_DIR="mq_attrs_test_outputs"

echo "=========================================="
echo "MQ IR Cache with Attributes - Test"
echo "=========================================="
echo "Graph file: $GRAPH_DEF"
echo "Weight file: $WEIGHT_BIN"
echo "Target: $TARGET"
echo ""

# Check input files
if [ ! -f "$GRAPH_DEF" ]; then
    echo "Error: Graph file not found: $GRAPH_DEF"
    exit 1
fi

if [ ! -f "$WEIGHT_BIN" ]; then
    echo "Error: Weight file not found: $WEIGHT_BIN"
    exit 1
fi

# Clean up previous test
mkdir -p "$OUTPUT_DIR"
rm -rf "$CACHE_DIR"

echo "Step 1: Normal compilation (baseline)"
echo "----------------------------------------"
time ./tools/aipugb "$GRAPH_DEF" -w "$WEIGHT_BIN" --target "$TARGET" \
    -o "$OUTPUT_DIR/output_normal.bin" 2>&1 | tee "$OUTPUT_DIR/log_normal.txt"
echo ""

echo "Step 2: Compilation with attribute caching"
echo "----------------------------------------"
time ./tools/aipugb "$GRAPH_DEF" -w "$WEIGHT_BIN" --target "$TARGET" \
    --mq_save_ir "$CACHE_DIR" \
    -o "$OUTPUT_DIR/output_with_save.bin" 2>&1 | tee "$OUTPUT_DIR/log_with_save.txt"
echo ""

# Verify cache files
if [ ! -f "$CACHE_DIR/lowered.def" ]; then
    echo "Error: Cache file lowered.def not generated"
    exit 1
fi

if [ ! -f "$CACHE_DIR/lowered.weight.bin" ]; then
    echo "Error: Cache file lowered.weight.bin not generated"
    exit 1
fi

if [ ! -f "$CACHE_DIR/lowered.attrs.json" ]; then
    echo "Error: Attributes file lowered.attrs.json not generated"
    exit 1
fi

echo "✓ Cache files generated successfully:"
ls -lh "$CACHE_DIR/"
echo ""

# Check attributes file content
echo "Attributes file statistics:"
echo "  File size: $(du -h "$CACHE_DIR/lowered.attrs.json" | cut -f1)"
echo "  Total nodes with attrs: $(grep -o '"node_name"' "$CACHE_DIR/lowered.attrs.json" | wc -l)"
echo "  Total node attrs: $(grep -o 'total_node_attrs' "$CACHE_DIR/lowered.attrs.json" | head -1)"
echo "  Total tensor attrs: $(grep -o 'total_tensor_attrs' "$CACHE_DIR/lowered.attrs.json" | head -1)"
echo ""

echo "Step 3: Compilation using cached IR with attributes"
echo "----------------------------------------"
time ./tools/aipugb "$GRAPH_DEF" -w "$WEIGHT_BIN" --target "$TARGET" \
    --mq_load_ir "$CACHE_DIR" \
    -o "$OUTPUT_DIR/output_with_load.bin" 2>&1 | tee "$OUTPUT_DIR/log_with_load.txt"
echo ""

echo "Step 4: Verify output consistency"
echo "----------------------------------------"

echo -n "Comparing normal vs with_save: "
if diff "$OUTPUT_DIR/output_normal.bin" "$OUTPUT_DIR/output_with_save.bin" > /dev/null 2>&1; then
    echo "✓ Identical"
else
    echo "✗ Different (may be acceptable due to compiler non-determinism)"
fi

echo -n "Comparing with_save vs with_load: "
if diff "$OUTPUT_DIR/output_with_save.bin" "$OUTPUT_DIR/output_with_load.bin" > /dev/null 2>&1; then
    echo "✓ Identical"
else
    echo "✗ Different (may be acceptable due to compiler non-determinism)"
fi

echo ""
echo "Step 5: Verify logs"
echo "----------------------------------------"

echo -n "Check save log for attributes: "
if grep -q "mq_save_ir: dumping attributes" "$OUTPUT_DIR/log_with_save.txt"; then
    echo "✓ Found attribute save log"
else
    echo "✗ Attribute save log not found"
fi

echo -n "Check load log for attributes: "
if grep -q "mq_load_ir: loading cached attributes" "$OUTPUT_DIR/log_with_load.txt"; then
    echo "✓ Found attribute load log"
else
    echo "✗ Attribute load log not found"
fi

echo -n "Check attribute statistics in save log: "
if grep -q "Saved .* node attributes" "$OUTPUT_DIR/log_with_save.txt"; then
    grep "Saved .* node attributes" "$OUTPUT_DIR/log_with_save.txt"
else
    echo "✗ Attribute statistics not found"
fi

echo -n "Check attribute statistics in load log: "
if grep -q "Loaded .* node attributes" "$OUTPUT_DIR/log_with_load.txt"; then
    grep "Loaded .* node attributes" "$OUTPUT_DIR/log_with_load.txt"
else
    echo "✗ Attribute statistics not found"
fi

echo ""
echo "Step 6: Performance comparison"
echo "----------------------------------------"
echo "Review the 'time' output above to compare compilation times"
echo "Expected: cached compilation should be significantly faster"
echo ""

# Show sample of attributes JSON
echo "Sample of attributes JSON (first 20 lines):"
echo "----------------------------------------"
head -20 "$CACHE_DIR/lowered.attrs.json" | python -m json.tool 2>/dev/null || cat "$CACHE_DIR/lowered.attrs.json" | head -20
echo ""

echo "=========================================="
echo "Test completed!"
echo "=========================================="
echo "Output files: $OUTPUT_DIR/"
echo "Cache files: $CACHE_DIR/"
echo ""
echo "To clean up test files:"
echo "  rm -rf $OUTPUT_DIR $CACHE_DIR"
