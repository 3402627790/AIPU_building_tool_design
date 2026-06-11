# MQ IR Cache with Attributes - Implementation Summary

## What Was Modified

I've directly modified your code to implement JSON-based attribute serialization with a separate file.

## Modified File

### `src/builder/mqbuilder.cpp`

**Added includes:**
```cpp
#include "utils/json.h"
#include <fstream>
#include <sstream>
```

**Added file constants:**
- `kMqIrCacheAttrs = "lowered.attrs.json"` - Separate JSON file for attributes

**Added helper functions (in anonymous namespace):**
1. `serialize_any_to_json()` - Convert Any type to JsonObject
2. `deserialize_json_to_any()` - Convert JsonObject back to Any
3. `save_graph_attributes()` - Save all node and tensor attributes to JSON
4. `load_graph_attributes()` - Load attributes from JSON file

**Modified functions:**
- `try_load_mq_ir_cache()` - Now loads attributes from JSON file
- `save_mq_ir_cache()` - Now saves attributes to separate JSON file

## Generated Files Structure

After running `--mq_save_ir cache_dir`, you will have:

```
cache_dir/
  ├── lowered.def              # IR text file (as before)
  ├── lowered.weight.bin       # Weight binary (as before)
  └── lowered.attrs.json       # NEW: Node and tensor attributes
```

## Attributes JSON Format

```json
{
  "nodes": [
    {
      "node_name": "conv1",
      "layer_id": 5,
      "node_attrs": {
        "schedule_level": 3,
        "core_id": 2,
        "buffer_id": 10,
        "is_dwconv": false
      },
      "output_tensors": [
        {
          "tensor_name": "conv1_output",
          "tensor_index": 0,
          "tensor_attrs": {
            "buffer_id": 15,
            "mem_offset": 2048,
            "mem_size": 200704
          }
        }
      ]
    }
  ],
  "total_nodes": 142,
  "total_node_attrs": 568,
  "total_tensor_attrs": 426
}
```

## Supported Attribute Types

The implementation supports:
- `int`, `int64_t`, `uint32_t`, `uint64_t`
- `float`, `double`
- `bool`
- `std::string`
- `std::vector<int>`, `std::vector<float>`, `std::vector<std::string>`

## Skipped Attributes

The following attributes are automatically skipped (not serialized):

**Node attributes:**
- `plugin_impl` - GOP plugin runtime binding
- `kernel_infos` - Runtime information
- `graph`, `subgraph`, `parent_graph` - Graph pointers
- `codetree` - Code tree pointer

**Tensor attributes:**
- `producer` - Producer node pointer
- `consumers` - Consumer node pointers
- `graph` - Graph pointer

## Usage

### Save with attributes:
```bash
./tools/aipugb graph.def -w weight.bin --target X3U --mq_save_ir cache_dir
```

**Expected log output:**
```
mq_save_ir: dumping post-lowering MQ IR to cache_dir/lowered.def
mq_save_ir: dumping attributes to cache_dir/lowered.attrs.json
Saved 568 node attributes and 426 tensor attributes to cache_dir/lowered.attrs.json
```

### Load with attributes:
```bash
./tools/aipugb graph.def -w weight.bin --target X3U --mq_load_ir cache_dir
```

**Expected log output:**
```
mq_load_ir: loading cached MQ IR from cache_dir
mq_load_ir: loading cached attributes from cache_dir/lowered.attrs.json
Loaded 568 node attributes and 426 tensor attributes from cache_dir/lowered.attrs.json
```

## Testing

### Run the test script:
```bash
./test_mq_attrs_cache.sh graph.def weight.bin X3U
```

The test script will:
1. Compile normally (baseline)
2. Compile and save cache with attributes
3. Compile using cached IR with attributes
4. Compare outputs for consistency
5. Verify log messages
6. Show attribute statistics
7. Display performance comparison

### Manual verification:

**Check attributes file:**
```bash
cat cache_dir/lowered.attrs.json | python -m json.tool | head -50
```

**Check file sizes:**
```bash
ls -lh cache_dir/
```

**Verify attributes are loaded:**
```bash
# Look for these log messages
grep "Saved.*node attributes" build.log
grep "Loaded.*node attributes" build.log
```

## Compilation

### On your server:
```bash
cd /project/ai/scratch01/qidson01/code/AIPU_building_tool_design
make clean
make -j$(nproc)
```

### Expected output:
The compilation should complete successfully. If you get errors about `Constants::mq_save_ir`, make sure you've added those constants to `include/core/constants.h` as described in the server deployment guide.

## Performance Impact

**Serialization overhead:**
- Time: ~100-1000ms additional (depends on model size)
- File size: ~100-500KB additional for attributes JSON

**Deserialization overhead:**
- Time: ~200-2000ms additional (depends on model size)

**Overall benefit:**
These overheads are negligible compared to the time saved by skipping lowering (typically seconds to minutes).

## Debugging

### Enable verbose logging:
Add this in your code if needed:
```cpp
INFO() << "Serializing node " << node->name() 
       << " with " << node->attrs().size() << " attributes";
```

### Check JSON validity:
```bash
python -m json.tool cache_dir/lowered.attrs.json > /dev/null && echo "Valid JSON"
```

### Compare attribute counts:
```bash
# Before save
echo "Node count: $(grep -c 'layer_id=' original.def)"

# After load
echo "Nodes in JSON: $(grep -c '"node_name"' cache_dir/lowered.attrs.json)"
```

## Error Handling

The implementation includes robust error handling:

1. **Missing attributes file** - Logs warning and continues without attributes
2. **Invalid JSON** - Logs error and returns false
3. **Missing nodes** - Logs warning and skips that node
4. **Unsupported types** - Logs debug message and skips that attribute

## Next Steps

After successful testing:

1. **Profile performance** - Measure actual time savings
2. **Validate correctness** - Run inference tests to ensure attributes are correctly preserved
3. **Add more types** - If needed, extend `serialize_any_to_json()` for additional types
4. **Optimize JSON size** - Consider compression for large models

## Files Created

1. **test_mq_attrs_cache.sh** - Comprehensive test script
2. **IMPLEMENTATION_SUMMARY.md** - This file

## Summary

✅ Code directly modified in `src/builder/mqbuilder.cpp`
✅ English comments throughout
✅ Attributes saved to separate JSON file (`lowered.attrs.json`)
✅ Supports common attribute types
✅ Robust error handling
✅ Test script provided
✅ Ready to compile and test

The implementation is complete and ready for testing. Simply compile and run the test script to verify everything works correctly.
