---
name: add-aipu-op
description: Add a new operation to AIPUBuilder with full pipeline support (IR, quantization, lowering, builder)
parameters:
  - name: op_name
    description: "Name of the operation (e.g., 'Round', 'Trunc', 'Sin')"
    required: true
  - name: op_type
    description: "Type of operation: compute, activation, reduction, transform, or other"
    required: true
  - name: inputs
    description: "Number and type of inputs (e.g., '1 tensor', '2 tensors + 1 scalar')"
    required: true
  - name: outputs
    description: "Number and type of outputs (e.g., '1 tensor', '2 tensors')"
    required: true
  - name: attributes
    description: "Optional attributes (e.g., 'axis, keepdims')"
    required: false
---

# Add AIPU Operation: {{op_name}}

Follow these steps to add a new operation to AIPUBuilder pipeline:

## Step 1: Define Op in Core IR

**Location**: `src/core/ops/` or `src/ops/`

Create `{{op_name}}.h` and `{{op_name}}.cpp`:

```cpp
// {{op_name}}.h
#pragma once
#include "core.h"

class {{op_name}}Op : public OpBase {
public:
  {{op_name}}Op(const std::string &name, const std::vector<Tensor*> &inputs);
  
  // Shape inference - compute output shape from input shapes
  std::vector<Shape> inferShape() const override;
  
  // Type inference - compute output type from input types
  DataType inferType() const override;
  
  // Compute operation (for reference/testing)
  void compute(Context &ctx) override;
  
  // Serialization
  void serialize(Serializer &s) const override;
  void deserialize(Deserializer &d) override;
  
private:
  // Attributes: {{attributes}}
};
```

## Step 2: Register in OpType Enum

**Location**: `src/core/internal_optype.h`

Add to enum:
```cpp
enum class OpType {
  // ... existing ops
  {{op_name | upper}},
};
```

Update op factory to handle the new type.

## Step 3: Add Python Quantization Support

**Location**: `python/src/AIPUBuilder/Optimizer/ops/`

Create `{{op_name}}.py`:

```python
from AIPUBuilder.Optimizer.framework import OpQuantizer, register_quantizer
import numpy as np

@register_quantizer("{{op_name}}")
class {{op_name}}Quantizer(OpQuantizer):
    def quantize(self, node, context):
        """
        Quantize {{op_name}} operation.
        
        Inputs: {{inputs}}
        Outputs: {{outputs}}
        Attributes: {{attributes}}
        """
        # Get input quantization params
        inp_scale = context.get_scale(node.inputs[0])
        inp_zp = context.get_zero_point(node.inputs[0])
        
        # For {{op_type}} operations, typically:
        # - Compute ops: may need requantization
        # - Activation ops: preserve scale or use LUT
        # - Reduction ops: handle scale changes
        # - Transform ops: preserve quantization params
        
        # Compute output scale and zero point
        out_scale = inp_scale  # Adjust based on op semantics
        out_zp = inp_zp        # Adjust based on op semantics
        
        # Set output quantization info
        context.set_scale(node.outputs[0], out_scale)
        context.set_zero_point(node.outputs[0], out_zp)
        
    def calibrate(self, node, statistics):
        """Compute quantization params from calibration statistics."""
        f_min = statistics.get_min(node.outputs[0])
        f_max = statistics.get_max(node.outputs[0])
        
        # Use symmetric or asymmetric quantization
        return self.compute_quant_params(f_min, f_max)
```

## Step 4: Implement MLIR Lowering (if needed)

**Location**: `src/mlir/lib/aipugc/`

If the op needs special MLIR lowering:

```cpp
struct {{op_name}}Lowering : public OpConversionPattern<{{op_name}}Op> {
  using OpConversionPattern::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      {{op_name}}Op op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    // Lower to AIPU-specific operations
    // or decompose into primitive ops
    
    return success();
  }
};
```

## Step 5: Add Builder Support

**Location**: `src/builder/ops/` or `src/builder/lowering/`

Implement lowering to AIPU instructions:

```cpp
class {{op_name}}Builder {
public:
  void build({{op_name}}Op *op, BuilderContext &ctx) {
    auto input = ctx.getTensor(op->input(0));
    auto output = ctx.allocTensor(op->output(0));
    
    // Generate AIPU instructions for {{op_name}}
    // Consider:
    // - SRAM allocation
    // - Instruction scheduling
    // - Data dependencies
    
    ctx.emit(InstructionType::{{op_name | upper}}, input, output);
  }
};
```

## Step 6: Add Tests

**Location**: `test/core/ops/` or `test/builder/`

Create `test_{{op_name}}.cpp`:

```cpp
#include "test_core.h"
#include "ops/{{op_name}}.h"

TEST({{op_name}}OpTest, BasicFunctionality) {
  // Test shape inference
  // Test type inference
  // Test compute
}

TEST({{op_name}}OpTest, Quantization) {
  // Test quantized version
}

TEST({{op_name}}OpTest, Lowering) {
  // Test MLIR lowering if applicable
}
```

## Step 7: Update Documentation

Add to op documentation:
- Input/output specifications
- Quantization behavior
- Supported AIPU targets
- Performance characteristics

## Step 8: Validate

```bash
# Build and test
mkdir -p build && cd build
cmake .. && make

# Run tests
./test_{{op_name}}

# Test full pipeline with a model containing {{op_name}}
aipuparser -c test_config.cfg
aipuaqt --cfg test_config.cfg --graph model.def --bin model.bin2
aipugb model.def -w model.bin2 -o output.bin
aipurun model.def -w model.bin2 -i input.bin -o output.bin

# Verify output correctness
```

## Checklist

- [ ] Op defined in `src/core/ops/` with shape/type inference
- [ ] OpType enum updated in `internal_optype.h`
- [ ] Op factory updated to create the op
- [ ] Python quantizer added in `python/src/AIPUBuilder/Optimizer/ops/`
- [ ] MLIR lowering implemented (if needed)
- [ ] Builder support added for code generation
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] Full pipeline tested (parse → quantize → build → run)

## Common Issues

1. **Shape inference errors**: Ensure output shape computation handles all edge cases
2. **Quantization accuracy**: Test with representative data, compare float vs quantized
3. **Memory allocation**: Ensure SRAM allocation is efficient
4. **Performance**: Profile on simulator, optimize hotspots

## Related Operations

Look at similar operations for reference:
- Compute ops: `src/ops/` (Add, Mul, etc.)
- Activation ops: (ReLU, Sigmoid, etc.)
- Reduction ops: (ReduceSum, ReduceMean, etc.)
- Transform ops: (Transpose, Reshape, etc.)
