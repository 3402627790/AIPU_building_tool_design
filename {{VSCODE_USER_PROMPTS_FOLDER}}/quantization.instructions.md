---
name: quantization
description: "Use when: working on quantization, calibration, int8 conversion, or files in python/src/AIPUBuilder/Optimizer/"
applyTo:
  - "python/src/AIPUBuilder/Optimizer/**/*.py"
  - "**/*quant*.py"
  - "**/*quantiz*.py"
  - "**/*calibrat*.py"
---

# Quantization Development Guidelines

## Quantization Overview

AIPUBuilder quantizes float32 models to int8 for AIPU execution. The quantization tool (AQT) is in `python/src/AIPUBuilder/Optimizer/`.

**Key concepts:**
- **Calibration**: Collect activation statistics from representative dataset
- **Quantization methods**: SYMMETRIC or ASYMMETRIC
- **Per-channel**: Some ops support per-channel quantization (e.g., DepthwiseConv)

## Quantization Formula

**Symmetric:**
```
q = round(f / scale)
scale = max(abs(f_max), abs(f_min)) / 127
```

**Asymmetric:**
```
q = round(f / scale) + zero_point
scale = (f_max - f_min) / 255
zero_point = round(-f_min / scale)
```

## Adding Quantization for New Ops

1. **Define op quantization in `ops/` directory**:
```python
from AIPUBuilder.Optimizer.framework import OpQuantizer

@register_quantizer("YourOp")
class YourOpQuantizer(OpQuantizer):
    def quantize(self, node, context):
        # Get input quantization params
        inp_scale = context.get_scale(node.inputs[0])
        inp_zp = context.get_zero_point(node.inputs[0])
        
        # Compute output quantization params
        out_scale, out_zp = self.compute_output_params(node)
        
        # Set quantization info
        context.set_scale(node.outputs[0], out_scale)
        context.set_zero_point(node.outputs[0], out_zp)
```

2. **Handle calibration statistics**:
```python
def calibrate(self, node, statistics):
    # Get min/max from statistics
    f_min = statistics.get_min(node.outputs[0])
    f_max = statistics.get_max(node.outputs[0])
    
    # Compute scale and zero point
    return self.compute_quant_params(f_min, f_max)
```

## Op-Specific Patterns

### Convolution/Linear Layers
- Weights: Per-channel quantization supported
- Bias: Quantized to int32 with `scale_w * scale_in`
- Requantize output back to int8

### Element-wise Ops
- Align input scales when needed
- Preserve accuracy for ops like Add/Mul

### Activation Functions
- ReLU: Simple clipping, no rescale needed
- Sigmoid/Tanh: Use lookup tables
- Clip: Quantize threshold values

### Pooling
- No quantization params needed
- Just apply pooling in quantized domain

## Configuration

Quantization config in INI format:

```ini
[AutoQuantizationTool]
quantize_method = SYMMETRIC
ops_per_channel = DepthwiseConv,Conv2d
data_for_test_or_quantize = /path/to/dataset
quantize_batch_size = 20
input_scale = 1.0
input_mean = [123.68, 116.78, 103.94]
input_max = 151
input_min = -128
preprocess_mode = normalize
```

## Statistics Collection

```python
# Collect activation statistics
def collect_statistics(model, dataset):
    stats = {}
    for batch in dataset:
        outputs = model.forward(batch)
        for tensor_name, value in outputs.items():
            if tensor_name not in stats:
                stats[tensor_name] = {'min': [], 'max': []}
            stats[tensor_name]['min'].append(value.min())
            stats[tensor_name]['max'].append(value.max())
    
    # Aggregate statistics (e.g., percentile-based)
    return aggregate_stats(stats)
```

## Debugging Quantization

### Enable Logging
```bash
aipuaqt --cfg config.cfg --log
```

### Dump Layer Outputs
```bash
aipuaqt --cfg config.cfg --dump --dump_dir output/
```

### Use Pre-computed Statistics
```bash
aipuaqt --ts_max_file max_values.txt --ts_min_file min_values.txt
```

### Visualize Quantization
```python
# Compare float vs quantized
import numpy as np

def compare_accuracy(float_out, quant_out, scale, zp):
    dequant_out = (quant_out - zp) * scale
    mse = np.mean((float_out - dequant_out) ** 2)
    print(f"MSE: {mse}")
```

## Common Issues

1. **Accuracy drop**: 
   - Check calibration dataset is representative
   - Verify statistics are not clipped too aggressively
   - Consider per-channel quantization for sensitive layers

2. **Overflow/underflow**:
   - Ensure intermediate results fit in int32
   - Check requantization shift amounts

3. **Zero-point issues**:
   - SYMMETRIC mode sets zero_point=0
   - ASYMMETRIC mode computes zero_point

4. **Scale mismatch**:
   - Element-wise ops need aligned scales
   - Use requantization when needed

## Testing Quantized Models

```bash
# Run quantized model on simulator
aipurun quantized.def -w quantized.bin2 -i input.bin -o output.bin

# Compare with float model output
python compare_outputs.py float_output.npy quant_output.bin
```

## Per-Channel Quantization

For Conv2D/DepthwiseConv:
```python
# Weights shape: [out_channels, in_channels, h, w]
for c in range(out_channels):
    w_min = weights[c].min()
    w_max = weights[c].max()
    scales[c] = max(abs(w_min), abs(w_max)) / 127
    
    # Quantize per channel
    quant_weights[c] = np.round(weights[c] / scales[c])
```

## References

- `python/src/AIPUBuilder/Optimizer/doc_internal/`
- `python/src/AIPUBuilder/Optimizer/qat/readme.md`
- Quantization white paper in `doc_internal/quantization_white_paper/`
