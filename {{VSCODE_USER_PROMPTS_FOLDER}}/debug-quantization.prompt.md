---
name: debug-quantization
description: Debug quantization issues and accuracy problems in AIPU models
parameters:
  - name: issue_type
    description: "Type of issue: accuracy_drop, overflow, nan_values, or other"
    required: true
  - name: model_name
    description: "Name of the model being debugged"
    required: true
  - name: problem_layer
    description: "Specific layer with issues (if known)"
    required: false
---

# Debug Quantization: {{model_name}}

**Issue**: {{issue_type}}
**Problematic layer**: {{problem_layer | default('Unknown - will investigate')}}

## Step 1: Enable Verbose Logging

```bash
# Run quantization with logging enabled
aipuaqt --cfg config.cfg \
  --graph {{model_name}}.def \
  --bin {{model_name}}.bin2 \
  --log \
  --dump \
  --dump_dir debug_output/
```

This generates:
- Per-layer quantization statistics
- Scale and zero-point values
- Intermediate tensor dumps

## Step 2: Analyze Quantization Statistics

Check `debug_output/` for quantization parameters:

```python
import numpy as np

# Load statistics
def analyze_quant_params(layer_name):
    # Check if scale is reasonable
    scale = load_scale(layer_name)
    zp = load_zero_point(layer_name)
    
    print(f"Layer: {layer_name}")
    print(f"  Scale: {scale}")
    print(f"  Zero point: {zp}")
    print(f"  Effective range: [{-128*scale + zp}, {127*scale + zp}]")
    
    # Warning signs:
    if scale > 1.0:
        print("  ⚠️  Large scale - may lose precision")
    if scale < 1e-6:
        print("  ⚠️  Tiny scale - may overflow during requantization")
    if abs(zp) > 128:
        print("  ⚠️  Invalid zero point")
```

## Step 3: Compare Float vs Quantized Outputs

{% if issue_type == "accuracy_drop" %}
### Accuracy Drop Analysis

```python
import numpy as np

def compare_outputs(float_output, quant_output, scale, zp):
    # Dequantize
    dequant = (quant_output.astype(np.float32) - zp) * scale
    
    # Compute metrics
    mse = np.mean((float_output - dequant) ** 2)
    mae = np.mean(np.abs(float_output - dequant))
    max_error = np.max(np.abs(float_output - dequant))
    
    # Relative error
    rel_error = np.mean(np.abs(float_output - dequant) / (np.abs(float_output) + 1e-8))
    
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Max Error: {max_error:.6f}")
    print(f"Relative Error: {rel_error:.4%}")
    
    # Layer-by-layer comparison
    # Identify which layer causes accuracy drop
    if rel_error > 0.05:  # >5% error
        print("⚠️  Significant accuracy loss - investigate this layer")
        
    return mse, mae, max_error

# Run comparison
for layer_name in model.layers:
    float_out = run_float(layer_name)
    quant_out = run_quantized(layer_name)
    scale = get_scale(layer_name)
    zp = get_zero_point(layer_name)
    
    print(f"\n{layer_name}:")
    compare_outputs(float_out, quant_out, scale, zp)
```

### Common Accuracy Issues

1. **Insufficient calibration data**
   - Use more calibration samples (50-1000 images)
   - Ensure calibration data is representative

2. **Aggressive quantization**
   - Try per-channel quantization for Conv/FC layers
   - Use ASYMMETRIC instead of SYMMETRIC

3. **Sensitive layers**
   - Keep some layers in higher precision (int16/fp16)
   - Use mixed precision quantization

{% elif issue_type == "overflow" %}
### Overflow Analysis

```python
def check_overflow(tensor, dtype='int8'):
    """Check if tensor values overflow int8 range."""
    if dtype == 'int8':
        overflow_max = np.sum(tensor > 127)
        overflow_min = np.sum(tensor < -128)
    elif dtype == 'int32':
        overflow_max = np.sum(tensor > 2147483647)
        overflow_min = np.sum(tensor < -2147483648)
    
    total = tensor.size
    
    if overflow_max > 0 or overflow_min > 0:
        print(f"⚠️  Overflow detected!")
        print(f"  Max overflows: {overflow_max} / {total} ({100*overflow_max/total:.2f}%)")
        print(f"  Min overflows: {overflow_min} / {total} ({100*overflow_min/total:.2f}%)")
        print(f"  Actual range: [{tensor.min()}, {tensor.max()}]")
        return True
    return False

# Check each layer
for layer in problematic_layers:
    output = layer.compute_quantized()
    if check_overflow(output):
        # Need to adjust scale or use requantization
        print(f"Fix: Adjust scale for {layer.name}")
```

### Overflow Fixes

1. **Adjust scales**
   ```python
   # Use statistics with clipping (e.g., 99.9 percentile)
   f_min = np.percentile(activations, 0.1)
   f_max = np.percentile(activations, 99.9)
   ```

2. **Requantization**
   ```cpp
   // For Conv: output in int32, requantize to int8
   int32_t acc = conv_output;
   int8_t final = requantize(acc, scale_out / (scale_w * scale_in));
   ```

3. **Intermediate precision**
   - Use int16 for accumulation
   - Requantize back to int8 after operation

{% elif issue_type == "nan_values" %}
### NaN Values Analysis

```python
def find_nan_source(model):
    """Find which layer produces NaN."""
    for i, layer in enumerate(model.layers):
        output = layer.compute()
        
        if np.isnan(output).any():
            print(f"⚠️  NaN detected at layer {i}: {layer.name}")
            print(f"  Input stats: min={layer.input.min()}, max={layer.input.max()}")
            print(f"  Scale: {layer.scale}")
            print(f"  Zero point: {layer.zero_point}")
            
            # Check for common causes
            if layer.scale == 0:
                print("  Issue: Scale is zero!")
            if np.isinf(layer.scale):
                print("  Issue: Scale is infinite!")
            
            return layer
    
    return None

# Debug NaN
nan_layer = find_nan_source(model)
if nan_layer:
    # Inspect the problematic layer
    print(f"\nInvestigating {nan_layer.name}...")
```

### NaN Fixes

1. **Zero scale**
   - Caused by constant activations during calibration
   - Fix: Use minimum scale threshold (e.g., 1e-6)

2. **Division by zero**
   - Check for zero denominators in quantization
   - Add epsilon: `scale = (f_max - f_min + 1e-8) / 255`

3. **Invalid statistics**
   - Calibration data has NaN/Inf
   - Filter invalid samples before calibration

{% endif %}

## Step 4: Visualize Quantization Effects

```python
import matplotlib.pyplot as plt

def visualize_quantization(float_tensor, quant_tensor, scale, zp):
    """Visualize quantization effects."""
    dequant_tensor = (quant_tensor - zp) * scale
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Float distribution
    axes[0].hist(float_tensor.flatten(), bins=100)
    axes[0].set_title('Float32 Distribution')
    
    # Quantized distribution
    axes[1].hist(quant_tensor.flatten(), bins=256, range=(-128, 127))
    axes[1].set_title('Int8 Distribution')
    
    # Error distribution
    error = float_tensor - dequant_tensor
    axes[2].hist(error.flatten(), bins=100)
    axes[2].set_title('Quantization Error')
    
    plt.tight_layout()
    plt.savefig('quant_analysis.png')
    print("Visualization saved to quant_analysis.png")
```

## Step 5: Adjust Quantization Configuration

Based on findings, update `config.cfg`:

```ini
[AutoQuantizationTool]
# Try different quantization method
quantize_method = ASYMMETRIC  # or SYMMETRIC

# Enable per-channel for Conv/FC layers
ops_per_channel = Conv2d,DepthwiseConv,Linear

# Adjust calibration batch size
quantize_batch_size = 100

# Adjust input range (if needed)
input_max = 255
input_min = 0

# Use custom statistics (if computed separately)
ts_max_file = custom_max_stats.txt
ts_min_file = custom_min_stats.txt
```

## Step 6: Layer-Specific Debugging

```python
def debug_layer(layer_name):
    """Deep dive into specific layer."""
    print(f"\n=== Debugging {layer_name} ===")
    
    # 1. Check input distribution
    input_tensor = get_layer_input(layer_name)
    print(f"Input: min={input_tensor.min():.4f}, max={input_tensor.max():.4f}, mean={input_tensor.mean():.4f}")
    
    # 2. Check weight distribution (if applicable)
    if has_weights(layer_name):
        weights = get_layer_weights(layer_name)
        print(f"Weights: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")
    
    # 3. Check output distribution
    output_float = run_layer_float(layer_name)
    output_quant = run_layer_quantized(layer_name)
    
    print(f"Output (float): min={output_float.min():.4f}, max={output_float.max():.4f}")
    print(f"Output (quant): min={output_quant.min()}, max={output_quant.max()}")
    
    # 4. Check quantization params
    scale = get_scale(layer_name)
    zp = get_zero_point(layer_name)
    print(f"Scale: {scale:.6f}, Zero point: {zp}")
    
    # 5. Compute error
    dequant = (output_quant - zp) * scale
    error = np.abs(output_float - dequant)
    print(f"Quantization error: mean={error.mean():.6f}, max={error.max():.6f}")
```

## Common Fixes

### Fix 1: Better Calibration
```bash
# Use more calibration data
# Ensure diverse samples in calibration set
# Use percentile-based clipping (not min/max)
```

### Fix 2: Per-Channel Quantization
```python
# For Conv layers, use per-output-channel quantization
# More accurate but slightly slower
ops_per_channel = Conv2d,DepthwiseConv
```

### Fix 3: Mixed Precision
```python
# Keep sensitive layers in higher precision
# E.g., first/last layer, attention layers
# Requires builder support for mixed precision
```

### Fix 4: Calibration Strategy
```python
# Instead of min/max, use percentile
def compute_scale_percentile(activations):
    f_min = np.percentile(activations, 0.1)   # 0.1th percentile
    f_max = np.percentile(activations, 99.9)  # 99.9th percentile
    scale = max(abs(f_min), abs(f_max)) / 127
    return scale
```

## Validation

After fixes, validate:

```bash
# 1. Re-run quantization
aipuaqt --cfg updated_config.cfg --graph {{model_name}}.def --bin {{model_name}}.bin2

# 2. Build and run
aipugb {{model_name}}.def -w {{model_name}}.bin2 -o output.bin
aipurun {{model_name}}.def -w {{model_name}}.bin2 -i input.bin -o output.bin

# 3. Measure accuracy
python validate_accuracy.py --quant-output output.bin --float-model model.pb
```

## Checklist

- [ ] Enabled logging and dumping
- [ ] Analyzed quantization statistics
- [ ] Compared float vs quantized outputs layer-by-layer
- [ ] Identified problematic layer(s)
- [ ] Root cause identified
- [ ] Applied fix (calibration/per-channel/mixed-precision)
- [ ] Validated accuracy improvement
- [ ] Performance impact acceptable

## References

- `python/src/AIPUBuilder/Optimizer/qat/readme.md`
- `doc_internal/quantization_white_paper/`
- `tools/NodeAttributesTuningTools/README.md`
