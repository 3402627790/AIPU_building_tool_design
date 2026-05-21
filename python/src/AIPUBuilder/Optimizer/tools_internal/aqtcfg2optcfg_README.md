<font size=10>aqtcfg2optcfg</font>

The aqtcfg2optcfg python script is used to transfer the old aipubuild cfg file, which has the '[AutoQuantizationTool]' section, to the new aipubuild cfg file.

# introduction

when tranfering the fields in '[AutoQuantizationTool]' section, there are three rules:

1. the field which still uses in the new cfg file will directly save to the new cfg file:
    *calibration_data, data, label ...*
2. the field should be transfered to new names, for example:
    *calibration_method -> calibration_strategy_for_activation*,
    *quantize_batch_size -> calibration_batch_size and metric_batch_size*
3. the field should be obsoleted, and will comment in the new cfg file: for example:
    *calibration_label, label_offset_id, reverse_rgb, ops_per_channel ...*

The 3th rule will output log messages to suggest users how to deal with this obsoleted fields.

This script will save the new filename: *oldname_opt.cfg* (if the old cfg filename is oldname.cfg)

# usage

how to run this python script?

1. the old cfg file input to this script.

    ```python
        # transfer one cfg file
        python3 aqtcfg2optcfg.py aipu_config.cfg
    ```
2. a directory path also can input to this script and then transfer all the cfg files'.

    ```python
        # transfer the cfg file in the path
        python3 aqtcfg2optcfg.py /path/
    ```

# example

the old aipubuild cfg file(only '[AutoQuantizationTool]' section part):
```
...

[AutoQuantizationTool]
output_nodes =
quantize_method = SYMMETRIC
ops_per_channel = DepthwiseConv
reverse_rgb = True
data = ./dataset.npy
label = ./label.npy
data_for_test_or_quantize = ./segmentation/
quantize_batch_size = 1
output_dir = ./IR_int
input_scale = 1.0
input_mean = [103.94, 123.68, 116.78]
label_id_offset = 0
ts_max_file = ./fcn_max.npy
ts_min_file = ./fcn_min.npy
input_max = 151
input_min = -122
quant_precision=int8
preprocess_mode = normalize
norm_flag = False
dataset_name =
log = False
dump = False

...
```

after transfering,

```
[Optimizer]
data=./dataset.npy
label=./label.npy
calibration_batch_size=1
metric_batch_size=1
output_dir=./IR_int
activation_bits=8
weight_bits=8
bias_bits=32
dump=False

# Optimizer does not need these fields:
#output_nodes=
#quantize_method=SYMMETRIC
#ops_per_channel=DepthwiseConv
#reverse_rgb=True
#data_for_test_or_quantize=./segmentation/
#input_scale=1.0
#input_mean=[103.94,123.68,116.78]
#label_id_offset=0
#ts_max_file=./fcn_max.npy
#ts_min_file=./fcn_min.npy
#input_max=151
#input_min=-122
#preprocess_mode=normalize
#norm_flag=False
#dataset_name=
#log=False
```
