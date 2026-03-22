# AIPUBuilder User Guide

* [Introduction](#introduction)
* [Installation](#installation)
* [Usage](#usage)
* [Example](#example)

## Introduction

AIPUBuilder is a serial build tool for compiling a NN model to a AIPU executable file.
The AIPUBuilder includes mainly 3 parts:
#### Parser

The model parse module (also called the Parser) will parse a third-party model and convert to an NPU internal IR. The supported third-party models include TensorFlow, PyTorch, ONNX and etc.

#### Optimization

The quantization module (also called the OPT, short for Optimizer) will quantize the model into the quantized model if needed, reorganize the quantized data, and then transfer the quantized model to the generation module. 

#### GBuilder
The generation module (also called the GBuilder) will optimize the execute path and lower the model to fit NPU instructions, and then build the model to an executable file.  


## Installation

The AIPUBuilder  is packed as a python wheel package, i.e. WHL file. So you can just use pip to install it.
```shell
pip3 install AIPUBuilder-xxx-xxx.whl
```

Please note that:

* use a compatible python version to install whl.
* use `sudo` or `--user` option when you don't have root permission.
* the package will depend Tensorflow, numpy, networks and other 3rd packages, please make sure the dependencies are meet.

## Usage

After installation, the command line `aipubuild` should in your system path, and you can run it in terminal.

### `aipubuild`
The *aipubuild* is the entry point of the package. You can use *-h* or
*--help* to get the help information. The tool uses a configuration file
for its parameters. You need to specify all the related parameters to
run it.

The following is a simple example:

$aipubuild test.cfg

The configuration file is a standard .*ini* configuration file. It
consists of four sections:

-   Common
-   Parser
-   Optimizer
-   GBuilder

#### Common section

The Common section only contains one key mode that will tell *aipubuild*
which mode to run in. This key is in run mode by default, and there are
only two modes available:

-   run: Run the third-party model on the NPU simulator.
-   build: Build the third-party model into an NPU executable file.

#### Parser section

This section is for the model parse module. The following table lists
the configuration fields for the Parser section.

Configuration fields for the Parser section
| Configuration field   | Attribute | Description |
|-----------------------|-----------|-------------|
| model_type            | Optional  | The model format of the input model. By default, it is TensorFlow. The supported types are: TensorFlow TFLite ONNX Caffe Torch |
| model_name            | Required  | The name of the input model, must be a string. It is used to identify a model. |
| model_domain          | Required  | The domain of the model. There are only 5 available options: image_classification , object_detection , keyword_spotting , speech_recognition , image_segmentation. detection_postprocess is required when model_domain is object_detection. |
| detection_postprocess | Optional  | If your model_domain is object_detection and you are using the official detection model, you need to specify your detection post process. Currently, only the following types of post process are supported: SSD , SSD_RESNET , YOLO2 , YOLO3_TINY , YOLO3_FULL , CAFFE_FASTERRCNN |
| input_model           | Required  | File path of the input third-party model. Currently, it supports TensorFlow frozen pb, h5, SaveModel (the input will be a folder), tflite, onnx, and caffemodel. |
| caffe_prototxt        | Optional  | Prototxt file path of the Caffe model. |
| input                 | Optional  | The input node name of the model. If you have several inputs, use commas to separate them. The input tensor name is experimentally supported for this field. If you do not know the node name or the node name is empty, you can try to use the tensor name here. It should be required if the input node information is not included in the model. |
| output                | Optional  | The output node name of the model. If you have several outputs, use commas to separate them. The output tensor name is supported for this field. If you do not know the node name or the node name is empty, you can try to use the tensor name here. It should be required if the output node information is not included in the model. |
| input_shape           | Optional  | The input shape of the model. Usually it is a single tensor shape, for example, input_shape =[1,224,224,3]. For models with unknown input shape, this field must be set. If you have several inputs, use commas to separate them, for example, input_shape=[1,224,224,3],[1,112,112,3]. If this field is present, the input field must also be present, and the number of input shapes must be equal to the number of inputs. |
| similarity_input_npy  | Optional  | The path of a binary file in NumPy .npy format. A dictionary in format {input tensor name 1: input tensor 1, …, input tensor name N: input tensor N}, in which input tensor names are string and input tensors are NumPy array, should be saved in the file. The cosine similarity and mean errors between outputs from the Optimizer forwarding with float IR or quantized IR and outputs from runtime of the model’s framework will be shown if the file path is provided. |
| force_float_ir        | Optional  | If this field is set to true in configuration, the Parser will convert the quantized op in the QAT model into float op and the output IR will be a float IR. For example, force_float_ir=true. If this field is set to false in configuration or is not present, the Parser will keep the quantized op in the model if it exists and the output IR will be a quantized IR or a float IR according to different models. For example, force_float_ir=false. |
| force_cpu             | Optional  | This field is used only for parsing a CPU TorchScript (the tensor saved as a CPU device) file to ensure that it is processed by the CPU in a CUDA available device. |
| iou_threshold         | Optional  | Overlap threshold value between two boxes. |
| obj_threshold         | Optional  | Confidence threshold value of the current box. |
| max_box_num           | Optional  | Maximum box number of some detection layers (such as Region/DecodeBox). |
| image_width           | Optional  | Input image width. |
| image_height          | Optional  | Input image height. |
| preprocess_type       | Optional  | The pre-processing (running on the NPU) to be applied on the model inputs. Currently it supports RESIZE, RGB2BGR, BGR2RGB, RGB2GRAY, BGR2GRAY, and STANDARDIZATION. When several pre-processing types are set, they will run on the NPU sequentially. |
| preprocess_params     | Optional  | The parameters needed in pre-processing. The parameters of each pre-processing should be set with a brace. When the pre-processing does not need parameters, an empty brace {} should be used. 1. For RGB2BGR, BGR2RGB, RGB2GRAY, and BGR2GRAY, no parameters are needed.  2. For RESIZE, it needs two parameters. One is the resize method, which can be bilinear or nearest. The other is the input image shape. The destination shape is the original model shape that you do not need to set. For example, the setting of a bilinear RESIZE, with an input image shape [1,448,448,3], should be {bilinear,[1,448,448,3]}. 3. For STANDARDIZATION, it needs one parameter—axis. For the axis setting, it can be any combination of different dimensions. For example, for a 3-dimensional input, {[0]} means that only the dimension-0 axis is processed. {[1,2]} means that both the dimension-1 axis and dimension-2 axis are processed. {[0,1,2]} means that all the three dimension axis are processed. When the setting is empty, such as {} or {[]}, it also means that all the axis are processed. Therefore, for an input that applies RESIZE, RGB2BGR, and then STANDARDIZATION pre-processing, the preprocess_type should be RESIZE, RGB2BGR, STANDARDIZATION. The preprocess_params should be similar to ‘{bilinear,[1,448,448,3]}, {}, {[1,2]}’. |


#### Optimizer section

This section is for the quantization module. The following table lists
the configuration fields for the Optimizer section.

Configuration fields for the Optimizer section

| Configuration field | Attribute | Description |
|----------------------|-----------|-------------|
| dataset | Optional | A dataset plugin name used to create a dataset to handle the data and label. It is recommended that you create a custom dataset plugin for your own particular model. |
| calibration_data | Optional | A dataset path for calibrating a model. |
| calibration_batch_size | Optional | The batch size of inferencing the graph in quantization calibration. It may affect the quantization accuracy if the selected calibration strategy uses statistic values that depend on the batch_size. |
| calibration_shuffe | Optional | Indicates whether to apply shuffle over the calibration dataset. It may affect the quantization accuracy. |
| calibration_strategy_for_activation | Optional | The activation calibration strategy. Currently, it supports ‘mean’, ‘extrema’, ‘Nstd’, ‘KLD’, and other strategies. It defaults to ‘mean’. |
| calibration_strategy_for_weigth | Optional | The weight calibration strategy. Currently, it supports ‘extrema’, ‘Nstd’, ‘KLD’, and other strategies. It defaults to ‘extrema’. |
| weight_bits | Optional | Weight bits for quantizing weight data. It supports 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 and 16. It defaults to 8. Currently only 8 and 16 are supported in Zhouyi Compass operators. See the *Arm China Zhouyi Compass Operators Specification Application Note* for more information. |
| bias_bits | Optional | Bias bits for quantizing bias data. It supports 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47 and 48. It defaults to 32. |
| activation_bits | Optional | Activation bits for quantizing activation data. It supports 8, 9, 10, 11, 12, 13, 14, 15 and 16. It defaults to 8. Currently only 8 and 16 are supported in Zhouyi Compass operators. See the *Arm China Zhouyi Compass Operators Specification Application Note* for more information. |
| lut_items_in_bits | Optional | The maximum number of LUT items (in bits, because it only supports LUT with 2**N items) when representing nonlinear functions in quantization. It defaults to 8. It is recommended to set it to a value of 10 or greater when quantizing activations to 16-bit. |
| running_statistic_momentum | Optional | Momentum (range[0.0, 1.0]) used for calculating weighted average of some statistics when the calibration dataset has multiple batches, such as min value = momentum * pre min value + (1-momentum) * cur_min_value. It defaults to 0.9. |
| histc_bins | Optional | Bins for statistic histograms of each tensor. |
| trim_infinity_before_statistic | Optional | Exclude the infinite or equivalent very large/small values from statistic. |
| save_statistic_info | Optional | Indicates whether to save and dump the statistic information file. |
| quantize_method_for_weight | Optional | Quantization method for weight and bias. |
| quantize_method_for_activation | Optional | Quantization method for activation. |
| global_calibration | Optional | Global calibration (scale/zero_point/rounding optimization) method. |
| statistic_file | Optional | A statistic file path. |
| out_ir_name | Optional | The output IR file name. |
| output_dir | Optional | The output directory for quantized IR, calibration statistic file, and quantization configuration JSON file. |
| metric | Optional | The metric plugin name for metricizing a model. |
| data | Optional | A dataset path for metricizing a model. |
| label | Optional | A label path for metricizing a model. |
| metric_batch_size | Optional | Batch size when metricizing a model. |
| unify_scales_for_concat | Optional | Indicates whether to unify scales of each concat's branch when possible. |
| unify_scales_for_concat_max_depth | Optional | When enabling unify_scales_for_concat (=True), the maximum search depth can be settled for speeding up. |
| unify_scales_for_multi_inputs_operators | Optional | Indicates whether to unify scales of input branches for assigned operators when possible. |
| with_winograd | Optional | Indicates whether to enable the Winograd algorithm when possible. |
| opt_config | Optional | File path. This JSON file stores configurations of each node. |
| dump | Optional | Indicates whether to enable the feature of dumping all tensors and other data. |
| dump_dir | Optional | Path. All dump data files will save to this path when 'dump=true’. |
| quantize_method_for_weight | Optional | Quantization method for weight and bias. |
| quantize_method_for_activation | Optional | Quantization method for activation. |
| global_calibration | Optional | Global calibration (scale/zero_point/rounding optimization) method. |
| statistic_file | Optional | A statistic file path. |
| out_ir_name | Optional | The output IR file name. |
| output_dir | Optional | The output directory for quantized IR, calibration statistic file, and quantization configuration JSON file. |
| metric | Optional | The metric plugin name for metricizing a model. |
| data | Optional | A dataset path for metricizing a model. |
| label | Optional | A label path for metricizing a model. |
| metric_batch_size | Optional | Batch size when metricizing a model. |
| unify_scales_for_concat | Optional | Indicates whether to unify scales of each concat's branch when possible. |
| unify_scales_for_concat_max_depth | Optional | When enabling unify_scales_for_concat (=True), the maximum search depth can be settled for speeding up. |
| unify_scales_for_multi_inputs_operators | Optional | Indicates whether to unify scales of input branches for assigned operators when possible. |
| with_winograd | Optional | Indicates whether to enable the Winograd algorithm when possible. |
| opt_config | Optional | File path. This JSON file stores configurations of each node. |
| dump | Optional | Indicates whether to enable the feature of dumping all tensors and other data. |
| dump_dir | Optional | Path. All dump data files will save to this path when 'dump=true’. |

You can also use the 'global\_value &
&lt;\[(layer\_id1,layer\_id2),(layer\_id3,layer\_id4),
...\]:local\_value1&gt; &lt; \[operator\_type1, operator\_type2,
...\]:local\_value2&gt; ...' format for per-layer configuration directly
in the configuration file. Where:

-   'global\_value' is the default value to configure each layer.

-   'local\_value' is for overwriting the default value on specific
    layers you assigned.

-   'layer\_id' stands for layer\_id in the input IR and '(layer\_id1,
    layer\_id2)' specify the layers which will be applied (layer\_id1
    &lt;= layer\_id &lt;= layer\_id2).

-   'operator\_type' stands for valid operator type names that specify
    the operators which will be applied.

Currently, the supported fields include:

-   calibration\_strategy\_for\_activation

-   calibration\_strategy\_for\_weight

-   quantize\_method\_for\_weight

-   quantize\_method\_for\_activation

-   weight\_bits

-   activation\_bits

-   bias\_bits

-   lut\_items\_in\_bits

-   force\_dtype\_int

-   force\_shift\_positive

-   running\_statistic\_momentum

-   histc\_bins

-   resize\_degrade\_to\_nearest

-   with\_winograd

-   trigger\_float\_op

-   trim\_infinity\_before\_statistic

For example, for a model with int8 quantized weights, to set int16
quantized weights for layer\_id 5-6, layer\_id 10-12 and the Abs
operator, you can set
‘weight\_bits=8&&lt;\[(5,6),(10,12)\]:16&gt;&lt;\[Abs\]:16&gt;’ (without
blank).

#### GBuilder section

This section describes the options for controlling the GBuilder module
behavior. The following table lists the configuration options for the
GBuilder section.

Configuration options for the GBuilder section
| Configuration option | Attribute       | Description                                                                                                                                                                                                                   |
|----------------------|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| simulator            | Optional        | The path of the simulator when the mode is run. If you leave it empty, it will use the simulator path of the last run.                                                                                                       |
| inputs               | Required in run mode | The input file. It is only used when the mode is run. The input file is for running on the simulator. Input files are usually some binary files, for example, a raw image, or a raw tensor. If you have several inputs, use commas to separate them. |
| local_lib            | Optional        | The path of the local library if you want to use a custom library. Usually, you can leave it blank.                                                                                                                          |
| target               | Optional        | The target hardware that you want to build/run. By default, it is X2_1204. It supports all the Zhouyi series platforms. Currently it supports Z2_0901, Z2_1002, Z2_1104, Z3_0901, Z3_1104, X1_1204, X2_1204, X2_1204MP3, X3_1304, and X3_1304MP2. |
| profile              | Optional        | Define whether to enable the profiler. For the usage of the profiler, see 7 Profiling a network. When set to True, some profiling data will be created in the running path if the related simulator supports the profiling feature. See the technical overview for more information. |
| prof_unit            | Optional        | Define which hardware unit the profiler will profile. Currently, it supports TPC, DMA and AIFF. For the detailed setting rules, see 7.1 Overview of the profiler tool.                                                         |
| fast_perf            |                 | Disable the DMA/AIFF profile, which may cost much extra time/cycle. Only for Zhouyi X2 and above.                                                                                                                             |
| max_sample_point_num |                 | Maximum sample point number of the profiler, only for Zhouyi X2 and above.                                                                                                                                                    |
| max_aiff_sample_point_num |             | Maximum AIFF sample point number of the profiler, only for Zhouyi X2 and above.                                                                                                                                                |
| max_dma_sample_point_num |           | Maximum DMA sample point number of the profiler, only for Zhouyi X2 and above.                                                                                                                                                 |
| max_custom_sample_point_num |     | Maximum custom sample point number of the profiler, only for Zhouyi X2 and above.                                                                                                                                               |
| stack_size           |                 | The stack size (KB) for the NPU. The default value is 257 (KB).                                                                                                                                                               |
| png                  |                 | A file name with suffix .png for visualization of the graph in PNG format. To generate the PNG image, you must install the graphviz package. It is recommended to use SVG which is faster and smaller.                         |
| svg                  |                 | A file name with suffix .svg for visualization of the graph in SVG format. To generate the SVG image, you must install the graphviz package.                                                                                   |
| tcm_size             | Optional        | The DTCM size of the X1_1204 SoC platform. By default, it is 4MB. It also can be a value in kilobytes.                                                                                                                       |
| gm_size              | Optional        | The Global Memory size of Zhouyi X2 and above SoC platforms. It accepts a number in Kbytes. The default value will be the same as the hardware configuration.                                                                 |
| tiling               | Optional        | Currently, there are 2 modes available: fps mode and fps_dev mode. fps mode is the standard performance mode, and fps_dev mode includes some more aggressive strategies based on the fps mode. For different scenarios, these strategies sometimes have additional benefits. |
| ocm_size             | Optional        | Set the SoC SRAM size or On-Chip Memory (OCM) size of the Zhouyi X2 and above SoC platform. The default value is 0. The value can be a number in Kbytes.                                                                       |
| ocm_addr             | Optional        | Set the OCM address. Only for run mode.                                                                                                                                                                                       |
| gt                   | Optional        | Set to True to run aipuexe to get the ground truth of the model. The ground truth output filename will be the same as the output filename, but with an additional suffix .gt.                                                |
| disable_input_buffer_reuse | Optional | Set to True to disable the reuse of input buffers, so that all input buffers will not be reused by other feature maps.                                                                                                       |
| disable_output_buffer_reuse | Optional | Set to True to disable the reuse of output buffers, so that all output buffers will not be reused by other feature maps.                                                                                                     |
| max_wt_size          | Optional        | For versions X2 and above, due to the limitation of the elf32 format which only supports files smaller than 4GB, any weight file exceeding this size must be divided into several segments. Expected a int number in MBytes.    |


##### Example configuration file

The following is an example file.

```ini
[Common]
mode=run

[Parser]
model_name=resnet_50
model_type=tensorflow
output=resnet_v1_50/predictions/Reshape
input_model=./resnet_50/frozen.pb
input=Placeholder
input_shape=[1,224,224,3]
preprocess_type=RESIZE, RGB2BGR
preprocess_params={bilinear,[1,448,448,3]},{}

[Optimizer]
model_name=resnet_50
calibration_data=./Dataset/ImageNet/preprocess/resnet/calibration_100.npy
data=./Dataset/ImageNet/preprocess/resnet/dataset_1000.npy
label=./Dataset/ImageNet/preprocess/resnet/label_1000.npy
calibration_batch_size=20
metric_batch_size=50
dataset=NumpyDataset
metric=TopKMetric

[GBuilder]
simulator=./aipu_simulator
target=X2_1204
inputs=./resnet_50/input.bin
outputs=./resnet_50/output_resnet_50.bin
profile=True
prof_unit=DMA
tiling=fps
```