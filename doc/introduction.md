# AIPUBuilder User Guide

AIPUBuilder is a serial build tool for compiling a Tensorflow frozen model to a AIPU executable file, it will support other framework latter.
The AIPUBuilder includes mainly 3 parts:
* Parser
* Quantization tool
* Builder

The parser will parse a 3rd party model and  convert to AIPU internal intermediate representation(IR), then the quantization tool will try to quantize the model into int8 model, then transfer the int8 model to the builder, the builder will optimize the execute path and lower the model to fit AIPU instructions, then will build the model to an executable file.

## Installation

The AIPUBuilder  is packed as a python wheel package, i.e. WHL file. So you can just use pip to install it.
```shell
pip install AIPUBuilder-xxx-xxx.whl
```

The CPP Builder don't need installation, we distribute it in binary form for dev.


Please note that:

* use a compatible python version to install whl.
* use `sudo` or `--user` option when you don't have root permission.
* the package will depend Tensorflow, numpy, networks and other 3rd packages, please make sure the dependencies are meet.

## Usage

After installation, AIPUBuilder will provide 5 commands to use it:
* aipuparser
* aipuaqt
* aipugb
* aipugb(cpp)
* aipurun
* aipurun(cpp)
* aipubuild
* aipu_profiler(cpp)


### `aipuparser`
`aipuparser`  uses config file for parameters transport. The usage of command is simple:` aipuparser -c cfg`

Bellow is the `aipuparser`'s config file description.

* `model_name` [__required__ ]

    the name for the input model, must be a string, it use to identify the model.

* `model_domain`  [__required__ ]

    the domain of the model, there are only 4 available options:
    * `image_classification`
    * `object_detection`
    * `keyword_spotting`
    * `speech_recognition`
    * ~~`image_segmentation`~~

* `detection_postprocess`  [__required__ when `model_domain` is  `object_detection`]

    if your model_domain is `object_detection`, and if you a using the official detection model, please specify your detection post process. Now it only support tow types of post process:

    * `yolo`
    * `ssd`

* `input_model`  [__required__ ]

    file path of the input 3rd party model, current only support tensorflow frozen pb.

* `input`  [__required__ ]

    the input(s) node(s)' name of the model. if you have several inputs, use `,` to separate each one.

* `input_shape` [__required__ ]

    the input shape(s) of model, usually it is a single tensor shape, for example: `input_shape=[1,224,224,3]`

    if you have several inputs, please use comma the separate them, for example: `input_shape=[1,224,224,3],[1,112,112,3]`

* `output`  [__required__ ]

    the output(s) node(s)' name of the model. if you have several outputs, use `,` to separate each one.

* `output_dir` [__required__ ]

    the output directory for store the graph ir. the output would be two files, one is the graph definition file with extension `.def`, the name is `%model_name%.def` and another is the weight file with extension `.bin2`, the name is `%model_name%.bin2`.

 Here is an example for the `aipuparser`. Please note that all options are in the **[Common]** section.
```ini
[Common]
model_name = resnet50
model_domain = image_classification
detection_postprocess =
input_model = /project/ai/scratch01/guaxu01/eac_models/IR/code/Parser_2019_july/src/tmp/resnet50_v1.pb
input = input
output = resnet_v1_50/predictions/Reshape
input_shape = [1,224,224,3]
output_dir = /project/ai/scratch01/guaxu01/eac_models/IR/code/Parser_2019_july/src/
```

### `aipuaqt`
the usage of `aipuaqt` you can just type `aipuaqt -h` in shell to get the help info:
```shell
usage: aipuaqt [-h] [--cfg CFG] [--graph GRAPH] [--bin BIN]
               [--model_name MODEL_NAME] [--output_nodes OUTPUT_NODES]
               [--model_domain {IMAGE_CLASSIFICATION,OBJECT_DETECTION,SPEECH_RECOGNITION,IMAGE_SEGMENTATION,KEYWORD_SPOTTING}]
               [--quantize_method {SYMMETRIC,ASYMMETRIC}]
               [--ops_per_channel OPS_PER_CHANNEL] [--reverse_rgb] [--input_data_format]
               [--data_for_test_or_quantize DATA_FOR_TEST_OR_QUANTIZE]
               [--quantize_batch_size QUANTIZE_BATCH_SIZE]
               [--output_dir OUTPUT_DIR] [--input_scale INPUT_SCALE]
               [--input_mean INPUT_MEAN [INPUT_MEAN ...]]
               [--label_id_offset LABEL_ID_OFFSET] [--ts_max_file TS_MAX_FILE]
               [--ts_min_file TS_MIN_FILE] [--input_max INPUT_MAX]
               [--input_min INPUT_MIN] [--preprocess_mode PREPROCESS_MODE]
               [--norm_flag NORM_FLAG] [--dataset_name DATASET_NAME] [--log]
               [--dump] [--dump_dir DUMP_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --cfg CFG             The config file.
  --graph GRAPH         The float32 ir graph definition file.
  --bin BIN             The float32 ir binary.
  --model_name MODEL_NAME
                        Model name.
  --output_nodes OUTPUT_NODES
  --model_domain {IMAGE_CLASSIFICATION,OBJECT_DETECTION,SPEECH_RECOGNITION,IMAGE_SEGMENTATION,KEYWORD_SPOTTING}
  --quantize_method {SYMMETRIC,ASYMMETRIC}
  --ops_per_channel OPS_PER_CHANNEL
  --reverse_rgb
  --input_data_format
  --data_for_test_or_quantize DATA_FOR_TEST_OR_QUANTIZE
                        The dataset path for test the quantization method.
  --quantize_batch_size QUANTIZE_BATCH_SIZE
  --output_dir OUTPUT_DIR
                        Path to save the int8 ir after quantization.
  --input_scale INPUT_SCALE
  --input_mean INPUT_MEAN [INPUT_MEAN ...]
  --label_id_offset LABEL_ID_OFFSET
  --ts_max_file TS_MAX_FILE
                        The file path of the statistics max value of each
                        layer.the statistics max value of the post-process
                        layers for the object detection.
  --ts_min_file TS_MIN_FILE
                        The file path of the statistics min value of each
                        layer. the statistics min value of the post-process
                        layers for the object detection.
  --input_max INPUT_MAX
                        The max value of the input.
  --input_min INPUT_MIN
                        The min value of the input.
  --preprocess_mode PREPROCESS_MODE
  --norm_flag NORM_FLAG
  --dataset_name DATASET_NAME
  --log                 Enable the log.
  --dump                Enable dump the resut for each layer.
  --dump_dir DUMP_DIR   the dump dir.
```


### `aipugb`
the usage of `aipugb` you can just type `aipugb -h` in shell to get the help info:
```bash
usage: aipugb [-h] [-v] [-o <The output graph bin file>]
              [-w <The weight file of the graph>]
              [--dump_nodes <The dump nodes of the graph>]
              [--out_nodes <The output nodes of the graph, default is the last layer>]
              [--dump] [--target <target name>] [--png <graph.png>]
              [-L <Custom Assembly lib dir>] [--disable-qformat]
              [--new-interface] [--hwa-config <hwa.config>]
              <The graph definition file>

positional arguments:
  <The graph definition file>
                        graph definition file.

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -o <The output graph bin file>, --output <The output graph bin file>
                        generate the graph binary file for runtime.
  -w <The weight file of the graph>, --weight <The weight file of the graph>
                        graph weight file, if no, please leave empty.
  --dump_nodes <The dump nodes of the graph>
                        the dump nodes of the graph, default is the last
                        layer. please enter the dump nodes names, split by
                        ",". For example <--dump_nodes "probe1,fc2">
  --out_nodes <The output nodes of the graph, default is the last layer>
                        the output nodes of the graph, default is the last
                        layer. please enter the dump nodes names, split by
                        ",". For example <--out_nodes "probe1,fc2">
  --dump                dump the ro text data file in current directory
  --target <target name>
                        Setting a specific target, default is ZY_SC_000.
                        Supported targets
                        :ZY_SC_000,ZY_SC_001,ZY_SC_002,ZY_SC_003
  --png <graph.png>     dump the graph to image with png format, need
                        dependency pydot.
  -L <Custom Assembly lib dir>, --Lib <Custom Assembly lib dir>
                        Custom Assembly lib dir, building tool will find lib
                        in custom dir first.
  --disable-qformat     disbale use qformat for scale
  --new-interface       use New HWA Interface, the old interface will delete
                        soon,.
  --hwa-config <hwa.config>
                        write the hwa register configuration to txt file
```

### `aipugb(cpp)`
the usage of `aipugb` you can just type `aipugb  -h` in shell to get the complete help info:
```bash
Usage:
positional arguments:
  graphdef
        The input graph
optional arguments:
  -v,--version
        print the version.
  -w <WEIGHT>,--weight <WEIGHT>
        The weight file for the graph
  -h,--help
        print the help message.
  --dump_tensors <DUMP_TENSORS>
        The tensors name which need to dump. Default is empty.
  --out_tensors <OUT_TENSORS>
        The tensors need to outputs. Default is the graph output tensors.
  --dump
        dump the ro/text/data file for emulator
  --target <TARGET>
        The target to be built.
  --png <PNG>
        The graph in png format for vis.
  -L <LIB>,--Lib <LIB>
        The library search path for customize lib, use ":" as separator if have more than two lib pathes.
  -p,--profile
        enable profile counter.
  --show_profile_events
        print supported profile event id and relative description
  --profile_event <PROFILE_EVENT>
        select profile events
  --frequency <FREQUENCY>
        The runtime frequency(MHz), used for profiler, default is 800(MHz).
  -o <OUTPUT>,--output <OUTPUT>
        The output binary file.
```

### `aipurun`
the usage of `aipurun` you can just type `aipurun -h` in shell to get the help info:
```bash
usage: aipurun [-h] [-v] [-o <output bin file>] [-i <"input1,input2,input3">]
               [-w <The weight file of the graph>]
               [--dump_nodes <The dump nodes of the graph, default is the last layer>]
               [--target <target name>] [--terminator <nodename>]
               [--simulator <path to simulator>]
               [-L <Custom Assembly lib dir>] [--hwa-config <hwa.config>]
               [--png <graph.png>] [--enable-trace] [--enable-perf]
               [--diagnostics <level>] [--fast_fwd FAST_FWD] [--dump]
               [--new-interface] [--log <log for simulator>]
               [--disable-qformat]
               <The graph definition file>

positional arguments:
  <The graph definition file>
                        graph definition file.

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -o <output bin file>, --output <output bin file>
                        graph output
  -i <"input1,input2,input3">, --input <"input1,input2,input3">
                        graph input(s) split with ","
  -w <The weight file of the graph>, --weight <The weight file of the graph>
                        graph weight file, if no, please leave empty.
  --dump_nodes <The dump nodes of the graph, default is the last layer>
                        the dump nodes of the graph, default is the last
                        layer. please enter the dump nodes names, split by
                        ",". For example <--dump_nodes "probe1,fc2">
  --target <target name>
                        Setting a specific target, default is ZY_SC_000.
                        Supported targets
                        :ZY_SC_000,ZY_SC_001,ZY_SC_002,ZY_SC_003
  --terminator <nodename>
                        the last executed node name, exit the program after
                        this node>
  --simulator <path to simulator>
                        set simulator path
  -L <Custom Assembly lib dir>, --Lib <Custom Assembly lib dir>
                        Custom Assembly lib dir, building tool will find lib
                        in custom dir first.
  --hwa-config <hwa.config>
                        write the hwa register configuration to txt file
  --png <graph.png>     dump the graph to image with png format, need
                        dependency pydot.
  --enable-trace        enable-trace for simulator
  --enable-perf         enable-perf for simulator
  --diagnostics <level>
                        diagnostics for simulator
  --fast_fwd FAST_FWD   fast_fwd for simulator
  --dump                dump the ro text data file in current directory
  --new-interface       use New HWA Interface, the old interface will delete
                        soon,.
  --log <log for simulator>
                        Set the trace output file. if not set, it will be in
                        ./default.log.
  --disable-qformat     disable use qformat for scale
```

### `aipurun(cpp)`
the usage of `aipurun` you can just type `aipurun --help-hidden` in shell to get the help info:
```bash
Usage:
positional arguments:
  graphdef
        The input graph
optional arguments:
  -v,--version
        print the version.
  -w <WEIGHT>,--weight <WEIGHT>
        The weight file for the graph
  -h,--help
        print the help message.
  --dump_tensors <DUMP_TENSORS>
        The tensors name which need to dump. Default is empty.
  --out_tensors <OUT_TENSORS>
        The tensors need to outputs. Default is the graph output tensors.
  --dump
        dump the ro/text/data file for emulator
  --target <TARGET>
        The target to be built.
  --png <PNG>
        The graph in png format for vis.
  -L <LIB>,--Lib <LIB>
        The library search path for customize lib, use ":" as separator if have more than two lib pathes.
  -p,--profile
        enable profile counter.
  --show_profile_events
        print supported profile event id and relative description
  --profile_event <PROFILE_EVENT>
        select profile events
  --frequency <FREQUENCY>
        The runtime frequency(MHz), used for profiler, default is 800(MHz).
  --dot <DOT>
        The graph in dot format for vis.
  -D,--DEBUG
        Building graph in debug mode.
  --show-all-passes
        Show all passes for optimize graph.
  --dumpIR <DUMPIR>
        Dump the optimized IR. a <DUMPIR>.def and a<DUMPIR>.bin file would be generated.
  --serialize-before-pass <SERIALIZE-BEFORE-PASS>
        Dump the optimized IR before a pass name. If you have several passes, use ',' to separate them
  --serialize-after-pass <SERIALIZE-AFTER-PASS>
        Dump the optimized IR after a pass name. If you have several passes, use ',' to separate them
  --svg-before-pass <SVG-BEFORE-PASS>
        save graph to svg before a pass name. If you have several passes, use ',' to separate them
  --svg-after-pass <SVG-AFTER-PASS>
        save graph to svg before a pass name. If you have several passes, use ',' to separate them
  --disable-pass <DISABLE-PASS>
        Disable pass(es) by name. If you have several passes, use ',' to separate them
  --enable-pass <ENABLE-PASS>
        Enable pass(es) by name. If you have several passes, use ',' to separate them
  -i <INPUT>,--input <INPUT>
        The inputs file for the graph
  -o <OUTPUT>,--output <OUTPUT>
        The output binary file.
  --sim-args <SIM-ARGS>
        set other simulator supported arguments
  --simulator <SIMULATOR>
        The simulator executable path.
  --help-hidden
        Show this message
```

### `aipu_profiler(cpp)`
the `aipu_profiler` is used to generate perf html report, the report is layer-level granularity.
usage of `aipu_profiler` you can just type `aipu_profiler --help` in shell to get the help info:
```bash
Usage:
positional arguments:
  graphjson
  	The input graph json file
  result
  	The output result from runtime
optional arguments:
  -h,--help
  	print the help message.
  -o <OUTPUT>,--output <OUTPUT>
  	The output htmp report
  --mode <MODE>
  	generate report based on optimized normal nodes or original nodes
```

### `aipubuild`

The `aipubuild` is the entry point of the package, you can use `-h` or `--help` to get its help information.  It uses a configuration file for its parameters, in the config file, you need to specify all related parameters to run it.

Here is a simple example:
```shell
aipubuild test.cfg
```

#### `aipubuild` config file format

The config file is a standard `ini` configuration file. It consists four sections:
* Common
* Parser
* AutoQuantizationTool
* GBuilder

##### Common section
The common section just contains one key `mode`,  it will tell `aipubuild` run on which mode. This key by default is on `run` mode, and there are just two modes available:

* `run`: will run the 3rd party model on AIPU simulator.
* `build`: will build the 3rd party model to AIPU executable file.

##### Parser section
The section is for the parser part:

* `input_shape` [__required__ ]

    the input shape(s) of model, usually it is a single tensor shape, for example: `input_shape=[1,224,224,3]`

    if you have several inputs, please use comma the separate them, for example: `input_shape=[1,224,224,3],[1,112,112,3]`
* `model_name` [__required__ ]

    the name for the input model, must be a string, it use to identify the model.

* `model_domain`  [__required__ ]

    the domain of the model, there are only 4 available options:
    * `image_classification`
    * `object_detection`
    * `keyword_spotting`
    * `speech_recognition`
    * ~~`image_segmentation`~~

* `detection_postprocess`  [__required__ when `model_domain` is  `object_detection`]

    if your model_domain is `object_detection`, and if you a using the official detection model, please specify your detection post process. Now it only support tow types of post process:

    * `yolo`
    * `ssd`

* `input_model`  [__required__ ]

    file path of the input 3rd party model, current only support tensorflow frozen pb.

* `input`  [__required__ ]

    the input(s) node(s)' name of the model. if you have several inputs, use `,` to separate each one.

* `output`  [__required__ ]
    the output(s) node(s)' name of the model. if you have several outputs, use `,` to separate each one.

##### AutoQuantizationTool section

* `quantize_method` [__optional__ ]

    The quantization method, just support 'SYMMETRIC' mode yet.

* `ops_per_channel` [__optional__ ]

    The ops support per channel quantization, just support 'DepthwiseConv' yet.

* `reverse_rgb` [__optional__ ]

    In image domain, if need to reverse the rgb channels, default 'False'.

* `input_data_format` [__optional__ ]

    In image domain, if need to transpose input data format, like 'NCHW' to 'NHWC', this param should set 'NCHW', default 'NHWC'.

* `data_for_test_or_quantize` [__required__ ]

    The data-set path for test the quantization method. In image domain, only support `tfrecord` format yet. In Speech, support the __raw__ data.

* `quantize_batch_size` [__optional__ ]

    The batch size of the data. In object detection and speech, only support 1 yet.

* `input_scale` [__required__ when in image domain]

    The preprocess scale for the input.

* `input_mean` [__required__ when in image domain]

    The preprocess mean for the input.

* `label_id_offset` [__optional__ ]

    the label id offset for the model and the dataset, 0 by default

* `ts_max_file` [__optional__ ]


    The file path of the statistics max value of each layer, which including the statistics max value of the post-process layers for the object detection.
    if not set the file path, AQT will calculate the statistics.

* `ts_min_file` [__optional__ ]


    The file path of the statistics min value of each layer, which including the statistics min value of the post-process layers for the object detection.
    if not set the file path, AQT will calculate the statistics.


* `input_max` [__required__ when in image domain]

    The max value of the input.

* `input_min` [__required__ when in image domain]

    The min value of the input.

* `preprocess_mode` [__required__ ]

    The preprocess mode for the input. In the image domain, use `normalize`. In speech, support `mfcc` and `spectrogram`

* `norm_flag` [__optional__ ]

    Set True if need normalize in speech, else False, False by default.

* `dataset_name` [__optional__ ]

    The standard dataset name. In image, [VOC, COCO], in speech, [aishell]

* `log` [__optional__ ]

    Enable the log to print the quantization information. Default set False.


##### GBuilder section
This section is some options for control the GBuilder behavior.

* `simulator` [__optional__]

    the path of simulator when mode is `run`, if you left empty, it will use last run's simulator path.

* `inputs` [__required__  in `run` mode]

    the input file(s), it's  only used when mode is `run`. the input file(s) is for running on simulator, it's usually some binary files, for example a raw images, or a raw tensor.

    if you have several inputs, use `,` to separate each one.

* `png` [__optional__]

    a path for saving the model's graph to png files, for example `graph.png`

* `local_lib` [__optional__]

    the path of local lib if you want to you local lib or you want to use customize lib.
    Usually left it blank.

* `target` [__optional__]

    the target Hardware you want build/run, support:

    * ZY_SC_000
    * ZY_SC_001
    * ZY_SC_002
    * ZY_SC_003

    default value is `ZY_SC_000`

#### Example file
```ini
[Common]
mode=run
#[build, run], build will build a runnable bin for runtime, run will directly run on simulator

[Parser]
input_shape = [1,224,224,3]
model_name = resnet50
model_domain = image_classification
detection_postprocess =
input_model = /project/ai/scratch01/neowan01/git/buildtool/Parser/models/resnet50/frozen.pb
input = Placeholder
output = resnet_v1_50/predictions/Reshape

[AutoQuantizationTool]
quantize_method = SYMMETRIC
ops_per_channel = DepthwiseConv
reverse_rgb = False
data_for_test_or_quantize = /project/ai/scratch01/Dataset/ImageNet/tfrecord/imagenet_data_1000/
quantize_batch_size = 20
input_scale = 1.0
input_mean = [123.68, 116.78, 103.94]
label_id_offset = 0
ts_max_file =
ts_min_file =
input_max = 151
input_min = -128
preprocess_mode = normalize
norm_flag = False
dataset_name =
log = False

[GBuilder]
simulator=
inputs= intput.bin
outputs=
png=
hwa_config=
local_lib=
target=
```
