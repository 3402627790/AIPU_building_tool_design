- [AIPULLM 使用说明](#aipullm-使用说明)
  - [1.简介](#1简介)
  - [2.工作流程和内容介绍](#2工作流程和内容介绍)
    - [2.1 build IR](#21-build-ir)
    - [2.2 quantize IR](#22-quantize-ir)
    - [2.3 simplify IR](#23-simplify-ir)
    - [2.4 compile IR](#24-compile-ir)
    - [2.5 app](#25-app)
      - [2.5.1 pyllmchat](#251-pyllmchat)
      - [2.5.2 pdmodellmchat](#252-pdmodellmchat)
      - [2.5.3 sharewtllmchat](#253-sharewtllmchat)
      - [2.5.4 vlchat](#254-vlchat)
      - [2.5.5 metric app](#255-metric-app)
  - [3.如何使用?](#3如何使用)
    - [3.1 命令行参数和cfg配置参数介绍](#31-命令行参数和cfg配置参数介绍)
    - [3.2 配置文件cfg的介绍](#32-配置文件cfg的介绍)
      - [3.2.1 llm跑出词的cfg](#321-llm跑出词的cfg)
      - [3.2.2 多模态跑出词的cfg](#322-多模态跑出词的cfg)
      - [3.2.3 其他应用](#323-其他应用)
  - [4. 添加一个模型](#4-添加一个模型)
    - [4.1 构图](#41-构图)
      - [4.1.1 添加一个模型构图有两种方式](#411-添加一个模型构图有两种方式)
      - [4.1.2 构图OpsAPI](#412-构图opsapi)
      - [4.1.3 QAT量化信息更新](#413-qat量化信息更新)
    - [4.2 量化](#42-量化)
    - [4.3 测试](#43-测试)
  - [5. 一些使用小技巧](#5-一些使用小技巧)

# <center>AIPULLM 使用说明

## 1.简介

aipullm是从Huggingface模型到编译生成AIPU binary的工具，该工具可以帮助用户快速验证，部署相关大模型到AIPU simulator或者硬件上。

## 2.工作流程和内容介绍

aipullm的工作流程图如下：
<div align="center">
  <img src="./images/aipullm_workflow.png" alt="aipullm workflow", width=200>
</div>

### 2.1 build IR

在aipullm中使用NN-Compiler的OpsAPI构造与transformers库图结构等价的Compass IR。

```
# 导入OpsAPI module
from AIPUBuilder import ops
from AIPUBuilder.core import *
import numpy as np
with Graph("test") as g:
    t0 = Tensor(np.random.rand([1, 128]).astype(np.float32))
    t1 = Tensor(np.random.rand([1, 128]).astype(np.float32))
    t2 = t0 + t1
```

针对某个模型的构图，需要遵从两点：

  1. 该构图需要继承class BaseModel的基类，
  2. 需要通过 **register_causallm** 完成注册。

以Qwen2为例（具体的源码可见AIPUBuilder/model/llm/qwen2/qwen2.py），class Qwen2CausalLm继承BaseModel，完成qwen2的构图后通过register_causallm注册。

```

@register_causallm(['qwen2', 'qwen1.5'])
class Qwen2CausalLM(BaseModel):
    def __init__(self, ...):
        ...

    def forward(self, ...):
        with Graph('qwen') as g:
            ....
        return g
```

> 如果多个模型的构图过程一样，或者在构图过程可以区分不同，那么可以借鉴同一个构图代码，只需在register_causallm增加model_name即可。例如上述的qwen2, qwen1.5.

对于经典的LLM的模型，我们通常构图做法是将其拆分成embedding IR, (attention * blk + lmhead) IR 两部分IR。当然，也可以在构图的时候拆分成多个IR，如果想拆成多个IR，那么在构图的时候只需将拆出来的部分遵从上述2个规则进行处理即可。
例如qwen2.py的存在将NormLmhead拆成单独IR的操作。

```
@register_causallm(['qwen2_lmhead'])
class NormLmHead(BaseModel):
    def __init__(self, ...):
        ...

    def forward(self, ...):
        with Graph('qwen_lmhead') as g:
            ....
        return g
```

此外，对于embedding部分，如果想产生单独IR (见qwen2.py)，因为embedding的功能比较单一，从embedding data中gather出token idx对应的data， 因此只需embedding data的数据即可。构图在class BaseEmbedding完成。

```

@register_causallm(['qwen2_embedding', 'qwen1.5_embedding'])
class Qwen2EmbCausalLM(BaseEmbedding):
    def __init__(self, config, prefix=''):
        super().__init__(config, prefix)
        self.embed_weights = self.config.weights.get('model.embed_tokens.weight')
```

> 在aipullm使用pyllmchat等app时候，也可以不提供embedding的IR，aipullm会根据model_name自己从safetensors中获得embedding data，完成从token idx到embedding data的选择。

该阶段生成的IR会存在output_dir目录下， 并且IR文件以配置的model_name为文件名 例如model_name=qwen2, 那么量化后的IR文件名： qwen2.txt, qwen2.bin。

### 2.2 quantize IR

<a id="quantization"></a>

在完成上述构图后，可以选择调用量化策略对float IR进行量化。目前aipullm中支持量化方式包括：
1. 原始模型为QAT的模型，例如常见的AWQ，GPTQ量化完成的模型，如：[Qwen1.5-1.8B-Chat-AWQ](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat-AWQ)。aipullm会保持量化weight和scale/zp不变。
2. 原始模型为float32/bfloat16/float16未量化模型，进行weight-only-quantization, 例如wint4 *afp16, wint4* abf16, wfp4 *afp16, wfp4* abf16等。wint4 支持per-group 和per-channel-asymmetric量化， wfp4 支持per-group量化（wfp4目前只支持symmtric量化）。在per-group时，需要在cfg中设置 weight_block_size的字段，并且 block_size 只支持[32, 64, 128, 256]。
3. 原始模型进行带校准的量化，例如wfp8 *afp8，目前该量化方式需要配置更多的optimizer的配置，支持模型有限，建议采用Optimizer进行wfp8* afp8的量化调优
4. 支持编写定制化量化配置，此时遵从以下规则：
    1. 需要继承class ModelQuantize
    2. 需要实现update_qinfo函数
    3. 通过register_model_quantize注册该模型的量化方式。
    示例如下：假设qwen2的quant_method=w4，有特殊的处理，可以通过编写下面的类来实现。源码参见：AIPUBuilder/model/llm/qwen2/update_quantization.py

```
    @register_model_quantize(['qwen2_w4'])
    class Qwen2W4ModelQuantize(ModelQuantize):
        def __init__(self, ...):
            pass

        def update_qinfo(self, ...):
            pass
```

>note: 对于lm_head中的fullyconnected在QAT模型中往往采用fp16处理，如果此时想用更激进的量化，可以通过quantization_config=lmhead_fc_weight_type=w4"，具体见 quantization_config配置参数。

该阶段生成的IR会存在output_dir目录下， 并且IR文件以配置的model_name和量化方式为后缀结尾， 例如model_name=qwen2, quant_method=w4, 那么量化后的IR文件名： qwen2_w4.txt, qwen2_w4.bin。

### 2.3 simplify IR

aipullm会对上述量化图进行graph级别的图优化，该功能等同于aipugsim。该功能默认开启，如果想关闭，在cfg中配置：enable_gsim=false 即可。

该阶段生成的IR会存在output_dir目录下， 并且IR文件以 **_s.txt/bin** 结尾。

### 2.4 compile IR

上述产生IR，在cfg中设置target后，会进入编译IR的阶段。在该阶段，

  1. 会调用aipugb对Compass IR进行编译，生成AIPU binary。
  2. 编译阶段aipugb 默认参数为: --target X3S_1304 --enable_ds。
  3. 如果想配置更多aipugb的参数，可以通过cfg参数builder_config设置，配置参数名和设置完全遵循aipugb命令参数。举例： builder_config="--enable_ds=false"。
编译完成的binary会存在output_dir目录下。

### 2.5 app

在aipullm中，在上述流程产生的Compass IR 或者AIPU binary，可以进行对话出词，metric测试等应用，不仅如此，例如Compass IR或者 AIPU binary还可以进行其他应用，例如模型Cosine验证。

#### 2.5.1 pyllmchat

pyllmchat app 主要实现大模型的出词，pyllmchat使用prefill和decoder共用一套动态IR。该app的参数配置为

**app = pyllmchat(prompt="Where is the capital of China?")**。更多的输入设置，可以参见如下注释。

```
@register_app()
class PyLLMChat(LLMChat):
    """PYLLMChat is used for LLM model chat and chat scripts uses python."""

    def __init__(self, prompt: str = '', sys_prompt: str = '', sampling: str = "NONE", sampling_params: str = "", max_new_tokens: str = "", dump: bool = False, with_chat_template: str = "True"):
        """PYLLMChat constructor.

        Parameters
        ---------
        prompt: str
            the question discription for the LLM model.

        sys_prompt: str
            the prompt for getting better quality chat.

        sampling: str
            sampling method, now only support: NONE, RANDOM.

        sampling_params: str
            when sampling=RANDOM, can configure the sampling_params, the sampling_params
            can set 3 values, like "1.1,0.95,0"

        max_new_tokens: str
            if set, app will terminate at number of tokens, regardless of context status

        dump: Union[str, bool]
            whether dump the input/output tensors of graph when running.

        """
```

在使用pyllmchat时，需要配置engine字段:

  1. engine=opt，那么会将对应的IR运行在OptForward()上。
  2. engine=exe，那么会将其对应的aipugb编译后的图运行在Executor()上
  3. engine=sim, 会将编译后的AIPU binary，通过aipudumper在simulator上运行，此时**必须配置 simulator的路径**.
  4. engine=npu, 那么会将编译后的AIPU binary 通过Compass Runtime运行在硬件上或者FPGA或者Simulator上。

#### 2.5.2 pdmodellmchat

pdmodellmchat 也是实现大模型出词，主要适配PD分离的IR，即产生了prefill和decoder的IR或者AIPU binary的场景。
同样在使用pdmodellmchat时需要设置engine（同pyllmchat）。

该pdmodellmchat app 输入参数设置同pyllmchat。

#### 2.5.3 sharewtllmchat

sharewtllmchat 也是实现大模型出词，主要适配prefill和decoder编译产生binary，压缩后share weight的方式。
在使用sharewtllmchat时,只能设置为engine=npu。

该sharewtllmchat app 输入参数设置同pyllmchat。

#### 2.5.4 vlchat

vlchat app为支持多模态对话的应用程序，可以适配Qwen2.5-vl的图片和prompt的出词。
**app = vlchat(prompt="Describe this image.", image="10149.png")**

```
@register_app()
class VLChat(LLMChat):
    """VLChat is used for multi-modal chat and chat scripts uses python."""

    def __init__(self,
                 prompt: str = '',
                 sys_prompt: str = '',
                 sampling: str = "NONE",
                 sampling_params: str = "",
                 image: str = '',
                 video: str = '',
                 post_process: str = 'false',
                 **kwargs):
        """VLChat constructor.

        Parameters
        ---------
        prompt: str
            the question discription for the LLM model.

        sys_prompt: str
            the prompt for getting better quality chat.

        sampling: str
            sampling method, now only support: NONE, RANDOM.

        sampling_params: str
            when sampling=RANDOM, can configure the sampling_params, the sampling_params
            can set 3 values, like "1.1,0.95,0"

        image: str
            the image path

        video: str
            the video path
        """
```

#### 2.5.5 metric app

aipullm 还支持metric app，例如pplmetric，MMStarmetric等用于大模型指标复现的app。

1. pplmetric 输入参数设置如下： **app = pplmetric(wiki_text="raw.txt")**

```
@register_app()
class PPLMetric(LLMChat):
    def __init__(self, wiki_text='', max_seq_len='2048', model_name=''):
        """PPLMetric constructor.

        Parameters
        ---------
        wiki_text: str
            the wiki dataset path.

        max_seq_len: str
            the max sequence length of prompt for metric
        """
```

2. MMStarmetric 输入参数设置如下: **app = mmstarmetric(prompt="", post_process="true",dataset_path="./Dataset/MMStar", images_num="200")**

```
@register_app()
class MMStarMetric(LLMChat):
    """MMStarMetric is used for multi-modal chat and chat scripts uses python."""

    def __init__(self,
                 prompt: str = '',
                 sys_prompt: str = '',
                 sampling: str = "NONE",
                 sampling_params: str = "",
                 image: str = '',
                 video: str = '',
                 post_process: str = 'false',
                 dataset_path: str ='',
                 images_num: str = "200",
                 **kwargs):
        """MMStarMetric constructor.

        Parameters
        ---------
        prompt: str
            the question discription for the LLM model.

        sys_prompt: str
            the prompt for getting better quality chat.

        sampling: str
            sampling method, now only support: NONE, RANDOM.

        sampling_params: str
            when sampling=RANDOM, can configure the sampling_params, the sampling_params
            can set 3 values, like "1.1,0.95,0"

        image: str
            the image path

        video: str
            the video path

        post_process: bool
            whether enable post_process

        image_num: int
            the image number for this metric
        """
```

## 3.如何使用?

### 3.1 命令行参数和cfg配置参数介绍

使用aipullm方式为 **aipullm -c aipullm.cfg**， 具体的cfg配置见后续章节。
aipullm支持的cfg中配置参数如下，可以通过 **aipullm -h** 查看。

>使用aipullm命令工具需要先完成AIPUBuilder wheel包安装，设置生效PATH和PYTHONPATH环境变量。

```
usage: AIPULLM [-h] [-v] [-c CFG] [--model_path MODEL_PATH] [--config_name CONFIG_NAME] [--model_name MODEL_NAME] [--seq_len SEQ_LEN] [--past_len PAST_LEN]
               [--quant_type QUANT_TYPE] [--target TARGET] [--file_type FILE_TYPE] [--use_cache USE_CACHE] [--fill_llminfo FILL_LLMINFO]
               [--kvcache_optimization KVCACHE_OPTIMIZATION] [--vcache_axis VCACHE_AXIS] [--enable_subgraph ENABLE_SUBGRAPH] [--output_dir OUTPUT_DIR]
               [--max_wt_size MAX_WT_SIZE] [--vocab_path VOCAB_PATH] [--tokenizer_path TOKENIZER_PATH] [--app APP] [-i INPUT] [--engine ENGINE]
               [--option_config OPTION_CONFIG] [--quantization_config QUANTIZATION_CONFIG] [--builder_config BUILDER_CONFIG] [--aipu_binary_path AIPU_BINARY_PATH]
               [--simulator SIMULATOR] [--enable_gsim ENABLE_GSIM] [--only_fp_ir] [--only_qt_ir] [--serialize_fp_ir SERIALIZE_FP_IR]

options:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -c CFG, --cfg CFG     AIPULLM config file
  --model_path MODEL_PATH
                        model path , like meta-llama/llama2_chn
  --config_name CONFIG_NAME
                        json file name
  --model_name MODEL_NAME
                        if model_name is "", the model_type in config.json will set to LlmInfo. The model_type is relative with building the compass IR, tokenizer in
                        c-runtime etc.
  --seq_len SEQ_LEN     the current seq_len in Compass IR and the max sequence length in chat.
  --past_len PAST_LEN   the past_len in Compass IR of cache.
  --quant_type QUANT_TYPE
                        the quantization type for float IR, like w4 is meaning: activation fp16, weight int4.
  --target TARGET       zhouyi target
  --file_type FILE_TYPE
                        specify the file type for weight, like safetensors, torch.bin or GGUF file. when has multi-gguf files with different quantization type, this field
                        should be with quantization type.
  --use_cache USE_CACHE
                        whether enable kv cache
  --fill_llminfo FILL_LLMINFO
                        whether fill llm information, like embedding data to aipubinary
  --kvcache_optimization KVCACHE_OPTIMIZATION
                        whether enable kv cache optimization, like cacheupdate op
  --vcache_axis VCACHE_AXIS
                        set v cacheupdate concat axis, default value is 2.
  --enable_subgraph ENABLE_SUBGRAPH
                        split the subgraph of forward function which decorated by @gen_subgraph to subgraph
  --output_dir OUTPUT_DIR
                        the output file's directory
  --max_wt_size MAX_WT_SIZE
                        the aipubinary size
  --vocab_path VOCAB_PATH
                        set the vocab file path
  --tokenizer_path TOKENIZER_PATH
                        set the tokenizer path for gguf.
  --app APP             model app plugin, its configuration format like: app=appname(arg_name0="arg_value0", arg_name1="arg_value1", ...), which is like function
                        definition in the right side of the equals sign. The charater of appname should be in [a-zA-Z_0-9]. The contents within the parentheses are the
                        passed-in parameter variables. The variables should be with arg_key=arg_value format, and the arg_value should be enclosed with double quotes. The
                        variables are splitted by comma. if arg_value has double quote, please escape with a backslash. After parsing, all arg_keys and arg_values will be
                        as the initial parameter of appname. You can reference the qwen2_aipullm.cfg example.
  -i INPUT, --input INPUT
                        The inputs data file for the graph, if you have multiple inputs, use comma "," to split them. and now only support npy file format.
  --engine ENGINE       set the forward engine, like opt, exe, fpga.
  --option_config OPTION_CONFIG
                        configuration field in config.json wants to set, and use ',' to separate. In addition to the fields defined in cofig.json(like
                        num_hidden_layers=1), aipullm can also configure: num_hidden_layers(indicates block numbers, which should lookup the config.json to decide this
                        field name, like num_layers in chatglm), can be setted integer, list, or nest-list, when integer, will generate range(0, num_hidden_layers) block
                        index, when list, like [2,5], will generate [2, 3, 4, 5] block index, when nest-list, like [[2,2],[5,6]], will generate [2, 5, 6] block index.
                        use_cache=false, will generate un-kvcache IR; with_lm=false, will exclude the lm_head part in IR; gptq_gidx_to_fc=true, will change the gidx of
                        gptq to fc when desc_act=true in gptq, otherwise, to gather; force_float_ir=true, will dequantize the weight if its is quantized and serialize the
                        float IR; broadcast_matmul=false, will use matmul's broadcast feature to eliminate tile op for GQA; swap_matmul_input=false, will try to swap
                        certain input dims of matmul to improve the performance;
  --quantization_config QUANTIZATION_CONFIG
                        set quantization feilds, now supported: lmhead_fc_weight_type=w4 or fp16, will set fc weight in lmhead to int4 or fp16; is_perf_mode=true, will
                        enable non-linear activation with aiff implementation for higher performance; cast_dtypes_for_lib=false, if true, will insert cast op depending on
                        the data type of inputs, which is not implemented by lib; quantize_method_for_weight='', which can set: per_tensor_symmtric_restricted_range,
                        per_channel_symmetric_restricted_range, per_channel_asymmetric, per_vector_symmetric_restricted_range, per_vector_asymmetric; weight_block_size=-1,
                        if in [32, 64, 128, 256], will enable aipu blockwise quantization. weight_only_quantization_op=[fullyconnected], if use quant_type=w4, only fc
                        layer is configurated to weight only quantization. if want to add other layer, please set like weight_only_quantization=[fullyconnected,
                        convolution]
  --builder_config BUILDER_CONFIG
                        set aipugb compile feilds, when want to set more aipugb commands, using builder_config=key1=value1,key2=value2,key3=value3 to set.
  --aipu_binary_path AIPU_BINARY_PATH
                        if setted aipu_binary_path, the process of building fpIR/QIR/aipubin all skips, directly run this aipubin
  --simulator SIMULATOR
                        the simulator executable path. when engine=sim, should set the simulator field.
  --enable_gsim ENABLE_GSIM
                        whether skip gsim function.
  --only_fp_ir          generate float IR only
  --only_qt_ir          generate float IR and quantized IR, not build and run
  --serialize_fp_ir SERIALIZE_FP_IR
                        whether serialize float IR

```

### 3.2 配置文件cfg的介绍

#### 3.2.1 llm跑出词的cfg

以qwen2-awq为例，完成出词流程，配置如下: 必须配置model_path, model_name。
    - model_path用于找到跟模型相关配置的config.json以及safetensors
    - model_name用于识别OpsAPI的构图，生成Compass Float IR.

```
[Common]
model_path = Qwen/Qwen2.5-1.5B-Instruct-AWQ
model_name = qwen2
output_dir = ./
quant_type=w4
target = X3S_1304
app = pyllmchat(prompt="Where is the capital of China?")
engine=exe
```

#### 3.2.2 多模态跑出词的cfg

Qwen2.5-VL-3B-Instruct-AWQ 为例，多模态包括vision 和llm两部分，分开实现构图，即qwen2_5_vl_vision 和 qwen2_5_vl_llm。

```
[Common]
model_path = Qwen/Qwen2.5-VL-3B-Instruct-AWQ
output_dir = ./
quant_type=w4
target=X3S_1304
app = vlchat(prompt="Describe this image.", image="10149.png")
engine=opt

[vit]
option_config='image_width=[314],image_height=[281],using_3d=false'
model_name = qwen2_5_vl_vision
// quantization_config='is_perf_mode=false'


[llm]
model_name = qwen2_5_vl_llm
quantization_config='is_perf_mode=false'

```

#### 3.2.3 其他应用

1. 产生单block的IR

```

[Common]
model_path = Qwen/Qwen1.5-1.8B-Chat
model_name = qwen2
output_dir = ./
quant_type=w4
target=X3S_1304
option_config="num_hidden_layers=1,with_lm=false"

```
2. 产生prefill 和 decoder的 单block IR

```
[Common]
model_path = Qwen/Qwen1.5-1.8B-Chat
model_name = qwen2
output_dir = ./
quant_type=w4
target=X3S_1304

[prefill_sl256_pl0]
option_config = num_hidden_layers=1,with_lm=false,use_cache=false
seq_len = 256
past_len = 0

[decoder_sl1_pl255]
option_config = num_hidden_layers=1,with_lm=false
seq_len = 1
past_len = 255
````

## 4. 添加一个模型

添加一个模型遵循的简单流程：
<div align="center">
  <img src="./images/add_new_model.png" alt="add_new_model", width=200>
</div>

### 4.1 构图

#### 4.1.1 添加一个模型构图有两种方式

1. 使用plugin的方式，该方式遵循NN-Compiler添加plugin的方式：
   1. 构图所在的文件名以aipubt_开头,
   2. 设置AIPUPLUGIN_PATH环境变量，在运行aipullm时，找到构图代码，并且完成注册。
2. 在源码上开发，因为aipullm会源码发布，所以在完成安装后，可以在AIPUBuilder/model目录下进行源码开发。

#### 4.1.2 构图OpsAPI

在构图中，需参照OpsAPI的使用文档，并且为了更加贴合torch.nn以及torch.nn.module的使用，NN Compiler也实现了AIPUBuilder.Module，参见AIPUBuilder/nn/module.py, 目前支持的Module有

['Add', 'BatchNorm', 'Concat', 'Constant', 'Conv2d', 'Conv3d', 'Crop', 'Embedding', 'Gather', 'GroupNorm', 'KVCacheModule', 'LayerNorm', 'Linear', 'MatMul',  'Mul', 'NewGELUActivation',  'QuickGELUActivation', 'RMSNorm', 'Reshape', 'Softmax', 'Tile', 'Transpose', 'Where', 'gelu', 'gelu_new', 'gelu_pytorch_tanh', 'leaky_relu','mish', 'relu', 'relu1', 'relu6', 'sigmoid', 'silu', 'swish', 'tanh']

>后续会持续扩展AIPUBuilder.nn的Module实现。

#### 4.1.3 QAT量化信息更新

如果是QAT模型，模型中自带的scale/zp如何设置到对应的tensor上？
aipullm通常是在opsapi构图的时候把weight根据weight name 从safetensors中拿出weight data，然后更新到weight_tensor.weights中，如下图
<div align="center">
  <img src="./images/update_tensor.png" alt="update_tensor", width=500>
</div>

对于AWQ或者GPTQ量化的weight，其weight name 并不是 ‘xx.weight'， 正常是 ’xx.qweight‘，所以aipullm会根据规则从weights中取出qweight 以及该tensor量化对应的qzeros/scales， 并做相应的unpack操作，一起更新到相应的tensor下。

如果配置了需要反量化，那么利用unpack_qweight, unpack_zeros， scales对数据进行反量化，然后再更新到tensor的数据下。

### 4.2 量化

量化参见：[quantize IR](#quantization)部分的介绍。

### 4.3 测试

添加完模型，量化后想对模型测试能否产生IR，以及float IR, 量化IR是否符合要求，需要对齐测试。测试得方式有两种，

1. 编写一个cfg（见cfg的使用），跑cfg，看出词。
2. 是源码调用的方式编写测试脚本：

```
from AIPUBuilder.model import AIPULLM, AppPipeLine
 config_fields = {
        "model_path": "/project/ai/scratch01/Model_Resource/huggingface/Qwen/Qwen1.5-1.8B-Chat",
        "model_type": "qwen2",
        "seq_len": 256,
        "past_len": 256,
        "quant_type": "w4",
        "target": "X3P_1304",
        "app": "pyllmchat(prompt="where is·the·capital.of.china?")",
        "engine": "exe",
        "output_dir": "./",
    }
    llm = AIPULLM.from_config(config_fields)
    llm.load_huggingface()
    ret = llm.build(quantization_type=config_fields['quant_type'], target=config_fields['target'])
    app = AppPipeLine([llm.config])
    app.forward()
```

## 5. 一些使用小技巧

|no.   |   使用场景        |  配置方式 | 说明|
|:--- | :--------------       |:----------|-----|
|1    |不保存float IR     |serialize_fp_ir=false|若是模型比较大，往往serialize时间较长，那么可以将序列化float IR的过程跳过。目前不支持跳过quantized IR，因为往往quantized IR 会用来跑opt或者exe的forward|
|2    |已经有quantized IR |aipu_binary_path=/pth/to/quantized.txt|快速跳过构图过程，进入编译或者app阶段|
|3    |已经有aipu binary  |aipu_binary_path=/pth/to/aipu.bin|快速跳过构图和编译过程，进入app阶段|
