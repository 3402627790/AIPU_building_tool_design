# cython: language_level=3
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import configparser
import sys
import os
import copy
from collections import OrderedDict
from AIPUBuilder.core import check_graph
from AIPUBuilder.logger import INFO, ERROR, WARN, DEBUG
from AIPUBuilder.model import CAUSALLMS
from AIPUBuilder.core import Target
from AIPUBuilder.model.utils import string_to_base_type
from AIPUBuilder.model.pipeline import PipeLine
from AIPUBuilder.model.qinfo import QuantType
from AIPUBuilder.model.engines import *
from AIPUBuilder.model.app import *


def check_cfg(cfg):
    ret = True
    if not os.path.exists(cfg):
        ret = False
    else:
        with open(cfg, 'r') as fi:
            lines = fi.readlines()
            lines = [l.strip().replace(' ', '') for l in lines]
            lines = [l for l in lines if len(l) != 0 and not l.startswith(('#', ',', ';', "//"))]
            if lines[0] != '[Common]':
                ERROR(f"llm cfg file should have the [Common] section field.")
                ret = False
    return ret


def check_args(all_args):
    def _check(sec_name, args, has_multi_section):
        ret = True
        model_path = args.model_path
        if model_path == "":
            ERROR(f"[{sec_name}]: model_path should be setted in config.json or settted by customer")
            ret = False
        if ret:
            if not os.path.exists(model_path):
                bp = os.environ.get("AIPU_LLM_MODEL_PATH", '/project/ai/scratch01/Model_Resource/huggingface')
                model_path = os.path.join(bp, model_path)
                if not os.path.exists(model_path):
                    ret = False
                    ERROR(f"[{sec_name}]: the LLM model path is not exists, please check the path = {model_path}")

        model_type = args.model_name
        if ret and model_type == "" and not (sec_name == "Common" and has_multi_section):
            ERROR(f"[{sec_name}]: model_name should be setted in config.json or setted by customer")
            ret = False

        if model_type != "" and model_type not in CAUSALLMS:
            ERROR(f"[{sec_name}]: {model_type} not in {CAUSALLMS.keys()}, please implement it.")
            ret = False

        # quant_type = args.quant_type
        # if ret and quant_type != "" and f"{model_type}_{quant_type}" not in MODELQUANTIZE:
        #     ERROR(
        #         f"[{sec_name}]: setted quant_type={model_type}_{quant_type} is not in {MODELQUANTIZE.keys()}, please check the setted quant_type")
        #     ret = False
        # if quant_type == "" and args.only_qt_ir:
        #     ERROR(f"[{sec_name}]: please set quant_type when enable only_qt_ir")
        #     ret = False

        target = args.target
        if target != "":
            if 'MP' in target:
                target = target.split('MP')[0]
            if target not in Target.get_support_targets():
                ERROR(f"[{sec_name}]: setted target={target} is not in {Target.get_support_targets()}, please check the setted target")
                ret = False

        if args.vocab_path != '':
            if not os.path.exists(args.vocab_path):
                ERROR(f"[{sec_name}]: setted vocab_path is not existed, please check the setted vocab_path")
                return False

        if args.aipu_binary_path != "":
            if not os.path.exists(args.aipu_binary_path):
                ERROR(f"[{sec_name}]: setted aipu_binary_path is not existed, please this path = {args.aipu_binary_path}")
                return False

        if args.quant_type != '':
            if not QuantType.is_supported_quant_mode(args.quant_type):
                ERROR(
                    f"please check the quant_type={args.quant_type}, which aipullm does not supported. only supported {QuantType.supported_quant_mode.keys()}")

        if args.quantization_config != '':
            qc = args.quantization_config
            if '=' not in qc:
                ERROR(f"[{sec_name}]: setted quantization_config format is error, please set like quantization_config=is_perf_mode=true,lmhead_fc_weight_type=fp16")
                return False

        if args.input != '':
            if args.engine == '':
                ERROR(f"[{sec_name}]: input field is setted, but engine is '', please set the engine filed.")
                return False

            inputs = args.input.replace(' ', '').split(',')
            for inp_file in inputs:
                if not os.path.exists(inp_file):
                    ERROR(f"[{sec_name}]: input file of {inp_file} does not exist, please check this file.")
                    return False

        if args.app != '':
            from AIPUBuilder.model.register import APP
            plugin = args.app.strip()
            if '(' in plugin or ')' in plugin:
                if plugin[-1] != ')':
                    ERROR(f"[{sec_name}]: setted app should with '()' for input args, and needs to end up with ')")
                    return False
                plugin_name = plugin[:plugin.index('(')]
            else:
                DEBUG(f"[{sec_name}]: setted app={plugin} does not give the input args")
                plugin_name = plugin
            if plugin_name.lower() not in APP:
                ERROR(f"[{sec_name}] app_name(={plugin_name}) should in {APP.keys()}.")
                return False

            if args.engine.lower() == 'sim' and args.simulator == '':
                ERROR(
                    f"[{sec_name}] section: when setting engine=sim,  should set simulator field, like simulator=/path/to/aipu_simulator.")
                return False
        return ret

    flag = True
    for sec_name, config_args in all_args.items():
        flag = _check(sec_name, config_args, has_multi_section=len(all_args) > 1) and flag
    return flag


def parse_cfg(args):

    defaults = args.__dict__
    with_default_config = configparser.ConfigParser(defaults=defaults)
    with_default_config.read(args.cfg)
    sections = with_default_config.sections()
    if 'Common' not in sections:
        ERROR(
            f"please check the cfg file, section [Common] must be in the cfg, but now only has {0 if len(sections) == 0 else sections} sections.")

    options = with_default_config.options('Common')
    fields = with_default_config['Common']
    for ok in options:
        ov = fields[ok]
        if ok not in ['app']:
            ov = ov.replace(" ", "")
        if ok not in defaults.keys():
            WARN(f"{ok} field does not supported in llm args, please remove or check it.")
            continue
        if len(ov) == 0:
            continue
        if hasattr(args, ok):
            args.__setattr__(ok, string_to_base_type(ov))

    all_args = OrderedDict({"Common": args})
    if len(sections) > 1:
        without_default_config = configparser.ConfigParser()
        without_default_config.read(args.cfg)
        without_default_sections = without_default_config.sections()
        without_default_sections.remove('Common')
        for sec in without_default_sections:
            cur_args = copy.deepcopy(all_args['Common'])
            fields = without_default_config[sec]
            for ck, cv in (dict(fields)).items():
                if ck not in defaults.keys():
                    WARN(f"{ck} field does not supported in llm args, please remove or check it.")
                    continue
                if ck not in ['context', 'prompt']:
                    cv = cv.replace(" ", "")
                if len(cv) == 0:
                    continue
                if hasattr(cur_args, ck):
                    cur_args.__setattr__(ck, string_to_base_type(cv))

            all_args.update({sec: cur_args})

    return all_args


def enable_gsim_type(value):
    if value.lower() in ('true', 't', '1'):
        return True
    elif value.lower() in ('false', 'f', '0'):
        return False
    else:
        return value


def main():
    args = argparse.ArgumentParser(prog="AIPULLM")
    args.add_argument('-v', '--version', action='version', version='AIPULLM version 0.2')
    args.add_argument('-c', '--cfg', default="",  help='AIPULLM config file')
    args.add_argument("--model_path", type=str, default="", help='model path , like meta-llama/llama2_chn')
    args.add_argument("--config_name", type=str, default="config.json", help='json file name')
    args.add_argument('--model_name', type=str, default="",
                      help='if model_name is "", the model_type in config.json will set to LlmInfo. \
                            The model_type is relative with building the compass IR, tokenizer in c-runtime etc.')
    args.add_argument('--seq_len', type=int, default=256,
                      help='the current seq_len in Compass IR and the max sequence length in chat.')
    args.add_argument('--past_len', type=int, default=256,
                      help='the past_len in Compass IR of cache.')
    args.add_argument('--quant_type', type=str, default="",
                      help='the quantization type for float IR, like w4 is \
                            meaning: activation fp16, weight int4.')
    args.add_argument('--target', type=str, default="", help="zhouyi target")
    args.add_argument('--file_type', type=str, default="",
                      help="specify the file type for weight, like safetensors, torch.bin or GGUF file. \
                            when has multi-gguf files with different quantization type, \
                            this field should be with quantization type.")
    args.add_argument('--use_cache', type=bool, default=True, help="whether enable kv cache")
    args.add_argument('--fill_llminfo', type=bool, default=False,
                      help="whether fill llm information, like embedding data to aipubinary")
    args.add_argument('--kvcache_optimization', type=bool, default=False,
                      help="whether enable kv cache optimization, like cacheupdate op")
    args.add_argument('--vcache_axis', type=int, default=2,
                      help="set v cacheupdate concat axis, default value is 2.")
    args.add_argument('--enable_subgraph', type=bool, default=False,
                      help="split the subgraph of forward function which decorated by @gen_subgraph to subgraph")
    args.add_argument('--output_dir', type=str, default='./', help="the output file's directory")
    args.add_argument('--max_wt_size', type=int, default=4096, help="the aipubinary size")
    args.add_argument('--vocab_path', type=str, default='', help="set the vocab file path")
    args.add_argument('--tokenizer_path', type=str, default='', help="set the tokenizer path for gguf.")
    args.add_argument('--app', type=str, default='',
                      help='model app plugin, its configuration format like: app=appname(arg_name0="arg_value0", arg_name1="arg_value1", ...),\
                           which is like function definition in the right side of the equals sign.\
                           The charater of appname should be in [a-zA-Z_0-9]. \
                           The contents within the parentheses are the passed-in parameter variables. \
                           The variables should be with arg_key=arg_value format, and the arg_value should be enclosed with double quotes. \
                           The variables are splitted by comma. if arg_value has double quote, please escape with a backslash. \
                           After parsing, all arg_keys and arg_values will be as the initial parameter of appname. \
                           You can reference the qwen2_aipullm.cfg example.')
    args.add_argument('-i', '--input', type=str, default='',
                      help='The inputs data file for the graph, if you have multiple inputs, use comma "," to split them. \
                      and now only support npy file format.')
    args.add_argument('--engine', type=str, default='', help="set the forward engine, like opt, exe, fpga.")
    args.add_argument('--option_config', type=str, default='',
                      help="configuration field in config.json wants to set, and use ',' to separate. \
                            In addition to the fields defined in cofig.json(like num_hidden_layers=1), aipullm can also configure: \
                            num_hidden_layers(indicates block numbers, which should lookup the config.json to decide this field name, like num_layers in chatglm), can be setted integer, list, or nest-list, when integer, will generate range(0, num_hidden_layers) block index, when list, like [2,5], will generate [2, 3, 4, 5] block index, when nest-list, like [[2,2],[5,6]], will generate [2, 5, 6] block index. \
                            use_cache=false, will generate un-kvcache IR; \
                            with_lm=false, will exclude the lm_head part in IR; \
                            gptq_gidx_to_fc=true, will change the gidx of gptq to fc when desc_act=true in gptq, otherwise, to gather; \
                            force_float_ir=true, will dequantize the weight if its is quantized and serialize the float IR; \
                            broadcast_matmul=false, will use matmul's broadcast feature to eliminate tile op for GQA; \
                            swap_matmul_input=false, will try to swap certain input dims of matmul to improve the performance;")
    args.add_argument('--quantization_config', type=str, default='weight_only_quantization_op=[fullyconnected]',
                      help="set quantization feilds, now supported: \
                            lmhead_fc_weight_type=w4 or fp16, will set fc weight in lmhead to int4 or fp16; \
                            is_perf_mode=true, will enable non-linear activation with aiff implementation for higher performance; \
                            cast_dtypes_for_lib=false, if true, will insert cast op depending on the data type of inputs, which is not implemented by lib; \
                            quantize_method_for_weight='', which can set: per_tensor_symmtric_restricted_range, per_channel_symmetric_restricted_range, per_channel_asymmetric, per_vector_symmetric_restricted_range, per_vector_asymmetric; \
                            weight_block_size=-1, if in [32, 64, 128, 256], will enable aipu blockwise quantization. \
                            weight_only_quantization_op=[fullyconnected], if use quant_type=w4, only fc layer is configurated to weight only quantization.\
                                                  if want to add other layer, please set like weight_only_quantization=[fullyconnected, convolution]")
    args.add_argument('--builder_config', type=str, default='',
                      help="set aipugb compile feilds, when want to set more aipugb commands, \
                            using builder_config=key1=value1,key2=value2,key3=value3 to set.")
    args.add_argument('--aipu_binary_path', type=str, default='',
                      help="if setted aipu_binary_path, the process of building fpIR/QIR/aipubin all skips, \
                            directly run this aipubin")
    args.add_argument('--simulator', type=str, default='',
                      help="the simulator executable path. when engine=sim, should set the simulator field.")
    args.add_argument('--enable_gsim', type=enable_gsim_type, default=True, help="whether skip gsim function.")
    args.add_argument('--only_fp_ir', action="store_true", help="generate float IR only")
    args.add_argument('--only_qt_ir', action="store_true", help="generate float IR and quantized IR, not build and run")
    args.add_argument('--serialize_fp_ir', type=bool, default=True, help="whether serialize float IR")

    options = args.parse_args(sys.argv[1:])

    ret = True
    if options.cfg != "":
        if not os.path.exists(options.cfg):
            ERROR(f"[LLM] cfg file does not exist, please check the cfg path ={options.cfg}.")
            return False
        INFO("[LLM] parse and check args doing...")
        all_options = parse_cfg(options)
        if not check_args(all_options):
            return False

        llm_pipeline = PipeLine(all_options)
        ret = llm_pipeline()

    return ret


if __name__ == "__main__":
    sys.exit(main())
