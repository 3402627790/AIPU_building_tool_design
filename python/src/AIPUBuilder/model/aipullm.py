import os
import sys
import numpy as np
import torch

from typing import Union
from AIPUBuilder.core import (Graph, aipugb, Target, use_cuda, check_graph,
                              OpType, TensorList, SubGraphNode, AIPUELF, LlmInfo, BuilderConfig)
from AIPUBuilder.logger import ERROR, INFO, DEBUG, WARN
from AIPUBuilder import ops
from AIPUBuilder.simplifier import gsim as aipu_gsim
from AIPUBuilder.model.register import CAUSALLMS, MODELQUANTIZE
from AIPUBuilder.model.modelbuild import ModelBuild
from AIPUBuilder.model.qinfo import ModelQuantize
from AIPUBuilder.model.config import PretrainedConfig
from AIPUBuilder.model.passes import conv2d_to_fc, convert_group_fp16_scales_to_fp8
from AIPUBuilder.model.llm.llmutils import get_embedding_data, build_ir_postprocess


class AIPULLM(object):

    def __init__(self,
                 model_path=None,
                 seq_len=None,
                 use_cache=None,
                 model_type=None,
                 output_dir=None,
                 kvcache_optimization=False,
                 max_wt_size=4096,
                 past_len=256,
                 **kwargs):

        self.hf_config = PretrainedConfig()
        if 'config' in kwargs:
            self.config = kwargs['config']
        else:
            if model_path is not None and model_type is not None:
                self.hf_config.update({
                    'model_path': model_path,
                    'model_type': model_type,
                    'use_cache': use_cache if use_cache is not None else True,
                    'output_dir': output_dir if output_dir is not None else './',
                    'seq_len': seq_len if seq_len is not None else 256,
                    'past_len': past_len,
                    'max_len': past_len + (seq_len if seq_len is not None else 256),
                    'max_wt_size': max_wt_size,
                    'kvcache_optimization': kvcache_optimization,
                })
                self.hf_config.update(kwargs)
        self.model_type = self.hf_config.get('model_type')
        self.output_dir = self.hf_config.get('output_dir', './')
        self.model = None
        self.aipu_bin = None

    @classmethod
    def from_config(cls, config):
        llm = cls(config=config)
        return llm

    def set_model(self, model):
        if self.model is None:
            self.model = model

    @property
    def config(self):
        return self.hf_config

    @config.setter
    def config(self, hf_config):
        if not isinstance(hf_config, (PretrainedConfig, dict)):
            ERROR(f"AIPULLM only use PretrainedConfig instance, but now is {type(hf_config)}")
            return
        _config = hf_config
        if isinstance(_config, dict):
            if _config.get('model_path', '') != '':
                json_path = os.path.join(_config.get('model_path'), 'config.json')
                _config = PretrainedConfig.from_json_file(json_path)
                _config.update(hf_config)
            else:
                _config = PretrainedConfig(**_config)
        self.hf_config = _config

    def load_huggingface(self):
        causal_instance = CAUSALLMS[self.hf_config.model_type]
        model_instance, _ = causal_instance.from_pretrained(self.hf_config)
        self.model = model_instance()
        build_ir_postprocess(self.model, self.hf_config)

        INFO(f"[LLM] begin to generate the compass IR of {self.model_type}...")
        ir_prefix = self.hf_config.get('ir_prefix', '')
        if not isinstance(self.model, Graph):
            ERROR(f"none Graph object returned after loading")
        else:
            self._graph_opt(self.model)
            self._check_aipu_graph(self.model)
            if self.hf_config.get("serialize_fp_ir", True):
                self._serialize(self.model, *self._output_file(prefix=ir_prefix), stage='float')

        return self.model

    def build(self, quantization_type, target, backend=None):
        """
        build method is including three part: quantize, gsim and gbuild.
        """
        ret = True

        enable_ds = self.hf_config.get('enable_ds', True)
        ir_prefix = self.hf_config.get('ir_prefix', '')
        output_dir = self.hf_config.get('output_dir', './')
        self.model.serialize_scale_zp = True

        if quantization_type is not None and quantization_type != "":
            ret = self._quantize(self.model, quantization_type, self.hf_config) and ret
            self._graph_opt_after_quantize(self.model)
            self._serialize(self.model,
                            *self._output_file(prefix=ir_prefix,
                                               postfix=quantization_type),
                            stage='quantized',
                            postfix=f'{quantization_type}')

        if self.hf_config.get('enable_gsim', False):
            from AIPUBuilder.core import BuilderConfig
            BuilderConfig.set("show_location", True)
            BuilderConfig.set("enable_ds", self.hf_config.use_cache and enable_ds)
            self._gsim(self.model)
            for n in self.model.nodes:
                if '.is_dynamic' in n.attrs:
                    n.attrs.pop('.is_dynamic')
            if '.is_dynamic' in self.model.attrs:
                self.model.attrs.pop(".is_dynamic")
            self._serialize(self.model,
                            *self._output_file(prefix=ir_prefix,
                                               postfix=f"{quantization_type}_s"),
                            stage='gsim',
                            postfix='_s')

        if target is not None and target != "":
            output = f'./{self.model_type}{"_"+ir_prefix if len(ir_prefix) else ""}{"_"+quantization_type if quantization_type is not None else ""}_aipu.bin'
            self.aipu_bin = os.path.join(output_dir, output[2:]) if output_dir is not None else output
            self._pre_gbuilder(self.model)
            ret = self._gbuilder(self.model,
                                 self.aipu_bin,
                                 self.model_type,
                                 target,
                                 self.hf_config,
                                 enable_ds=enable_ds) and ret

        is_fill_llm_info = self.hf_config.get('fill_llminfo', False)
        if is_fill_llm_info:
            ret = self.fill_llm_info(self.aipu_bin, self.aipu_bin) and ret
        self._remove_graph_betensor()
        return ret

    def fill_llm_info(self, aipu_bin_or_elf, output_dir):
        INFO("\t Fill the llm_info to binary")
        if isinstance(aipu_bin_or_elf, str) and os.path.exists(aipu_bin_or_elf):
            elf = AIPUELF()
            elf.load(aipu_bin_or_elf)
            ret = self._fill_info_to_aipubin(elf, output_dir)
        elif isinstance(aipu_bin_or_elf, AIPUELF):
            ret = self._fill_info_to_aipubin(aipu_bin_or_elf, output_dir)
        else:
            ERROR(f"[aipullm] fill_llm_info only supports AIPUELF or a existed path of aipu binary.")
            ret = False
        return ret

    def _graph_opt(self, model):
        nodes = model.nodes[:]
        for n in nodes:
            if len(n.children) == 0 and len(n.parents) == 0:
                if n.outputs[0] in model.output_tensors:
                    ots = list(model.output_tensors)
                    idx = ots.index(n.outputs[0])
                    model.output_tensors = TensorList(ots[:idx]+ots[idx+1:])
                n.clear_inputs()
                n.clear_outputs()
                model.remove_node(n)

    def _graph_opt_after_quantize(self, model):
        conv2d_to_fc(model)
        afp8_type = self.hf_config.aipu_quant_config.get('afp8_type', '')
        if afp8_type in ['float8_e4m3fn', 'float8_e5m2']:
            convert_group_fp16_scales_to_fp8(model, afp8_type)

    def _check_aipu_graph(self, g):
        if not check_graph(g):
            ERROR(f"check graph = {g} fialed.")
            return False
        return True

    def _remove_graph_betensor(self, graph=None):

        def _remove_one_node_betensor(n):
            for ot in n.outputs:
                ot.remove_betensor()
            for _, ct in n.constants.items():
                ct.remove_betensor()

        def _remove_node_betensor(n):
            if SubGraphNode.is_subgraph_node(n):
                sbg = SubGraphNode.from_node(n).subgraph
                for sbgn in sbg.nodes:
                    _remove_node_betensor(sbgn)
            else:
                _remove_one_node_betensor(n)

        graph = self.model if graph is None else graph
        if graph is None:
            return
        for n in graph.nodes:
            _remove_node_betensor(n)

        if use_cuda():
            torch.cuda.empty_cache()

    def _quantize(self, model, quantization_type, hf_config):

        quant_method = ''
        if hf_config is not None and hf_config.get('hf_quantization_config', None) is not None:
            quant_method = hf_config.get('hf_quantization_config', {}).get('quant_method', '')
        qname = f"{self.model_type}_{quantization_type}" + (quant_method if quant_method == '' else f"_{quant_method}")
        INFO(f"[LLM] begin to quantize the compass IR with {quantization_type} of {qname}...")
        mq = MODELQUANTIZE[qname] if qname in MODELQUANTIZE else ModelQuantize
        self.model = mq(model=self.model, model_type=self.model_type,
                        quantization_type=quantization_type, hf_config=hf_config)()
        return True

    def _gsim(self, model):
        def _show(before, after):
            header = f"\t GSim simplified result:\n" + "-" * 72 + "\n"
            s = ""
            for c in after:
                d = 0
                if c in before:
                    d = before.pop(c)
                d = after[c] - d
                if d != 0:
                    s += ("\t%20s: %+4d\n" % (str(c), d))
            for c in before:
                s += ("\t%20s: %+4d\n" % (str(c), 0 - before.get(c, 0)))
            if len(s):
                s += "-" * 72
                INFO(header + s)

        def _statistic_nodes_num(g: Graph):
            ret = dict()
            for n in g.nodes:
                ret[n.type] = ret.get(n.type, 0) + 1
            return ret

        def simplify(graph):
            return aipu_gsim(graph, True)

        def full_simplify(g: Graph):
            import time
            FULL_SIM_TIMEOUT = 120.  # 30s
            try:
                t0 = time.time()
                while simplify(g):
                    t1 = time.time()
                    if t1 - t0 > FULL_SIM_TIMEOUT:
                        DEBUG('full_simplify is timeout(run more than %d seconds)!' % FULL_SIM_TIMEOUT)
                        return True
            except KeyboardInterrupt:
                return simplify(g)

        INFO(f"[LLM] begin to simplifier the compass {self.model_type} IR...")
        before_ns = _statistic_nodes_num(model)
        if isinstance(self.hf_config.get('enable_gsim', False), str) and self.hf_config.get('enable_gsim', '').lower() == 'full':
            ret = full_simplify(model)
        else:
            ret = simplify(model)
        after_ns = _statistic_nodes_num(model)
        if ret:
            model.topological_sort()
            model.reset_layer_id()
        _show(before_ns, after_ns)
        return ret

    def _gbuilder(self, model, aipu_bin, model_type, target, hf_config, enable_ds=True):
        INFO(f"[LLM] begin to compile the compass {self.model_type} IR with {target} target...")
        mb = ModelBuild(model, model_type, target, hf_config)
        ret = mb(aipu_bin, enable_ds)
        compiled_graph = mb.get_compiled_graph()
        self._remove_graph_betensor(compiled_graph)
        self.hf_config.update({'aipugb_g': compiled_graph, 'aipu_binary_path': aipu_bin})
        return ret

    def _pre_gbuilder(self, model):
        # unfold subgraph before complier the graph to aipubinary
        nodes = self.model.nodes
        for n in nodes[:]:
            try:
                if 'is_call' in n.attrs and n.attrs['is_call']:  # n.type == OpType.Call:
                    sgn = SubGraphNode.from_node(n)
                    sgn.unfold()
            except Exception as e:
                ERROR(f"when unfold subgraph in aipullm, failed and the error is {e}")

        for n in self.model.nodes:
            for ot in n.outputs:
                if '_ordered_input_id' in ot._attrs:
                    ot._attrs.pop('_ordered_input_id')

    def _output_file(self, prefix='', postfix=''):
        prefix = '' if prefix == '' else f"_{prefix}"
        postfix = '' if postfix == '' else f"_{postfix}"
        irt = f"{self.model_type}{prefix}{postfix}.txt"
        irb = f"{self.model_type}{prefix}{postfix}.bin"
        if self.output_dir is not None:
            irt = os.path.join(self.output_dir, irt)
            irb = os.path.join(self.output_dir, irb)
        return irt, irb

    def _serialize(self, g: Graph, irt: str, irb: str, stage='', postfix=''):
        if len(postfix):
            for n in g.nodes:
                if n.type == OpType.SubGraphOp:
                    n.params['call_subgraph'] = n.params['call_subgraph'] + postfix
                    if 'txt_path' in n.attrs and len(n.attrs['txt_path']):
                        txt_path = n.attrs['txt_path']
                        pos = txt_path.rindex('.')
                        n.attrs['txt_path'] = txt_path[:pos] + postfix + txt_path[pos:]
        g.serialize_scale_zp = True
        g.serialize(irt, irb)
        self.hf_config.update({f'{stage}_ir_path': irt})

    @staticmethod
    def query_support(model_type, quantization_type, quant_method=''):
        qname = f"{model_type}_{quantization_type}" + (quant_method if quant_method == '' else f"_{quant_method}")
        return qname in MODELQUANTIZE

    def get_aipu_binary_path(self):
        return self.aipu_bin

    def _fill_info_to_aipubin(self, elf, output_bin):
        def _get_emb_data():
            return get_embedding_data(self.config, self.model_type, True)
        ret = True
        has_vocab = True

        if not hasattr(self.hf_config, 'vocab_json'):
            WARN(f"{self.hf_config.model_type} does not provide the vocab_json file")
            vocab_path = os.path.join(os.path.dirname(__file__), self.model_type, f'{self.model_type}_vocab.json')
        else:
            vocab_path = self.hf_config.vocab_json

        if vocab_path == '' or not os.path.exists(vocab_path):
            WARN(f"please provide the vocab_path for {self.model_type}.")
            has_vocab = False

        elf_target = elf.target
        INFO(f"\t The target in elf bin is {elf_target}.")
        if elf_target != self.hf_config.get('target', ''):
            DEBUG(
                f"written target(={elf_target}) in elf bin != the compiled target(={self.hf_config.get('target', '')}) in cfg file.")

        linfo = LlmInfo()
        linfo.hidden_size = self.hf_config.hidden_size
        linfo.cache_head = self.hf_config.num_key_value_heads
        linfo.cache_dim = self.hf_config.get('head_dim',
                                             self.hf_config.hidden_size // self.hf_config.num_attention_heads)
        linfo.logits_dim = self.hf_config.vocab_size
        linfo.model_type = self.hf_config.model_type
        linfo.use_cache = self.hf_config.use_cache
        linfo.hidden_layers = self.hf_config.num_hidden_layers

        embedding_data = _get_emb_data()
        if embedding_data is None:
            WARN("when modelbuild can't find embedding data in config, so we can not write embedding data to LlmInfo.")
        else:
            try:
                linfo.set_embd_wt(embedding_data.detach().half().cpu().numpy())
            except Exception as e:
                ERROR(f"write embedding to LlmInfo failed in {self.model_type} and error is {e}")
                ret = False
        if has_vocab:
            try:
                with open(vocab_path, 'r') as f:
                    content = f.read()
                    byte_stream = bytes(content, encoding='utf-8')
                    linfo.set_vocab(byte_stream, len(byte_stream))
            except Exception as e:
                ERROR(f"write vocab to LlmInfo failed in {self.model_type} and error is {e}")
                ret = False

        elf.llminfo = linfo
        elf.write(output_bin)

        # elf = AIPUELF()
        # elf.load(output_bin)

        # elf_target = elf.target
        # INFO(f"\t The target in elf bin is {elf_target}.")
        return ret
