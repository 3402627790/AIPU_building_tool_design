import re
import os
from AIPUBuilder.core import *  # noqa
from AIPUBuilder import ops
from AIPUBuilder.logger import ERROR
from ...utils import gen_quantization, set_quantization
from ...register import register_model_quantize
from ...qinfo import ModelQuantize, QConf
import numpy as np
import torch

@register_model_quantize(['deepseek_moe_w4'])
class DeepSeekMoeQuantize(ModelQuantize):
    def __init__(self, model, model_type, quantization_type='w4', hf_config=None):
        super().__init__(model, model_type, quantization_type, hf_config)
        self.bits = int(''.join(list(filter(str.isdigit, quantization_type))))
        if self.bits not in [4, 8]:
            ERROR(f"when activation is fp16, weight quantization bits should be in [4, 8], but now is {self.bits}")
        self.hf_weights = hf_config.weights

    def get_quant_config(self):
        lm_head_fp16 = False
        is_perf_mode = True
        if hasattr(self.hf_config, 'aipu_quant_config'):
            qc = self.hf_config.aipu_quant_config
            if 'lmhead_fc_weight_type' in qc:
                wt = qc['lmhead_fc_weight_type']
                if wt == 'fp16':
                    lm_head_fp16 = True
            if 'is_perf_mode' in qc:
                is_perf_mode = qc['is_perf_mode']

        return lm_head_fp16, is_perf_mode

    def update_qinfo(self):
        lm_head_fp16, is_perf_mode = self.get_quant_config()
        activation_type = self.hf_config.get('activation_type', 'float16')
        act_dtype = Dtype(activation_type.lower())

        def _set_node_qinfo(n):
            if n.type in [OpType.Convolution, OpType.FullyConnected]:
                w = n.constants['weights']
                if 'lm' in w.name and lm_head_fp16:
                    # lm head fc uses fp16 * fp16
                    n.attrs['trigger_float_op'] = NodeParamValue(str(Dtype.FP16))
                elif 'mlp.gate.weight' in w.name:
                    n.attrs['trigger_float_op'] = NodeParamValue(str(Dtype.FP16))
                else:
                    w.quantization.bits = self.bits
                    w.quantization.scheme = Scheme.Asymmetric if self.bits < 8 else Scheme.Symmetric
                    w.quantization.granularity = Granularity.Per_Group
                    w.quantization.quantized_range = QuantizedRange.Full_Range
                    w.quantization.aipu_dtype = Dtype.ALIGNED_INT4 if self.bits < 8 else Dtype.INT8
                    w.quantization.group = w.shape[0] * (w.shape[1] // 32)
                    trigger_mode = f"{activation_type.upper()}_ACT_INT{self.bits}_WHT"
                    n.attrs['trigger_float_op'] = NodeParamValue(trigger_mode)
            else:
                n.attrs['trigger_float_op'] = NodeParamValue(activation_type)

            n.attrs['unquantifiable'] = True
            n.params['unquantifiable'] = True

            if n.type in [OpType.Softmax]:
                n.attrs['min_compatible_zhouyi_target'] = 'X3P_1304'
                n.attrs['approx_params'] = NodeParamValue(int(1))
                n.attrs['lut_items_in_bits'] = 9
            if n.type in [OpType.Activation]:
                n.attrs['min_compatible_zhouyi_target'] = NodeParamValue('X3P_1304')
                n.attrs['approx_params'] = NodeParamValue(0)
                if n.params['method'].lower() == 'silu':
                    n.attrs['approx_params'] = NodeParamValue([1]) if is_perf_mode else NodeParamValue([0])

                n.attrs['lut_items_in_bits'] = 9
            if n.type == OpType.Input:
                if n.outputs[0].dtype.is_float:
                    n.outputs[0].dtype = act_dtype

            for ot in n.outputs:
                if ot.io_tensor and ot.dtype.is_float:
                    ot.dtype = act_dtype

        def set_qinfo(n):
            if SubGraphNode.is_subgraph_node(n):
                sbg = SubGraphNode.from_node(n).subgraph
                for sbgn in sbg.nodes:
                    set_qinfo(sbgn)
            else:
                _set_node_qinfo(n)

        for node in self.aipu_g.nodes:
            set_qinfo(node)