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


# @register_model_quantize(['qwen2_w4', 'qwen2_w8', 'qwen2_opt_w4'])
class Qwen2W4ModelQuantize(ModelQuantize):
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
                else:
                    w.quantization.bits = self.bits
                    w.quantization.scheme = Scheme.Asymmetric if self.bits < 8 else Scheme.Symmetric
                    w.quantization.granularity = Granularity.Per_Channel
                    w.quantization.quantized_range = QuantizedRange.Full_Range
                    w.quantization.aipu_dtype = Dtype.ALIGNED_INT4 if self.bits < 8 else Dtype.INT8
                    w.quantization.group = w.shape[0]
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
                sbg = n.subgraph
                for sbgn in sbg.nodes:
                    set_qinfo(sbgn)
            else:
                _set_node_qinfo(n)

        for node in self.aipu_g.nodes:
            set_qinfo(node)


# @register_model_quantize('qwen2_w4_awq')
class Qwen2W4AWQModelQuantize(ModelQuantize):
    def __init__(self, model, model_type, quantization_type='w4', hf_config=None):
        super().__init__(model, model_type, quantization_type, hf_config)
        self.hf_weights = hf_config.weights

    def update_qinfo(self):
        activation_type = self.hf_config.get('activation_type', 'float16')
        act_dtype = Dtype(activation_type.lower())
        wm = {'weights': 'weight'}
        group_size = self.hf_config.hf_quantization_config['group_size']
        for n in self.aipu_g.nodes:
            for ck, cv in n.constants.items():
                if ck in wm:
                    hf_k = wm[ck]
                    hf_name = "model." + cv.name
                    qw_name = hf_name.replace(hf_k, f'q{hf_k}')
                    if qw_name in self.hf_weights:
                        qw_data = self.hf_weights.get(qw_name)
                        qw_scale = self.hf_weights.get(hf_name.replace(hf_k, 'scales'))
                        qw_zp = self.hf_weights.get(hf_name.replace(hf_k, 'qzeros'))

                        # unpack_w = awq_unpack_int32_into_int8(qw_data)
                        # unpack_zp = awq_unpack_int32_into_int8(qw_zp)
                        unpack_w = awq_gemm_unpack_int32_into_int8(qw_data)
                        unpack_zp = awq_gemm_unpack_int32_into_int8(qw_zp)

                        # gptq scale/zp ==> aipu scale/zp
                        qw_scale = 1.0 / qw_scale
                        unpack_zp = 0 - unpack_zp

                        new_weight = Tensor(unpack_w.T.cpu().numpy().astype(np.int8))
                        new_weight.name = cv.name
                        new_weight.quantization.scales = qw_scale.T.contiguous().float().flatten().cpu().numpy().tolist()
                        new_weight.quantization.offsets = unpack_zp.T.int().flatten().cpu().numpy().tolist()
                        n.constants[ck] = new_weight
                        n.constants[ck].quantization.aipu_dtype = Dtype.ALIGNED_INT4
                        n.params['weight_block_size'] = group_size

            n.attrs['unquantifiable'] = True
            n.attrs['trigger_float_op'] = NodeParamValue(activation_type)
            n.params['unquantifiable'] = True

            if n.type == OpType.Input:
                if n.outputs[0].dtype.is_float:
                    n.outputs[0].dtype = act_dtype

            for ot in n.outputs:
                if ot.io_tensor and ot.dtype.is_float:
                    ot.dtype = act_dtype


@register_model_quantize(['qwen2_w8a8', 'qwen2_w16a16', 'qwen2_w16a16fp16', 'qwen2_w8a8fp16'])
class Qwen2W8A8ModelQuantize(ModelQuantize):
    def __init__(self, model, model_type, quantization_type='w8a8', hf_config=None):
        super().__init__(model, model_type, quantization_type, hf_config)

        self.model_path = os.path.join(os.path.dirname(__file__))
        json_file = os.path.join(self.model_path, f"{self.model_type}_{quantization_type}_qinfo.json")
        self.qconfig = QConf(json_file)

    def update_qinfo(self):
        self._update_qinfo_for_quantized(self.aipu_g, self.qconfig)
        # self._insert_cast(self.aipu_g, self.qconfig)
        self.eliminate_scale_normalization()
        trigger_float_op = self.get_float_ops()
        if len(trigger_float_op):
            self.insert_quantize_and_dequantize(trigger_float_op)
            self._update_qinfo_for_float(self.aipu_g, self.qconfig, trigger_float_op)
        self.transparent_quantization()

    def _update_qinfo_for_quantized(self, aipu_g, qconfig):
        for n in aipu_g.nodes:
            # update input tensors
            if n.type == OpType.Input:
                for k in ['embeded', 'position_id', 'attention_mask']:
                    if k in n.outputs[0].name.lower():
                        set_quantization(n.outputs[0], qconfig, f'{k}_0')

                    if k == 'attention_mask' and len(n.children) == 1 and n.children[0].type == OpType.Cast:
                        set_quantization(n.children[0].outputs[0], qconfig, f"{k}_0")
                        cn = n.children[0]
                        if len(cn.children) == 1 and cn.children[0].type == OpType.Tile:
                            set_quantization(cn.children[0].outputs[0], qconfig, f"{k}_0")

                t_name = n.outputs[0].name
                if '.key' in t_name or '.value' in t_name:
                    bidx = re.findall(r"\d+", t_name)
                    if len(bidx):
                        idx = int(bidx[0])
                        opt_t_name = f"cache_k_{idx}_0" if '.key' in t_name else f"cache_v_{idx}_0"
                        set_quantization(n.outputs[0], qconfig, opt_t_name)

            if n.type == OpType.Concat:
                # kvcache's concat
                pn_type = [pn.type for pn in n.parents]
                if OpType.Input in pn_type:
                    cache_inp_idx = pn_type.index(OpType.Input)
                    cache_n = n.parents[cache_inp_idx]
                    t_name = cache_n.outputs[0].name
                    if '.key' in t_name or '.value' in t_name:
                        bidx = re.findall(r"\d+", t_name)
                        if len(bidx):
                            idx = int(bidx[0])
                            opt_t_name = f"/layers.{idx}/self_attn/Concat_2_0" if '.key' in t_name else f"/layers.{idx}/self_attn/Concat_3_0"
                            set_quantization(n.outputs[0], qconfig, opt_t_name)

            if n.type == OpType.Constant:
                # update rotary_emb_sin/cos
                rule0 = r'^layers.(.*).attn.rotary_emb.sin_weight'
                rule1 = r'^layers.(.*).attn.rotary_emb.cos_weight'
                matched0 = re.match(rule0, n.constants['weights'].name)
                matched1 = re.match(rule1, n.constants['weights'].name)
                if matched0 is not None:
                    # name = matched0.group().replace('.', '/')[:-10].replace('layers/', 'layers.')
                    opt_w_name = f"Constant_683_0weights"
                    set_quantization(n.outputs[0], qconfig, opt_w_name)
                    set_quantization(n.constants['weights'], qconfig, opt_w_name)

                    # update sin/cos gather op
                    set_quantization(n.children[0].outputs[0], qconfig, opt_w_name)

                if matched1 is not None:
                    # name = matched1.group().replace('.', '/')[:-10].replace('layers/', 'layers.')
                    opt_w_name = f"Constant_682_0weights"
                    set_quantization(n.outputs[0], qconfig, opt_w_name)
                    set_quantization(n.constants['weights'], qconfig, opt_w_name)

                    # update sin/cos gather op
                    set_quantization(n.children[0].outputs[0], qconfig, opt_w_name)

            # update fc
            if n.type == OpType.FullyConnected:
                w_name = n.constants['weights'].name
                # assert 'model.layers' in w_name, f"{w_name} is not the formated name"
                w_name = w_name.replace('.', '/').replace('weight', 'MatMul').replace('layers/', 'layers.')

                opt_w_name = f"/{w_name}_0weights"
                opt_b_name = f"/{w_name}_0biases"
                opt_t_name = f"/{w_name}_0"
                set_quantization(n.constants['weights'], qconfig, opt_w_name)
                set_quantization(n.constants['biases'], qconfig, opt_b_name)
                set_quantization(n.outputs[0], qconfig, opt_t_name)
                # gate_proj + activation_silu
                if len(n.children) == 1 and n.children[0].type == OpType.Activation and n.children[0].params['method'] == 'SILU':
                    silu_n = n.children[0]
                    silu_t_name = f"{opt_t_name[:-1]}post_reshape_0"
                    set_quantization(silu_n.outputs[0], qconfig, silu_t_name)
                    bits = silu_n.outputs[0].quantization.aipu_dtype.bits
                    lut_bits = min(12, bits)
                    silu_n.attrs['lut_items_in_bits'] = NodeParamValue(lut_bits)

                    silu_child_n = silu_n.children[0]
                    if silu_child_n.type == OpType.Eltwise:
                        opt_e_t_name = f"{opt_t_name[:opt_t_name.index('mlp')]}mlp/Mul"
                        set_quantization(silu_child_n.outputs[0], qconfig, opt_e_t_name)

            if n.type == OpType.RMSNorm:
                w_name = n.constants['weights'].name
                # assert 'model.layers' in w_name, f"{w_name} is not the formated name"
                w_name = w_name.replace('.', '/').replace('weight', 'Mul_1_0').replace('layers/', 'layers.')
                opt_w_name = f"/{w_name}weights"
                opt_t_name = f"/{w_name}"
                set_quantization(n.constants['weights'], qconfig, opt_w_name)
                set_quantization(n.outputs[0], qconfig, opt_t_name)

            if n.type == OpType.Softmax:
                ld = 0 if n.name.lower() == 'softmax' else int(n.name.lower().split('_')[-1]) + 1
                s_opt_t_name = f'/layers.{ld}/self_attn/Softmax_0'
                set_quantization(n.outputs[0], qconfig, s_opt_t_name)

                # updata eltwise_add which uses to add mask
                pn = n.parents[0]
                if pn.type == OpType.Eltwise and pn.params['method'] == 'ADD':
                    opt_t_name = f"{s_opt_t_name[:-9]}Add_2_0"
                    set_quantization(pn.outputs[0], qconfig, opt_t_name)

            if n.type == OpType.MatMul:
                opt_t_name = n.name[:n.name.index('_matmul')].replace('.', '/').replace('layers/', 'layers.')
                opt_t_name = f"/{opt_t_name}/MatMul_1_0" if 'kqv' in n.name else f"/{opt_t_name}/MatMul_0"
                set_quantization(n.outputs[0], qconfig, opt_t_name)
                # if n.name.endswith('kq'):
                #     n.outputs[0].quantization.scales= [ n.outputs[0].quantization.scales[0] / 0.088388]
            if n.type == OpType.Eltwise:
                if len(n.children):
                    # if child_n.type == OpType.Softmax: # has done in softmax
                    #     pass
                    child_types = [cn.type for cn in n.children]
                    if OpType.RMSNorm in child_types:
                        child_n = n.children[child_types.index(OpType.RMSNorm)]
                        w_name = child_n.constants['weights'].name.replace('.', '/').replace('layers/', 'layers.')
                        if 'post_attention_layernorm' in w_name:
                            prefix = w_name[:w_name.index('post_attention_layernorm')]
                            eltwise_opt_t_name = f"/{prefix}Add_0"
                            set_quantization(n.outputs[0], qconfig, eltwise_opt_t_name)
                        elif 'input_layernorm' in w_name:
                            prefix = w_name[:w_name.index('input_layernorm')]
                            lid = prefix[:-1].split('.')[-1]
                            prefix = prefix.replace(lid, str(int(lid) - 1))
                            eltwise_opt_t_name = f"/{prefix}Add_1_0"
                            set_quantization(n.outputs[0], qconfig, eltwise_opt_t_name)
                        elif 'normlm/norm' in w_name:
                            eltwise_opt_t_name = '/layers.23/Add_1_0'
                            set_quantization(n.outputs[0], qconfig, eltwise_opt_t_name)

                    if OpType.FullyConnected in child_types and len(child_types) == 1:
                        child_n = n.children[0]
                        w_name = child_n.constants['weights'].name.replace('.', '/').replace('layers/', 'layers.')
                        if 'down_proj' in w_name:
                            prefix = w_name[:w_name.index('mlp')]
                            eltwise_opt_t_name = f"/{prefix}mlp/Mul"
                            set_quantization(n.outputs[0], qconfig, eltwise_opt_t_name)

            if n.type == OpType.Negative:
                op_name = n.name[:n.name.index('negative')].replace('.', '/').replace('layers/', 'layers.')
                opt_t_name = f"/{op_name}Neg_0" if 'negative_0' in n.name else f"/{op_name}Neg_1_0"
                set_quantization(n.outputs[0], qconfig, opt_t_name)

                child_n = n.children[0]
                if child_n.type == OpType.Concat:
                    opt_cn_name = f"{op_name}Concat" if 'Neg_0' in opt_t_name else f"{op_name}Concat_1"
                    opt_ct_name = f"/{opt_cn_name}_0"
                    set_quantization(child_n.outputs[0], qconfig, opt_ct_name)

                    child_child_n = child_n.children[0]
                    if child_child_n.type == OpType.Eltwise and child_child_n.params['method'] == 'MUL':
                        opt_ccn_name = f"{op_name}Mul_1" if 'Neg_0' in opt_t_name else f"{op_name}Mul_3"
                        opt_cct_name = f"/{opt_ccn_name}_0"
                        set_quantization(child_child_n.outputs[0], qconfig, opt_cct_name)

                        ccchild_n = child_child_n.children[0]
                        if ccchild_n.type == OpType.Eltwise and ccchild_n.params['method'] == "ADD":
                            opt_cccn_name = f"{op_name}Add" if 'Neg_0' in opt_t_name else f"{op_name}Add_1"
                            opt_ccct_name = f"/{opt_cccn_name}_0"
                            set_quantization(ccchild_n.outputs[0], qconfig, opt_ccct_name)

                            ccchild_inp_node = ccchild_n.inputs[0].op if ccchild_n.inputs[1].op == child_child_n else ccchild_n.inputs[1].op
                            if ccchild_inp_node.type == OpType.Eltwise and ccchild_inp_node.params['method'] == 'MUL':
                                opt_ccn_other_name = f"{op_name}Mul" if 'Neg_0' in opt_t_name else f"{op_name}Mul_2"
                                opt_cct_other_name = f"/{opt_ccn_other_name}_0"
                                set_quantization(ccchild_inp_node.outputs[0], qconfig, opt_cct_other_name)

            if n.type == OpType.Eltwise and n.params['method'] == 'ADD':
                n.attrs['eltwise_quantization'] = True

    def _update_qinfo_for_float(self, aipu_g, qconfig, trigger_float_op=None):
        if trigger_float_op is None:
            trigger_float_op = qconfig.get_info('trigger_float_op')
            trigger_float_op = [] if 'op' not in trigger_float_op else trigger_float_op['op']
        if len(trigger_float_op):
            def _get_matched(keys, rule):
                matcheds = []
                for k in keys:
                    m = re.match(rule, k)
                    if m is not None:
                        matcheds.append(m.group())
                return matcheds

            for n in aipu_g.nodes:
                # if n.type == OpType.Constant:
                #     if n.constants['weights'].name == 'scale_for_normalization':
                #         n.attrs['trigger_float_op'] = NodeParamValue('float16')
                #         n.attrs['unquantifiable'] = True
                #         n.outputs[0].quantization = Quantization(1.0, 0)
                #         n.outputs[0].quantization.aipu_dtype = Dtype.FP16

                if n.type == OpType.DeQuantize:
                    if n.children[0].type.name.lower() not in trigger_float_op:
                        ERROR(
                            f"Dequantize op's child node {n.children[0]} should be float op and in trigger_float_op={trigger_float_op}")
                    # aipu_dtype = n.children[0].outputs[0].quantization.aipu_dtype
                    # if aipu_dtype == Dtype.VOID:
                    #     ERROR(f" aipu dtype is void")
                    # n.attrs['trigger_float_op'] = NodeParamValue(str(aipu_dtype))
                    n.attrs['trigger_float_op'] = NodeParamValue('float16')
                    n.attrs['unquantifiable'] = True
                if n.type.name.lower() in trigger_float_op:
                    if n.type.name.lower() == 'softmax' and n.children[0].type == OpType.Quantize:
                        ld = 0 if n.name.lower() == 'softmax' else int(n.name.split('_')[-1]) + 1
                        # opt_t_name = f'/model/layers.{ld}/self_attn/Softmax'
                        rule = r'^/layers.(.*)/self_attn/Softmax(.*)Quantize_tensor(.*)'
                        matcheds = _get_matched(qconfig.keys(), rule)
                        # print(f"ld = {ld}, and matched = {matcheds}")
                        if not len(matcheds):
                            ERROR(f"softmax matched size == 0")
                        for m in matcheds:
                            if f'/layers.{ld}' in m:
                                set_quantization(n.children[0].outputs[0], qconfig, m)
                                break

                    if n.type.name.lower() == 'rmsnorm' and n.children[0].type == OpType.Quantize:
                        w_name = n.constants['weights'].name
                        w_name = w_name.replace('.', '/').replace('weight', 'Mul_1_0').replace('layers/', 'layers.')
                        rule = f"/{w_name[:-1]}(.)*Quantize_tensor(.*)"
                        matcheds = _get_matched(qconfig.keys(), rule)
                        if len(matcheds):
                            set_quantization(n.children[0].outputs[0], qconfig, matcheds[0])
                        else:
                            ERROR(f"not found the matched key in {n} and rule = {rule}")

                    if n.type.name.lower() == 'activation' and n.params['method'].lower() == 'silu' and n.children[0].type == OpType.Quantize:
                        fc = n.parents[0].parents[0]
                        if fc.type.name.lower() == 'fullyconnected':
                            w_name = fc.constants['weights'].name
                            w_name = w_name.replace('.', '/').replace('weight',
                                                                      'MatMul_post_reshape_0').replace('layers/', 'layers.')
                            rule = f"/{w_name}(.)*Quantize_tensor(.*)"
                            matcheds = _get_matched(qconfig.keys(), rule)
                            if len(matcheds):
                                set_quantization(n.children[0].outputs[0], qconfig, matcheds[0])
                            else:
                                ERROR(f"not found the matched key in {n} and rule = {rule}")

                    if n.type.name.lower() == 'eltwise' and n.children[0].type == OpType.Quantize:
                        cchild_n = n.children[0].children[0]
                        # if cchild_n.type == OpType.BatchNorm:
                        #     name = cchild_n.constants['weights'].name[:cchild_n.constants['weights'].name.index(
                        #         'scale')-1].replace('.', '/').replace('layers/', 'layers.')
                        #     i = int(n.name.split('_')[-1]) if '_' in n.name else -1
                        #     midfix = 'Add_1_0' if i % 5 == 0 else 'Add_0'
                        #     rule = f"/{name}/" + midfix + ('(.*)Quantize_tensor(.*)')
                        #     matcheds = _get_matched(qconfig.keys(), rule)
                        #     if len(matcheds):
                        #         set_quantization(n.children[0].outputs[0], qconfig, matcheds[0])
                        #         # n.children[0].outputs[0].quantization.scales = [n.children[0].outputs[0].quantization.scales[0] / 0.2973]
                        #     else:
                        #         ERROR(f"not found the matched key in {n} and rule = {rule}")
                        if cchild_n.type == OpType.MatMul:
                            name = cchild_n.name[:cchild_n.name.index(
                                'matmul')-1].replace('.', '/').replace('layers/', 'layers.')
                            rule = f"/{name}/" + ('Add_1_0' if 'Add_0' in n.name else 'Add_0') + \
                                ('(.*)Quantize_tensor(.*)')
                            matcheds = _get_matched(qconfig.keys(), rule)
                            if len(matcheds):
                                set_quantization(n.children[0].outputs[0], qconfig, matcheds[0])
                                # n.children[0].outputs[0].quantization.scales = [n.children[0].outputs[0].quantization.scales[0] / 0.2973]
                            else:
                                ERROR(f"not found the matched key in {n} and rule = {rule}")

                        if cchild_n.type == OpType.Concat:
                            idx = int(cchild_n.name.split('_')[-1])
                            if idx % 4 == 1:
                                rule = f"/layers.{idx//4}/self_attn/Add_1_(.*)Quantize_tensor(.*)"
                                matcheds = _get_matched(qconfig.keys(), rule)
                                if len(matcheds):
                                    set_quantization(n.children[0].outputs[0], qconfig, matcheds[0])
                                    # n.children[0].outputs[0].quantization.scales = [n.children[0].outputs[0].quantization.scales[0] / 0.2973]
                                else:
                                    ERROR(f"not found the matched key in {n} and rule = {rule}")

                        if cchild_n.type == OpType.FullyConnected:
                            w_name = cchild_n.constants['weights'].name
                            w_name = w_name.replace('.', '/').replace('weight', 'MatMul').replace('layers/', 'layers.')
                            rule = f"/{w_name[:w_name.index('mlp')+3]}/Mul(.*)Quantize_tensor(.*)"
                            matcheds = _get_matched(qconfig.keys(), rule)
                            if len(matcheds):
                                set_quantization(n.children[0].outputs[0], qconfig, matcheds[0])
                            else:
                                ERROR(f"not found the matched key in {n} and rule = {rule}")

    def _insert_cast(self, aipu_g, qconfig):
        def _get_matched(keys, rule):
            matcheds = []
            for k in keys:
                m = re.match(rule, k)
                if m is not None:
                    matcheds.append(m.group())
            return matcheds
        saved_n = {}
        for n in aipu_g.nodes:
            cur_inp_dtypes = [t.quantization.aipu_dtype for t in n.inputs]
            cur_out_dtypes = [t.quantization.aipu_dtype for t in n.outputs]
            if n.type not in saved_n:
                lib_specs = get_dtypes(n.type)
                spec_dict = {}
                for i, spec in enumerate(lib_specs):
                    lib_inp_dtypes = spec.in_dtypes
                    lib_out_dtypes = spec.out_dtypes
                    spec_dict.update({i: [lib_inp_dtypes, lib_out_dtypes]})
                saved_n.update({n.type: spec_dict})
            else:
                spec_dict = saved_n[n.type]

            lib_impled = False
            for i, (lib_inp_dtypes, lib_out_dtypes) in spec_dict.items():
                if len(lib_inp_dtypes) != len(cur_inp_dtypes) or len(lib_out_dtypes) != len(cur_out_dtypes):
                    continue
                if cur_inp_dtypes == lib_inp_dtypes and cur_out_dtypes == lib_out_dtypes:
                    lib_impled = True
                    break
            if not lib_impled:
                # need to insert cast to adapt the lib's impl
                if n.type == OpType.BatchNorm and n.parents[0].type == OpType.Cast:
                    pn = n.parents[0]
                    cast0 = ops.cast(pn.outputs[0], pn.outputs[0].quantization.aipu_dtype,
                                     ignore_scale_zp=True, clip_mode='TRUNCATION')
                    cast_op = cast0.op
                    idx = n.remove_input(pn.outputs[0])
                    n.add_input(cast0, idx)
                    aipu_g.add_node(cast_op)
                    rule = f"{r'^'}" + f"/model/Cast_6_0_Cast_tensor(.*)"
                    ms = _get_matched(qconfig.keys(), rule)
                    if len(ms):
                        set_quantization(cast0, qconfig, ms[0])
                    else:
                        ERROR(f"not found the matched key in {n} and rule = {rule}")
