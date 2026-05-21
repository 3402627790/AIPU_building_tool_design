import re
import os
import numpy as np
import torch
from AIPUBuilder.core import *  # noqa
from AIPUBuilder import ops
from AIPUBuilder.logger import ERROR
from ...utils import gen_quantization, set_quantization
from ...register import register_model_quantize
from ...qinfo import ModelQuantize, QConf


@register_model_quantize(['llama_w16a16mix', 'llama2_w16a16mix', 'llama2_chn_w16a16mix'])
class Llama2W16A16MixModelQuantize(ModelQuantize):
    def __init__(self, model, model_type, quantization_type='w16a16mix', hf_config=None):
        super().__init__(model, model_type, quantization_type, hf_config)

        self.model_path = os.path.join(os.path.dirname(__file__))
        json_file = os.path.join(self.model_path, f"llama_w16a16mix_qinfo.json")
        self.qconfig = QConf(json_file)

    def _set_maltiplier_bits(self):
        for n in self.aipu_g.nodes:
            output_dtypes = [o.quantization.aipu_dtype.bits for o in n.outputs]
            if 16 in output_dtypes:
                n.attrs['multiplier_bits'] = 13

    def update_qinfo(self):
        self._update_qinfo_for_quantized(self.aipu_g, self.qconfig)
        self._insert_cast()
        self.eliminate_scale_normalization()
        self.trigger_float_op = self.get_float_ops()
        if len(self.trigger_float_op):
            self.insert_quantize_and_dequantize(self.trigger_float_op)
            self._update_qinfo_for_float(self.aipu_g, self.qconfig, self.trigger_float_op)
        self.transparent_quantization()
        # self._set_maltiplier_bits()

    def _update_qinfo_for_quantized(self, aipu_g, qconfig):
        for n in aipu_g.nodes:
            # update input tensors
            if n.type == OpType.Input:
                for k in ['embeded', 'position_id', 'attention_mask']:
                    if k in n.outputs[0].name.lower():
                        set_quantization(n.outputs[0], qconfig, f'{k}_0')

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
                    # opt_w_name = f"sin_cached_0weights"
                    opt_w_name = f"/layers.0Constant_25_0"
                    set_quantization(n.outputs[0], qconfig, opt_w_name)
                    set_quantization(n.constants['weights'], qconfig, opt_w_name)

                    # update sin/cos gather op
                    set_quantization(n.children[0].outputs[0], qconfig, opt_w_name)

                if matched1 is not None:
                    # name = matched1.group().replace('.', '/')[:-10].replace('layers/', 'layers.')
                    opt_w_name = f"/layers.0Constant_24_0"
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

                    silu_child_n = silu_n.children[0]
                    if silu_child_n.type == OpType.Eltwise:
                        opt_e_t_name = f"{opt_t_name[:opt_t_name.index('mlp')]}mlp/up_proj/MatMul_post_reshape_0"
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
                            eltwise_opt_t_name = f"/{prefix}Add_1_0"
                            set_quantization(n.outputs[0], qconfig, eltwise_opt_t_name)
                        elif 'normlm/norm' in w_name:
                            eltwise_opt_t_name = '/layers.31/Add_1_0'
                            set_quantization(n.outputs[0], qconfig, eltwise_opt_t_name)

                    if OpType.FullyConnected in child_types and len(child_types) == 1:
                        child_n = n.children[0]
                        w_name = child_n.constants['weights'].name.replace('.', '/').replace('layers/', 'layers.')
                        if 'down_proj' in w_name:
                            prefix = w_name[:w_name.index('mlp')]
                            eltwise_opt_t_name = f"/{prefix}mlp/up_proj/MatMul_post_reshape_0"
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

    def _update_qinfo_for_float(self, aipu_g, qconfig, trigger_float_op):
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

    def _insert_cast(self):
        def _get_matched(keys, rule):
            matcheds = []
            for k in keys:
                m = re.match(rule, k)
                if m is not None:
                    matcheds.append(m.group())
            return matcheds

        for n in self.aipu_g.nodes:
            if n.type == OpType.FullyConnected:
                w_name = n.constants['weights'].name.replace('.', '/').replace('layers/', 'layers.')
                o_dtype = n.outputs[0].quantization.aipu_dtype
                if 'mlp/down_proj' in w_name and o_dtype == Dtype.INT8:
                    cn = n.children[0]
                    cast0 = ops.cast(n.outputs[0], Dtype.INT16,
                                     ignore_scale_zp=False, clip_mode='TRUNCATION')
                    cast_op = cast0.op
                    idx = cn.remove_input(n.outputs[0])
                    cn.add_input(cast0, idx)
                    self.aipu_g.add_node(cast_op)
                    prefix = w_name[:w_name.index('mlp')]
                    rule = f"{r'^'}" + f"/{prefix}mlp/down_proj/MatMul_post_reshape_0"
                    ms = _get_matched(self.qconfig.keys(), rule)
                    if len(ms):
                        set_quantization(cast0, self.qconfig, ms[0])
                    else:
                        ERROR(f"not found the matched key in {n} and rule = {rule}")
