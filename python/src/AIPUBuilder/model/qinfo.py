# cython: language_level=3
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import json
import torch
import os
import numpy as np
from collections import namedtuple
from enum import Enum, auto
from AIPUBuilder.core import NodeParamValue, Tensor, Dtype, OpType, Target, GraphMatcher, MatchMode, SubGraphNode
from AIPUBuilder.core import Quantization, Quantizer, Scheme, QuantizedRange, Granularity, cast_dtypes_for_lib, use_cuda
from AIPUBuilder import ops
from AIPUBuilder.logger import WARN, INFO, ERROR, DEBUG
from AIPUBuilder.model.utils import DEFAULT_TARGET, AIPU_BLOCK_SIZE, AIPU_DEFAULT_BLOCK_SIZE, get_quant_config, string_to_base_type


__all__ = ['QConf', 'QInfoParam', 'ModelQuantize', 'DEFAULT_TARGET']


class QInfoParam(object):
    SCALE = "scale"
    ZEROP = "zerop"
    DTYPE = "dtype"
    BITS = "bits"
    GIDX = "gidx"


class QConf(object):
    def __init__(self, json_path):
        if not os.path.exists(json_path):
            WARN(f"{json_path} is not existed, and cannot load.")
        else:
            with open(json_path) as f:
                self.config = json.load(f)

    def get_info(self, tensor_name):
        if tensor_name not in self.config:
            WARN(f"{tensor_name} not in qconfiguration")
        return self.config.get(tensor_name, None)

    def keys(self):
        return self.config.keys()


class QuantType(object):
    _QT = namedtuple('QuantType', ['name', 'is_weight_only_quantization', 'weight_type', 'activation_type'])

    # ptq int quantization
    W8A8 = _QT('w8a8', False, 'int8', 'int8')  # weight 8bits quantization, activation 8bits quantization
    W16A16 = _QT('w16a16', False, 'int16', 'int16')  # weight 16bits quantization, activation 16bits quantization
    W4A16 = _QT('w4a16', False, 'int4', 'int16')  # weight 4bits quantization, activation 16bits quantization
    W16A16MIX = _QT('w16a16mix', False, 'int16', 'int16')

    # WOQ(weight_only_quantization)
    W4 = _QT('w4', True, 'aligned_int4', 'float16')  # WOQ, w4 is w4afp16, weight int4, activation float16
    W8 = _QT('w8', True, 'int8', 'float16')  # wOQ, w8 is w8afp16, weight int8, activation float16
    W4AFP16 = _QT('w4afp16', True, 'aligned_int4', 'float16')  # WOQ, weight int4, activation float16
    W8AFP16 = _QT('w8afp16', True, 'int8', 'float16')  # WOQ, weight int8, activation float16

    W4ABF16 = _QT('w4abf16', True, 'aligned_int4', 'bfloat16')  # WOQ, weight int4, activation bfloat16
    W8ABF16 = _QT('w8abf16', True, 'int8', 'bfloat16')  # WOQ, weight int8, activation bfloat16

    WFP4 = _QT('wfp4', True, 'float4_e2m1fn', 'float16')  # WOQ, weight fp4e2m1fn, activation float16
    WFP4AFP16 = _QT('wfp4afp16', True, 'float4_e2m1fn', 'float16')  # WOQ, weight fp4e2m1fn, activation float16
    WFP4ABF16 = _QT('wfp4abf16', True, 'float4_e2m1fn', 'bfloat16')  # WOQ, weight fp4e2m1fn, activation bfloat16

    # only for test
    WFP4AFP32 = _QT('wfp4afp32', True, 'float4_e2m1fn', 'float32')  # WOQ, weight fp4e2m1fn, activation float32

    WFP8 = _QT('wfp8', True, 'float8_e4m3fn', 'float16')  # WOQ, weight fp8e4m3fn, activation float16
    # ptq float quantization
    W4AFP8E5M2 = _QT('w4afp8e5m2', False, 'aligned_int4', 'float8_e5m2')    # weight int4, activation float8_e5m2
    W4AFP8E4M3 = _QT('w4afp8e4m3', False, 'aligned_int4', 'float8_e4m3fn')    # weight int4, activation float8_e4m3fn
    # weight float8_e4m3fn, activation float8_e4m3fn
    WFP8AFP8 = _QT('wfp8afp8', False, 'float8_e4m3fn', 'float8_e4m3fn')
    # weight float8_e5m2, activation float8_e5m2
    WFP8E5M2AFP8E5M2 = _QT('wfp8e5m2afp8e5m2', False, 'float8_e5m2', 'float8_e5m2')
    # weight float8_e4m3fn, activation float8_e4m3fn
    WFP8E4M3AFP8E4M3 = _QT('wfp8e4m3afp8e4m3', False, 'float8_e4m3fn', 'float8_e4m3fn')
    # weight float4_e2m1fn, activation float8_e4m3fn
    WFP4AFP8E4M3 = _QT('wfp4afp8e4m3', False, 'float4_e2m1fn', 'float8_e4m3fn')

    # keep unquantized
    FP16 = _QT('fp16', False, 'float16', 'float16')  # weight float16, activation float16
    BF16 = _QT('bf16', False, 'bfloat16', 'bfloat16')  # weight bfloat16, activation bfloat16
    BFP16 = _QT('bfp16', False, 'bfloat16', 'bfloat16')  # weight bfloat16, activation bfloat16

    @staticmethod
    def supported_quant_mode():
        quant_modes = {}
        for _, v in QuantType.__dict__.items():
            if isinstance(v, QuantType._QT):
                quant_modes.update({v.name: v})
        return quant_modes

    # @staticmethod
    # def supported_quant_mode_name_pair():
    #     quant_modes = QuantType.supported_quant_mode()
    #     qmode_str_pair = dict(zip(quant_modes.values(), quant_modes.keys()))
    #     return qmode_str_pair

    @staticmethod
    def _to_quant_mode(name_or_quantmode):
        nq = name_or_quantmode
        supported_quant_modes = QuantType.supported_quant_mode()
        if isinstance(nq, str):
            nq = supported_quant_modes.get(nq, None)
        if nq is None or (isinstance(nq, QuantType._QT) and nq not in supported_quant_modes.values()):
            ERROR(
                f"the quant_type = {str(name_or_quantmode)} is not supported, only supports: {supported_quant_modes}.")
        return nq

    @staticmethod
    def is_weight_only_quantization(name_or_quantmode):
        nq = QuantType._to_quant_mode(name_or_quantmode)
        return nq.is_weight_only_quantization if nq is not None else False

    @staticmethod
    def is_keep_unquantized(name_or_quantmode):
        nq = QuantType._to_quant_mode(name_or_quantmode)
        ret = False
        if nq in [QuantType.FP16, QuantType.BFP16, QuantType.BF16]:
            ret = True
        return ret

    @staticmethod
    def qmode_to_weight_and_activation_pair(name_or_quantmode):
        qm = QuantType._to_quant_mode(name_or_quantmode)

        wht_act_pair = {}
        if qm is not None:
            wht_act_pair.update({'weight_type': qm.weight_type})
            wht_act_pair.update({'activation_type': qm.activation_type})
        return wht_act_pair

    @staticmethod
    def is_supported_quant_mode(name_or_quantmode):
        nq = QuantType._to_quant_mode(name_or_quantmode)
        return True if nq is not None else False

    @staticmethod
    def weight_dtype(name_or_quantmode):
        weight_dtype = None
        if QuantType.is_supported_quant_mode(name_or_quantmode):
            qm = QuantType._to_quant_mode(name_or_quantmode)
            weight_dtype = qm.weight_type
        else:
            supported_quant_modes = QuantType.supported_quant_mode()
            WARN(
                f"the quant_type = {str(name_or_quantmode)} is not supported, only supports: {supported_quant_modes}.")
        return weight_dtype

    @staticmethod
    def activation_dtype(name_or_quantmode):
        activation_type = None
        if QuantType.is_supported_quant_mode(name_or_quantmode):
            qm = QuantType._to_quant_mode(name_or_quantmode)
            activation_type = qm.activation_type
        else:
            supported_quant_modes = QuantType.supported_quant_mode()
            WARN(
                f"the quant_type = {str(name_or_quantmode)} is not supported, only supports: {supported_quant_modes}.")
        return activation_type


class AIPUQuantizeStrategy(object):

    def __init__(self, config, aipug):
        self.config = config
        self.g = aipug

        self.quantization_type = self.config.quant_type
        if self.config.get('activation_type', None) is None or self.config.get('weight_type', None) is None:
            weight_activation_pair = QuantType.qmode_to_weight_and_activation_pair(self.quantization_type)
            self.config.update(weight_activation_pair)
        self.weight_type = self.config.weight_type
        self.activation_type = self.config.get('activation_type', 'float16')

    def _get_quant_config(self):
        ret = get_quant_config(self.config)
        return ret

    @staticmethod
    def _statistic(t, statistic_method='extrema', running_momentum=1.0):

        if t.betensor.numel() <= 0:
            WARN(f"tensor={t} doesnot statistic, because it doesnot have data.")
            return
        tensor = t.betensor
        qinfo = t.quantization
        granu = qinfo.granularity
        s_min = qinfo.mins
        s_max = qinfo.maxs
        cur_min = None
        cur_max = None
        if 'per_tensor' == str(granu):
            cur_min = tensor.min()
            cur_max = tensor.max()
        elif 'per_channel' == str(granu):
            # now defaultly key_axis = 0
            if tensor.dim() > 0:
                cur_min = tensor.reshape([tensor.shape[0], -1]).min(dim=-1).values
                cur_max = tensor.reshape([tensor.shape[0], -1]).max(dim=-1).values
            else:
                cur_min = tensor.min()
                cur_max = tensor.max()
        else:
            ERROR(f"granularity(={granu}) not in [per_tensor, per_channel].")

        if len(s_min) == 0 or len(s_max) == 0:
            qinfo.mins = cur_min.cpu().numpy()
            qinfo.maxs = cur_max.cpu().numpy()
            return

        default_device = tensor.device
        ud_mins, ud_maxs = torch.zeros(1).to(default_device), torch.zeros(1).to(default_device)
        if statistic_method == 'extrema':
            ud_mins = torch.min(cur_min.to(default_device), s_min.to(default_device))
            ud_maxs = torch.max(cur_max.to(default_device), s_max.to(default_device))
        elif statistic_method == 'mean':
            ud_mins = running_momentum * s_min + (1 - running_momentum) * cur_min
            ud_maxs = running_momentum * s_max + (1 - running_momentum) * cur_max
        else:
            ud_mins = cur_min
            ud_maxs = cur_max
            ERROR(f"statistic_method={statistic_method} not in [extrema, mean].")

        qinfo.mins = torch.min(ud_mins.to(default_device), torch.tensor(0.0).to(default_device)).cpu().numpy()
        qinfo.maxs = torch.max(ud_maxs.to(default_device), torch.tensor(0.0).to(default_device)).cpu().numpy()

    def is_quantization_dtype(self, dtype):
        if dtype in [Dtype.VOID, Dtype.BOOL]:
            return False
        return True if dtype.size < 2 or (dtype.size == 2 and dtype.is_int) else False

    def __call__(self):
        msg = f"\t quantize with quant_type={self.quantization_type}, which activation_dtype= {self.config.activation_type}, and weight_dtype={self.config.weight_type}"
        quant_method = ''
        if self.config.get('hf_quantization_config', None) is not None:
            quant_method = self.config.hf_quantization_config.get('quant_method', '')
            msg += f", quantization_config with quant_method = {quant_method}."
        INFO(msg)

        # first qat quantization, second aipu quantization
        if quant_method != '':
            if quant_method.lower() == 'gptq':
                self._gptq_quantization()
            elif quant_method.lower() == 'awq':
                self._awq_quantization()
            elif quant_method.lower() == 'mxfp4':
                self._mxfp4_quantization()
            elif quant_method.lower() == 'gguf_quant':
                self._gguf_quantization()
            elif quant_method.lower() == 'compressed-tensors':
                if QuantType.is_keep_unquantized(self.quantization_type):
                    self._fp_quantization()
                elif self.quantization_type in ['wfp8afp8', 'wfp8', 'wfp4']:
                    self._compressed_tensors_quantization()
                else:
                    self._aipu_weight_only_quantization()
            else:
                ERROR(f"[LLM]un-supported quant_method={quant_method}.")
        elif QuantType.is_weight_only_quantization(self.quantization_type):
            self._aipu_weight_only_quantization()
        elif QuantType.is_keep_unquantized(self.quantization_type):
            self._fp_quantization()
        else:
            ERROR(
                f"[LLM] when quantize {self.config.model_name}, does not support quantization_type={self.quantization_type}")

    def _configure_activation_qinfo(self, n, activation_dtype=None):
        target = DEFAULT_TARGET if self.config.get('target', '') == '' else self.config.target
        try:
            from AIPUBuilder.Optimizer.utils import Target as _PyTarget
            aiff_lut_items_in_bits = _PyTarget.aiff_lut_items_in_bits(target.split('_')[0])
        except:
            aiff_lut_items_in_bits = 9

        # hidden_act = self.config.get('hidden_act', 'silu')

        quant_config = self._get_quant_config()
        is_perf_mode = quant_config.get('is_perf_mode', True)
        is_perf_mode_for_softmax = quant_config.get('is_perf_mode_for_softmax', is_perf_mode)
        act_dtype = Dtype(self.activation_type.lower())

        n.attrs['unquantifiable'] = True
        n.params['unquantifiable'] = True

        Target.set_target(target)
        if n.type in [OpType.Softmax]:
            if is_perf_mode_for_softmax:
                n.attrs['approx_params'] = NodeParamValue(int(1))
                n.attrs['lut_items_in_bits'] = aiff_lut_items_in_bits
            else:
                n.attrs['approx_params'] = NodeParamValue(0)

        if n.type in [OpType.Activation]:
            if is_perf_mode:
                if n.params['method'].lower() in ['silu', 'sigmoid', 'swish', 'gelu', 'mish', 'hardswish']:
                    n.attrs['approx_params'] = NodeParamValue([1])
                    n.attrs['lut_items_in_bits'] = aiff_lut_items_in_bits
            else:
                n.attrs['approx_params'] = NodeParamValue(0)

        if 'trigger_float_op' not in n.attrs:
            if activation_dtype is None:
                activation_dtype = self.activation_type
            n.attrs['trigger_float_op'] = NodeParamValue(activation_dtype)

        if n.type == OpType.Input:
            if n.outputs[0].dtype.is_float:
                n.outputs[0].dtype = act_dtype

        for ot in n.outputs:
            if ot.io_tensor and ot.dtype.is_float:
                ot.dtype = act_dtype

    def _weight_only_quantization_one_node(self,
                                           n,
                                           bits,
                                           quant_config,
                                           quantization_type,
                                           activation_type,
                                           is_lm_head_fp16,
                                           qscheme=None,
                                           qgranularity=None,
                                           weight_only_quantization_op='fullyconnected',
                                           ):
        n.attrs['trigger_float_op'] = NodeParamValue(activation_type)
        if str(n.type)[7:].lower() in weight_only_quantization_op:
            w = n.constants['weights']
            weight_type = QuantType.supported_quant_mode()[quantization_type].weight_type.upper()
            cur_bits = bits
            skip_quantization = False
            if 'lm' in w.name:
                if not is_lm_head_fp16:
                    cur_bits = quant_config.get('lm_head_weight_bits', bits)
                    lmfc_weight_type = quant_config.get('lmhead_fc_weight_type', weight_type)
                    weight_type = QuantType.supported_quant_mode()[lmfc_weight_type].weight_type.upper()
                else:
                    # lm head fc uses fp16 * fp16
                    skip_quantization = True
            if not skip_quantization:
                aipu_weight_type = Dtype(weight_type.lower())
                w.quantization.bits = cur_bits
                if aipu_weight_type in [Dtype.FP4E2M1FN]:
                    w.quantization.scheme = Scheme.Symmetric if qscheme is None else qscheme
                else:
                    w.quantization.scheme = (Scheme.Asymmetric if cur_bits <
                                             8 else Scheme.Symmetric) if qscheme is None else qscheme
                w.quantization.granularity = Granularity.Per_Channel if qgranularity is None else qgranularity
                w.quantization.quantized_range = QuantizedRange.Full_Range
                # INFO(f"now {n}'s weight's quantization = {w.quantization}")
                # defaultly is per-channel quantization(key_axis = 0), if setted, will use the setted weight_block_size
                if 'weight_block_size' in quant_config:
                    group_size = quant_config.get('weight_block_size', w.shape[-1])
                    scale_size = w.shape[0] * w.shape[-1] // group_size
                    if w.shape[0] * w.shape[-1] / scale_size != group_size:
                        ERROR(
                            f"In per_vector quantization, ic={w.shape[-1]} should be divided by weight_group_size={group_size}")
                    w.quantization.group = scale_size
                else:
                    if aipu_weight_type in [Dtype.FP4E2M1FN]:
                        WARN(
                            f"when weight uses fp4e2m1fn quantization, we suggest the per-group quantization, and set 'weight_block_size' in cfg.")
                    w.quantization.group = quant_config.get('weight_block_size', w.shape[0])
                w.quantization.aipu_dtype = aipu_weight_type
                trigger_mode = f"{activation_type.upper()}_ACT_{weight_type}_WHT"
                n.attrs['trigger_float_op'] = NodeParamValue(trigger_mode)
        self._configure_activation_qinfo(n)

    def _aipu_weight_only_quantization(self):
        self.bits = int(''.join(list(filter(str.isdigit, self.weight_type))[0]))
        if self.bits not in [4, 8]:
            ERROR(f"when weight only quantization, weight qbits should be in [4, 8], but now is {self.bits}")

        quant_config = self._get_quant_config()
        self.lm_head_fp16 = quant_config.get('lm_head_fp16', True)
        self.weight_only_quantization_op = (self.config.get('aipu_quant_config', {}).get('weight_only_quantization_op',
                                                                                         'FullyConnected')).lower()
        self.qscheme = quant_config.get('qscheme', None)
        self.qgranularity = quant_config.get('qgranularity', None)

        def set_qinfo(n):
            if SubGraphNode.is_subgraph_node(n):
                sbg = SubGraphNode.from_node(n).subgraph
                for sbgn in sbg.nodes:
                    set_qinfo(sbgn)
            else:
                self._weight_only_quantization_one_node(n,
                                                        self.bits,
                                                        quant_config,
                                                        self.quantization_type,
                                                        self.activation_type,
                                                        self.lm_head_fp16,
                                                        self.qscheme,
                                                        self.qgranularity,
                                                        self.weight_only_quantization_op)

        for node in self.g.nodes:
            set_qinfo(node)

    def _gptq_quantization(self):
        bits = self.config.hf_quantization_config['bits']
        desc_act = self.config.hf_quantization_config['desc_act']
        sym = self.config.hf_quantization_config['sym']

        quant_config = self._get_quant_config()
        gptq_gidx_to_fc = self.config.get('gptq_gidx_to_fc', True)
        self.lm_head_fp16 = quant_config.get('lm_head_fp16', True)

        def _set_node_qinfo(n):
            is_gptq = True
            # whether lmhead fc will do aipu quantization in quant_method = gptq
            if n.type == OpType.FullyConnected and 'lm' in n.constants['weights'].name and not self.lm_head_fp16 and not len(n.constants['weights'].quantization.scales):
                INFO(f"lm_head fc({n}) will do aipu quantization in quant_type = GPTQ with quant_config={quant_config}")
                self.qscheme = quant_config.get('qscheme', None)
                self.qgranularity = quant_config.get('qgranularity', None)
                self._weight_only_quantization_one_node(n, bits, quant_config, self.quantization_type,
                                                        self.activation_type, False, self.qscheme, self.qgranularity)
                is_gptq = False

            if is_gptq:
                for ck, cv in n.constants.items():
                    # has already saved the GIDX in weight_tensor._attrs when using update_tensor_data
                    if QInfoParam.GIDX in cv._attrs and desc_act:
                        g_idx = cv._attrs[QInfoParam.GIDX]
                        _, indices = torch.sort(g_idx)
                        if not gptq_gidx_to_fc:
                            d = cv.betensor[:, indices]
                            cv.set_data(d.contiguous().cpu().numpy())
                            with self.g:
                                in0 = n.inputs[0]
                                cons_t = ops.constant(indices.cpu().numpy().astype(np.uint32),
                                                      quantization=Quantization(1.0, 0))
                                gather_t = ops.gather(in0, cons_t, axis=-1)
                                n.clear_inputs()
                                n.add_input(gather_t)
                        elif gptq_gidx_to_fc:
                            d = cv.betensor[:, indices]
                            cv.set_data(d.contiguous().cpu().numpy())
                            oc, ic = cv.betensor.shape
                            idx_array = np.zeros([ic, ic], dtype=np.int8)
                            idx_array[np.arange(ic), indices.cpu().numpy()] = 1
                            # for i, idx in enumerate(indices.cpu().numpy().tolist()):
                            #     arr = np.zeros(ic, dtype=np.int8)
                            #     arr[idx] = 1
                            #     idx_array[i] = arr
                            with self.g:
                                w = Tensor(idx_array)
                                w.quantization.scales = np.array([1.0]).astype(np.float32)
                                w.quantization.offsets = np.array([0]).astype(np.int32)
                                w.quantization.aipu_dtype = Dtype.ALIGNED_INT4
                                w.dtype = Dtype.ALIGNED_INT4
                                b = Tensor(np.zeros([ic], dtype=np.float32))
                                fc_t = ops.fully_connected(n.inputs[0], w, b)
                                n.clear_inputs()
                                n.add_input(fc_t)
                        else:
                            gidx_t = Tensor(g_idx.flatten().cpu().numpy().astype(np.int32))
                            gidx_t.quantization = Quantization(1.0, 0)
                            gidx_t.quantization.qinvariant = True
                            n.constants['gidx'] = gidx_t
                        pass
                    if len(cv.quantization.scales):
                        n.constants[ck].quantization.aipu_dtype = Dtype.ALIGNED_INT4 if bits <= 4 else Dtype.INT8
                        n.constants[ck].dtype = Dtype.ALIGNED_INT4 if bits <= 4 else Dtype.INT8

            self._configure_activation_qinfo(n)

        def set_qinfo(n):
            if SubGraphNode.is_subgraph_node(n):
                sbg = n.subgraph
                for sbgn in sbg.nodes:
                    set_qinfo(sbgn)
            else:
                _set_node_qinfo(n)

        nodes = self.g.nodes
        for node in nodes[:]:
            set_qinfo(node)

        if desc_act:
            self.g.topological_sort()

    def _awq_quantization(self):
        bits = self.config.hf_quantization_config['bits']
        quant_config = self._get_quant_config()
        self.lm_head_fp16 = quant_config.get('lm_head_fp16', True)

        def _set_node_qinfo(n):
            is_awq = True
            # whether lmhead fc will do aipu quantization in quant_method = awq
            if n.type == OpType.FullyConnected and 'lm' in n.constants['weights'].name and not self.lm_head_fp16 and not len(n.constants['weights'].quantization.scales):
                INFO(f"lm_head fc({n}) will do aipu quantization in quant_type = AWQ with quant_config={quant_config}")
                self.qscheme = quant_config.get('qscheme', None)
                self.qgranularity = quant_config.get('qgranularity', None)
                self._weight_only_quantization_one_node(n, bits, quant_config, self.quantization_type,
                                                        self.activation_type, False, self.qscheme, self.qgranularity)
                is_awq = False

            if is_awq:
                for ck, cv in n.constants.items():
                    if len(cv.quantization.scales):
                        n.constants[ck].quantization.aipu_dtype = Dtype.ALIGNED_INT4 if bits == 4 else Dtype.INT8
                        n.constants[ck].dtype = Dtype.ALIGNED_INT4 if bits == 4 else Dtype.INT8
            self._configure_activation_qinfo(n)

        def set_qinfo(n):
            if SubGraphNode.is_subgraph_node(n):
                sbg = n.subgraph
                for sbgn in sbg.nodes:
                    set_qinfo(sbgn)
            else:
                _set_node_qinfo(n)

        nodes = self.g.nodes
        for node in nodes[:]:
            set_qinfo(node)

    def _compressed_tensors_quantization(self):
        weight_type = QuantType.supported_quant_mode()[self.quantization_type].weight_type.upper()
        aipu_weight_type = Dtype(weight_type.lower())

        def _set_node_qinfo(n):
            flag = "float16"
            for ck, cv in n.constants.items():
                if ck == "weights" and 'lm' in n.constants[ck].name:
                    pass

                if len(cv.quantization.scales):
                    if not self.is_quantization_dtype(aipu_weight_type):
                        ERROR(
                            f"when node(={n}) sets Compressed quant method, the configured weight type(={aipu_weight_type}) is not quantization dtype, but its has quantized scales.")
                    n.constants[ck].quantization.aipu_dtype = aipu_weight_type
                    weight_data = n.constants[ck].betensor.cpu().numpy()
                    n.constants[ck] = Tensor.new(weight_data.astype(aipu_weight_type.np))
                    n.constants[ck].sync_betensor()
                    # n.constants[ck].quantization.scales = np.clip(cv.quantization.scales, 0., 448.0)
                    n.constants[ck].quantization.qscales = cv.quantization.qscales
                    n.constants[ck].quantization.qoffsets = cv.quantization.qoffsets
                    n.constants[ck].dtype = aipu_weight_type
                    flag = None
            self._configure_activation_qinfo(n, flag)

        def set_qinfo(n):
            if SubGraphNode.is_subgraph_node(n):
                sbg = n.subgraph
                for sbgn in sbg.nodes:
                    set_qinfo(sbgn)
            else:
                _set_node_qinfo(n)

        nodes = self.g.nodes
        for node in nodes[:]:
            set_qinfo(node)

    def _mxfp4_quantization(self):
        # bits = self.config.hf_quantization_config['bits']
        # group_size = self.config.hf_quantization_config['group_size']
        weight_type = QuantType.supported_quant_mode()[self.quantization_type].weight_type.upper()
        aipu_weight_type = Dtype(weight_type.lower())

        def _set_node_qinfo(n):
            if n.type == OpType.FullyConnected:
                for ck, cv in n.constants.items():
                    if len(cv.quantization.scales):
                        n.constants[ck].quantization.aipu_dtype = aipu_weight_type
                        weight_data = n.constants[ck].betensor.cpu().numpy()
                        n.constants[ck] = Tensor.new(weight_data.astype(aipu_weight_type.np))
                        n.constants[ck].sync_betensor()
                        # n.constants[ck].quantization.scales = np.clip(cv.quantization.scales, 0., 448.0)
                        n.constants[ck].quantization.scales = cv.quantization.scales
                        n.constants[ck].quantization.offsets = cv.quantization.offsets
                        n.constants[ck].dtype = aipu_weight_type
            self._configure_activation_qinfo(n)

        def set_qinfo(n):
            if SubGraphNode.is_subgraph_node(n):
                sbg = SubGraphNode.from_node(n).subgraph
                for sbgn in sbg.nodes:
                    set_qinfo(sbgn)
            else:
                _set_node_qinfo(n)

        nodes = self.g.nodes
        for node in nodes[:]:
            set_qinfo(node)

    def _gguf_quantization(self):
        for n in self.g.nodes:
            for ck, cv in n.constants.items():
                if len(cv.quantization.scales) and len(cv.quantization.offsets):
                    bits = cv.quantization.bits
                    if bits == -1 and not cv.dtype.is_float:
                        bits = torch.log2(cv.betensor.abs().max()).ceil().int().item()
                        if bits % 8 != 0:
                            bits = ((bits + 8) // 8) * 8
                    if bits == -1:
                        continue
                    sbits = f'int{bits}' if bits not in [4, 12] else f'aligned_int{bits}'
                    dtype = Dtype(sbits)
                    n.constants[ck].quantization.aipu_dtype = dtype
                    n.constants[ck].dtype = dtype
            self._configure_activation_qinfo(n)

    def _fp_quantization(self):
        activation_type = self.config.get('activation_type', 'float16')
        act_dtype = Dtype(activation_type.lower())

        target = self.config.get('target', '')
        target = DEFAULT_TARGET if target == '' else target
        quant_config = self._get_quant_config()
        is_perf_mode = quant_config.get('is_perf_mode', False)

        def _set_node_qinfo(n):
            n.attrs['trigger_float_op'] = NodeParamValue(activation_type)
            n.attrs['unquantifiable'] = True
            n.params['unquantifiable'] = True
            self._configure_activation_qinfo(n)

        def set_qinfo(n):
            if SubGraphNode.is_subgraph_node(n):
                sbg = SubGraphNode.from_node(n).subgraph
                for sbgn in sbg.nodes:
                    set_qinfo(sbgn)
            else:
                _set_node_qinfo(n)

        for node in self.g.nodes:
            set_qinfo(node)


class ModelQuantize(object):
    def __init__(self,
                 model,
                 model_type,
                 quantization_type='w8a8',
                 hf_config=None):
        self.aipu_g = model
        self.quantization_type = quantization_type
        # self.quantization_mode = QuantType.str_to_quantmode(quantization_type)
        self.model_type = model_type
        self.hf_config = hf_config
        self.kvcache_optimization = self.hf_config.kvcache_optimization if hasattr(
            self.hf_config, 'kvcache_optimization') else False
        self.target = self.hf_config.target if self.hf_config.get('target', '') != '' else DEFAULT_TARGET

    def insert_quantize_and_dequantize(self, trigger_float_ops=[]):
        # if trigger_float_ops is None:
        #     trigger_float_ops = self.get_float_ops()
        if len(trigger_float_ops) == 0:
            DEBUG(f"{self.model_type} doesnot need to insert quantize and dequantize.")
            return
        trigger_float_ops = [o.lower() for o in trigger_float_ops]
        add_nodes = []
        nodes = self.aipu_g.nodes
        for n in nodes[:]:
            if n.type.name.lower() in trigger_float_ops:
                output_dtypes = [ot.quantization.aipu_dtype.is_float for ot in n.outputs]
                if any(output_dtypes):
                    for _, inp in enumerate(n.inputs):
                        dq = ops.dequantize(inp)
                        idx = n.remove_input(inp)
                        n.add_input(dq, idx)
                        add_nodes.append(dq.op)

                    for _, out in enumerate(n.outputs):
                        consumers = out.consumers
                        q = ops.quantize(out)
                        for cn in consumers:
                            idx = cn.remove_input(out)
                            cn.add_input(q, idx)
                        add_nodes.append(q.op)
        for an in add_nodes:
            self.aipu_g.add_node(an)

        for n in self.aipu_g.nodes:
            if n is None:
                continue
            if n.type == OpType.Quantize:
                consumers = n.outputs[0].consumers
                need_merged = True
                for con in consumers:
                    if con.type != OpType.DeQuantize:
                        need_merged = False
                        break
                if need_merged:
                    inp = n.inputs[0]
                    for con in consumers:
                        dq_out = con.outputs[0]
                        for dq_con in dq_out.consumers:
                            idx = dq_con.remove_input(dq_out)
                            dq_con.add_input(inp, idx)
                        self.aipu_g.remove_node(con)
                    self.aipu_g.remove_node(n)
        if len(add_nodes):
            self.aipu_g.topological_sort()

        return self.aipu_g

    def get_float_ops(self):
        trigger_float_ops = set()
        for n in self.aipu_g.nodes:
            if n.type in [OpType.DeQuantize]:
                continue
            for ot in n.outputs:
                if ot.quantization.aipu_dtype.is_float:
                    trigger_float_ops.add(n.type.name.lower())
                    break
        return trigger_float_ops

    def eliminate_scale_normalization(self):
        gms = [GraphMatcher([("1", OpType.MatMul), ("2", OpType.Constant), ("3", OpType.Eltwise)], [("1", "3"), ("2", "3")]),
               GraphMatcher([("21", OpType.MatMul), ("22", OpType.Constant),
                            ("23", OpType.Mul)], [("21", "23"), ("22", "23")]),
               GraphMatcher([("11", OpType.MatMul), ("12", OpType.BatchNorm)], [("11", "12")])]
        for gm in gms:
            # gm.mode = MatchMode.StrictMode
            gm.mode = MatchMode.RelaxedMode
            matcheds = gm.match(self.aipu_g)
            for m in matcheds:
                if len(m) == 3 and ((m[2].type == OpType.Eltwise and m[2].params['method'].lower() == "mul") or
                                    m[2].type == OpType.Mul):
                    const_n = m[1] if m[1].type == OpType.Constant else m[0]
                    mt_n = m[0] if m[0].type == OpType.MatMul else m[1]

                elif len(m) == 2 and m[1].type == OpType.BatchNorm:
                    mt_n = m[0]
                    const_n = m[1]
                else:
                    ERROR(f"please recheck the patten")
                    continue

                const_w = const_n.constants['weights'].betensor
                if torch.equal(const_w, torch.full(const_w.shape, const_w.flatten()[0].item()).to(const_w.device)):
                    fused_multiplier = const_w.flatten()[0].item()
                    mt_n.params['fused_multiplier'] = fused_multiplier

                    const_n.clear_outputs()
                    for consumer in m[-1].outputs[0].consumers:
                        idx = consumer.remove_input(m[2].outputs[0])
                        consumer.add_input(mt_n.outputs[0], idx)

                    m[-1].clear_inputs()
                    m[-1].clear_outputs()
                    self.aipu_g.remove_node(const_n)
                    if len(m) == 3:
                        self.aipu_g.remove_node(m[2])

            # if len(matcheds):
            #     self.aipu_g.topological_sort()

    def update_qinfo(self):
        """default quantize method w4/w8/fp16/bfp16, and w4/w8 also include gptq and awq."""
        qs = AIPUQuantizeStrategy(self.hf_config, self.aipu_g)
        qs()

    def transparent_quantization(self):
        for n in self.aipu_g.nodes:
            if n.type in [OpType.Reshape, OpType.Transpose, OpType.Tile, OpType.Crop, OpType.Gather]:
                if not n.inputs[0].quantization.empty:
                    n.outputs[0].quantization = n.inputs[0].quantization
                    if not n.outputs[0].quantization.aipu_dtype.is_float:
                        n.outputs[0].dtype = n.outputs[0].quantization.aipu_dtype
            if n.type in [OpType.Logical]:
                if not n.inputs[0].quantization.empty:
                    n.outputs[0].quantization = n.inputs[0].quantization
                elif not n.inputs[1].quantization.empty:
                    n.outputs[0].quantization = n.inputs[1].quantization
                else:
                    n.outputs[0].quantization = Quantization(1.0, 0)
                assert not n.outputs[0].quantization.empty

    def quantize_transformation(self):
        # quantize_transform(self.aipu_g)
        quantizer = Quantizer(self.aipu_g)
        quantizer.quantize()

    def _optimize_kvcache(self):
        from AIPUBuilder import ops
        from AIPUBuilder.core import Tensor, TensorShape, TensorList, OpType
        nodes = self.aipu_g.nodes
        input_tensors = list(self.aipu_g.input_tensors)

        whole_k_cache, whole_v_cache = None, None
        graph_changed = False
        for n in nodes:
            def kvc_concat(cur_n):
                if cur_n.type == OpType.Concat and len(cur_n.inputs) == 2:
                    in_node_types = [inp.op.type for inp in cur_n.inputs]
                    # out_node_types = [consumer.type for consumer in cur_n.outputs[0].consumers]
                    if OpType.Input in in_node_types and cur_n.outputs[0] in self.aipu_g.output_tensors:
                        return True
                return False

            if kvc_concat(n):
                in_node_types = [inp.op.type for inp in n.inputs]
                block_num = (len(input_tensors) - 3) // 2
                one_k_cache = input_tensors[-2]
                max_len = self.hf_config.seq_len + self.hf_config.past_len
                hidden_size = one_k_cache.shape[1] * one_k_cache.shape[3]
                k_shape = [block_num, 1, max_len, hidden_size]
                if whole_k_cache is None:
                    with self.aipu_g:
                        whole_k_cache = ops.input(k_shape, one_k_cache.dtype, name='k_cache')
                if whole_v_cache is None:
                    with self.aipu_g:
                        whole_v_cache = ops.input(k_shape, one_k_cache.dtype, name='v_cache')

                pos_id_t = input_tensors[2]
                cur_kv = n.inputs[1 - in_node_types.index(OpType.Input)]
                cur_input = n.inputs[in_node_types.index(OpType.Input)]
                if '.key' in cur_input.name:
                    block_idx = input_tensors.index(cur_input) - 3
                    cache_outs = ops.cacheupdate(whole_k_cache, pos_id_t, cur_kv, block_idx)
                elif '.value' in cur_input.name:
                    block_idx = input_tensors.index(cur_input) - 3 - block_num
                    cache_outs = ops.cacheupdate(whole_v_cache, pos_id_t, cur_kv, block_idx)
                else:
                    block_idx = 0
                    ERROR(f"please check the kv cache input's name(={cur_input.name})")
                    cache_outs = None

                if cache_outs is not None:
                    inplace_cache = cache_outs[0]
                    cacheupdate_node = inplace_cache.op

                    consumers = n.outputs[0].consumers
                    for consumer in consumers:
                        idx = consumer.remove_input(n.outputs[0])
                        consumer.add_input(inplace_cache, idx)

                    n.clear_inputs()
                    n.clear_outputs()
                    self.aipu_g.remove_node(n)
                    self.aipu_g.add_node(cacheupdate_node)
                    graph_changed = True

        if graph_changed:
            if whole_v_cache is not None and whole_k_cache is not None:
                for t in self.aipu_g.input_tensors[3:]:
                    self.aipu_g.remove_node(t.op)
                self.aipu_g.add_node(whole_k_cache.op)
                self.aipu_g.add_node(whole_v_cache.op)
                self.aipu_g.input_tensors = TensorList(
                    [t for t in self.aipu_g.input_tensors[:3]] + [whole_k_cache, whole_v_cache])
                self.aipu_g.output_tensors = self.aipu_g.output_tensors[:1]
            self.aipu_g.topological_sort()

    def optimize_pass(self):
        if self.kvcache_optimization:
            self._optimize_kvcache()

        is_cast_dtypes_for_lib = False
        if hasattr(self.hf_config, 'aipu_quant_config'):
            qc = self.hf_config.aipu_quant_config
            is_cast_dtypes_for_lib = qc.get('cast_dtypes_for_lib', False)
        if is_cast_dtypes_for_lib:
            cast_dtypes_for_lib(self.aipu_g)
            # self.aipu_g.topological_sort()
            pass

    def __call__(self):
        Target.set_target(self.target)
        self.update_qinfo()
        self.quantize_transformation()
        self.optimize_pass()
        return self.aipu_g


class OptQuantize(ModelQuantize):
    def __init__(self,
                 model,
                 model_type,
                 quantization_type='w8a8',
                 hf_config=None):
        super().__init__(model, model_type, quantization_type, hf_config)
        self.model_type = model_type
        self.model_path = hf_config.model_path
        self.device = torch.device('cuda') if use_cuda() else torch.device('cpu')
        from AIPUBuilder.model.app.llmchat import AIPUTokenizer
        self.tokenizer = AIPUTokenizer(hf_config, model_type)
        self.optimizer_json = {}
        self.optimizer_argv = None
        self.opt_g = None
        self.quantgraph = None
        self.calibration_data = []

    def calibration_opt_graph(self):
        if self.opt_g is not None:
            from AIPUBuilder.model.calibration import Calibration
            calibration = Calibration()
            for node in self.opt_g.nodes:
                q_strategy_activation = self.optimizer_argv.calibration_strategy_for_activation.get(node)
                q_strategy_weight = self.optimizer_argv.calibration_strategy_for_weight(node)
                q_mode_activation = self.optimizer_argv.quantize_method_for_activation.get(node)
                q_mode_weight = self.optimizer_argv.quantize_method_for_weight.get(node)
                for o in node.outputs:
                    calibration(o, q_strategy_activation, q_mode_activation)
                for k, v in node.constants.items():
                    calibration(v, q_strategy_weight, q_mode_weight)

    def generate_dataloader(self):
        return None

    def global_calibrationn(self):
        if len(self.calibration_data):
            from AIPUBuilder.model.calibration import apply_global_calibration
            cloader = self.generate_dataloader()
            apply_global_calibration(self.opt_g, cloader, self.optimizer_argv.global_calibration)

    def calibration(self):
        self.calibration_opt_graph()
        self.global_calibrationn()

    def get_default_optimizer_config(self):
        from AIPUBuilder.Optimizer.config.cfg_fields import ALL_FIELDS
        defaults = {}
        for k, v in ALL_FIELDS.items():
            defaults.update({k: v.default()})
        return defaults

    def gen_optimizer_config(self, optimizer_argv):
        # from AIPUBuilder.Optimizer.utils.string_utils import string_to_base_type
        from AIPUBuilder.Optimizer.config.cfg_fields import ALL_FIELDS
        from AIPUBuilder.Optimizer.config.parser import cfg_parser, CfgParser
        from AIPUBuilder.Optimizer.framework.opt_register import QUANTIZE_METRIC_DICT, QUANTIZE_DATASET_DICT, \
            traverse_opt_plugins
        traverse_opt_plugins()
        argv = {}
        for opt_k, opt_v in optimizer_argv.items():
            opt_v = string_to_base_type(opt_v)
            if opt_k in ALL_FIELDS.keys():
                try:
                    field_obj = ALL_FIELDS[opt_k]
                    opt_flag, parsed_v = field_obj.parse(opt_v)
                    if not opt_flag:
                        ERROR(field_obj.error(opt_v))
                except Exception as e:
                    ERROR(f"when checking the '{opt_k}' field in cfg file meets error.")
                    raise e
                argv.update({opt_k: parsed_v})
        cfgparser = CfgParser(None, metric_dict=QUANTIZE_METRIC_DICT, dataset_dict=QUANTIZE_DATASET_DICT)
        # ret = cfgparser.checker(argv)
        cfgparser.argv = argv
        self.optimizer_argv = cfgparser.update()

        return self.optimizer_argv

    def get_zeros_calibration_data(self):
        from AIPUBuilder.Optimizer.utils import dtype2torch_type
        inputs_list = []
        input_tensors = self.opt_g.input_tensors
        for input_tensor in input_tensors:
            inp = torch.zeros(list(input_tensor.ir_shape), dtype=dtype2torch_type(
                input_tensor.dtype), device=self.device)
            inputs_list.append(inp)
        return inputs_list

    def get_inputs(self, prompt):
        prompt_after_template = self.tokenizer.apply_chat_template(prompt)
        tokens = self.tokenizer.encode(prompt_after_template)
        tokens_t = torch.tensor(tokens).to(self.device).flatten().reshape([1, -1])
        cache_shape = self.opt_g.input_tensors[-1].ir_shape
        cache_k, cache_v = [], []
        if self.hf_config.get('kvcache_optimization', False):
            cache_k = [torch.zeros(cache_shape)]
            cache_v = [torch.zeros(cache_shape)]
        else:
            cache_num = len(self.opt_g.input_tensors) - 3
            if cache_num % 2 != 0:
                ERROR(f"Cache length should be double of num_hidden_layer, but now is {cache_num}")

            if cache_num > 0:
                num_hidden_layers = cache_num // 2
                kcache_shape = list(self.opt_g.input_tensors[3].ir_shape)
                vcache_shape = list(self.opt_g.input_tensors[3 + num_hidden_layers].ir_shape)
                cache_k = [torch.zeros([1, kcache_shape[1], 0, kcache_shape[3]]).to(self.device)
                           for _ in range(num_hidden_layers)]
                cache_v = [torch.zeros([1, vcache_shape[1], 0, vcache_shape[3]]).to(self.device)
                           for _ in range(num_hidden_layers)]
        init_seq_len = len(tokens)
        iters = 0
        return (tokens_t, cache_k, cache_v, init_seq_len, iters)

    def get_opt_graph_from_aipu_graph(self):
        from AIPUBuilder.Optimizer.framework.pycore.pyir import convert_aipu_graph_to_opt_graph
        self.opt_g = convert_aipu_graph_to_opt_graph(self.aipu_g)

    def convert_opt_graph_to_aipu_graph(self):
        from AIPUBuilder.Optimizer.framework.pycore.pyir import convert_opt_graph_to_aipu_graph
        self.aipu_g = convert_opt_graph_to_aipu_graph(self.opt_g.quantgraph)

    def statistic_opt_graph(self):
        inputs = self.get_zeros_calibration_data()
        self.opt_g.statistic(inputs, self.optimizer_argv)

    def graph_pre_optimize(self):
        from AIPUBuilder.Optimizer.config.parser import get_info_from_graph
        from AIPUBuilder.Optimizer.passes.tune_op_extra_params_s1 import tune_op_passes
        from AIPUBuilder.Optimizer.passes.check_quantization_info_s1 import check_quantization_info
        from AIPUBuilder.Optimizer.passes.detect_inf_mask_nodes import detect_inf_mask_nodes
        config_info = get_info_from_graph(self.opt_g, self.optimizer_argv.data_batch_dim)
        self.opt_g.init_all_nodes_attrs(config_info, self.optimizer_argv)
        tune_op_passes(self.opt_g, self.optimizer_argv)
        check_quantization_info(self.opt_g, self.optimizer_argv)
        detect_inf_mask_nodes(self.opt_g, self.optimizer_argv)

    def insert_cast_op(self):
        from AIPUBuilder.Optimizer.passes.insert_op import InsertFakeOp, InsertQuantizeOp, InsertDeQuantizeOp, InsertCastOp
        from AIPUBuilder.Optimizer.passes.passes import insert_op_pass
        insert_obj = [InsertFakeOp, InsertQuantizeOp, InsertDeQuantizeOp, InsertCastOp]
        insert_op_pass(self.opt_g, self.optimizer_argv, insert_obj)

    def quantize(self):
        from AIPUBuilder.Optimizer.passes.passes import unify_scales_for_multi_inputs_op_pass
        self.opt_g.quantgraph = self.opt_g.clone()
        unify_scales_for_multi_inputs_op_pass(self.opt_g.quantgraph, self.optimizer_argv)
        self.opt_g.quantize()

    def graph_post_optimize(self):
        from AIPUBuilder.Optimizer.passes.merge_inserted_op import merge_inserted_op
        merge_inserted_op(self.opt_g.quantgraph, self.optimizer_argv)
        from AIPUBuilder.Optimizer.passes.eliminate_op import eliminate_ops
        eliminate_ops(self.opt_g.quantgraph, self.optimizer_argv)

    def clean(self):
        # clean memory
        if len(self.calibration_data):
            while self.calibration_data:
                tensor_list = self.calibration_data.pop()
                for tensor in tensor_list:
                    del tensor
            self.calibration_data = []
        if self.opt_g is not None:
            del self.opt_g
            self.opt_g = None

    def update_qinfo(self):
        pass

    def hf_config_syncto_optimizer_json(self, defaults):
        # if is_perf_mode=True in quantization_config, set silu/sigmoid/swish or other activation using lut mode
        ret = get_quant_config(self.hf_config)
        if ret.get('is_perf_mode', False):
            min_target_version = "X3"
            defaults['min_compatible_zhouyi_target'] = min_target_version.upper()
            if self.hf_config.target:
                target_version = self.hf_config.target.split('_')[0]
                from AIPUBuilder.Optimizer.utils import Target
                if Target.optimized_target_level(target_version) < Target.optimized_target_level(min_target_version):
                    WARN(f"target[{target_version}] does not support perf_mode, which may cause compilation errors.")
                else:
                    defaults['min_compatible_zhouyi_target'] = target_version.upper()
            defaults['enable_pass_tune_op_complicated_activations'] = "[0][1]"
            defaults['enable_pass_tune_op_softmax'] = "[0][1]"

        if ret.get('weight_block_size', None) is not None:
            defaults['weight_block_size'] = ret['weight_block_size']

    def opt_quantize(self):
        defaults = self.get_default_optimizer_config()
        defaults.update(**self.optimizer_json)
        self.hf_config_syncto_optimizer_json(defaults)
        self.gen_optimizer_config(defaults)
        self.graph_pre_optimize()
        self.statistic_opt_graph()
        self.calibration()
        self.insert_cast_op()
        self.quantize()
        self.graph_post_optimize()

    def __call__(self):
        # Target.set_target(self.target)
        self.update_qinfo()
        self.get_opt_graph_from_aipu_graph()
        self.opt_quantize()
        self.convert_opt_graph_to_aipu_graph()
        self.optimize_pass()
        self.clean()
        return self.aipu_g
