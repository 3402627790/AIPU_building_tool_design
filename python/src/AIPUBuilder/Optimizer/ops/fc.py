# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.ops.conv import *
from AIPUBuilder.Optimizer.ops.activation import apply_with_activation
from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *

import torch.nn as nn


@op_register(OpType.FullyConnected)
def fc(self, *args):
    inp, bias, weights = None, None, None
    if self.constants['weights'].dtype in [Dtype.FP32, Dtype.FP16, Dtype.BFP16]:
        inp = self.inputs[0].betensor.float()
        bias = self.constants["biases"].betensor.float()
        weights = self.constants["weights"].betensor.float()
    else:
        inp = self.inputs[0].betensor.double()
        bias = self.constants["biases"].betensor.clone().double()
        weights = self.constants["weights"].betensor.clone().double()
    aasrb = self.get_param('remain_shift',
                           optional=True, default_value=None)

    if self.quantized:
        # input's zerop has been absorbed to bias.
        # inp += self.inputs[0].zerop
        bias += self.constants['biases'].broadcast_zerop
        ##if input quantization is true, output quantization is false, such as fp8(input)->fp16(output), need set output.qmin/qmax
        if self.outputs[0].qmin is None or self.outputs[0].qmax is None:
            self.outputs[0].qmin, self.outputs[0].qmax = dtype2range(self.outputs[0].dtype)
        if all(sname in self.constants.keys() for sname in ['scale0', 'scale1']):
            weights,scale1 = linear_dequantize_for_gptq_w4afp8(self.constants)
            x = (inp@weights.double().T)*scale1.T+bias
            self.outputs[0].betensor = apply_with_activation(self, x, *args)
            return self.outputs[0].betensor
        weights += self.constants["weights"].broadcast_zerop
        if aasrb is not None and (dtype2bits(self.constants["weights"].dtype) > 8 or dtype2bits(self.inputs[0].dtype) > 8):
            x = inp @ weights.T
            self.outputs[0].betensor = apply_with_activation(self, x,
                                                             *args, aasrb=(aasrb, bias))
            return self.outputs[0].betensor

        if all(sname in self.constants.keys() for sname in ['scale1', 'scale2', 'scale3']):
            # act per channel quantization
            a_scale = self.constants['scale2'].betensor
            b_scale = self.constants['scale1'].betensor
            c_scale = self.constants['scale3'].betensor
            x = inp @ weights.T
            y = x * a_scale * b_scale + bias
            self.quantized = False
            y = apply_with_activation(self, y, *args)
            self.quantized = True
            self.outputs[0].betensor = linear_quantize_clip(y.float(
            ), c_scale, 0, self.outputs[0].qmin, self.outputs[0].qmax, round_func=get_round_func_according_to_dtype('ROUND_TO_EVEN', self.outputs[0].dtype))
            #torch.clamp((y * c_scale).round(), self.outputs[0].qmin, self.outputs[0].qmax)
            return self.outputs[0].betensor

    x = nn.functional.linear(inp, weights, bias,)
    self.outputs[0].betensor = apply_with_activation(self, x, *args)
    return self.outputs[0].betensor


def per_channel_activation_quantize(self, *args):
    quant_type = self.attrs.get('quant_type')
    q_mode_activation = self.attrs["q_mode_activation"]
    fpx_quantize = True if QuantType.is_fpx_quant_activation(quant_type) else False
    q_mode_weight = self.attrs["q_mode_weight"]
    q_bits_weight = self.attrs["q_bits_weight"]
    q_bits_activation = self.attrs["q_bits_activation"]
    w = self.constants["weights"]
    out = self.outputs[0]
    out_signed = with_activation_out_is_signed(self) or self.force_dtype_int
    out.qbits = q_bits_activation
    if fpx_quantize:
        q_type_activation = QuantType._to_Dtype(QuantType.activation_type(quant_type))
        out.dtype = q_type_activation
        out.scale, out.zerop, out.qmin, out.qmax = get_fpx_quant_params_from_tensor(
            out, q_mode_activation, q_type_activation)
        q_type_weight = QuantType._to_Dtype(QuantType.weight_type(quant_type))
        w.dtype = q_type_weight
        w.qbits = dtype2bits(w.dtype)
        w.scale, w.zerop, w.qmin, w.qmax = get_fpx_quant_params_from_tensor(w, q_mode_weight, q_type_weight)
        w.betensor = linear_quantize_clip(w.betensor.float(), w.broadcast_scale, w.broadcast_zerop,
                                          w.qmin, w.qmax,
                                          round_func=get_round_func_according_to_dtype('ROUND_TO_EVEN', w.dtype))
    else:
        out.scale, out.zerop, out.qmin, out.qmax, out.dtype = get_linear_quant_params_from_tensor(
            out, q_mode_activation, q_bits_activation, is_signed=out_signed)
        w.scale, w.zerop, w.qmin, w.qmax, w.dtype = get_linear_quant_params_from_tensor(w, q_mode_weight, q_bits_weight,
                                                                                        is_signed=True)
        w.betensor = linear_quantize_clip(w.betensor, w.broadcast_scale, w.broadcast_zerop, w.qmin, w.qmax)
        w.qbits = q_bits_weight
    out.qinvariant = False
    w.qinvariant = False

    b = self.constants['biases']
    b.scale = 1.0
    b.zerop = 0
    b.dtype = Dtype.FP32
    b.qbits = 32
    b.qinvariant = False

    inp = self.inputs[0]

    self.constants['scale2'] = PyTensor('scale2', 1.0 / inp.broadcast_scale, dtype=Dtype.FP32)
    self.constants['scale1'] = PyTensor('scale1', 1.0 / w.broadcast_scale.T, dtype=Dtype.FP32)
    self.constants['scale3'] = PyTensor('scale3', out.broadcast_scale, dtype=Dtype.FP32)


def w4fp8_quantize(self, *args):
    q_mode_weight = self.attrs["q_mode_weight"]
    quant_type = self.attrs.get('quant_type')

    act_aipu_dtype = str2dtype(QuantType.activation_type(quant_type))
    weight_aipu_dtype = str2dtype(QuantType.weight_type(quant_type))

    w = self.constants["weights"]
    origin_w_dtype = [Dtype.BFP16, Dtype.FP32, Dtype.FP16]
    origin_act_dtype = origin_w_dtype
    support_w_dtype = origin_w_dtype + [weight_aipu_dtype]

    if w.ir_dtype not in support_w_dtype:
        OPT_ERROR(
            f"currently the op{self} weight_type{w.ir_dtype} of IR can't support, weight_type only support such as {support_w_dtype}")
        return
    if self.outputs[0].ir_dtype not in origin_act_dtype:
        OPT_ERROR(
            f"currently the op{self} activation_type{self.outputs[0].ir_dtype} of IR can't support, activation_type only support such as {origin_act_dtype}")
        return

    if weight_aipu_dtype == Dtype.ALIGNED_INT4:
        activation_quantize(self, *args)
        if w.ir_dtype in origin_w_dtype:
            if QuantMode.is_per_tensor(q_mode_weight):
                OPT_ERROR(
                    f"currently the op{self} quant_type[{quant_type}] can't support per_tensor weight quant_mode, now quantize_method_for_weight is {q_mode_weight}.")
                return
            w.qbits = dtype2bits(weight_aipu_dtype)
            if QuantMode.is_per_block(q_mode_weight):
                weight_block_size = self.attrs["weight_block_size"]
                w.block_size = weight_block_size
                from AIPUBuilder.Optimizer.features import statistic_and_calibration
                wb = PyTensor('weight_blocks')
                remain =  w.betensor.shape[-1] % w.block_size
                if remain:
                    padding_num = w.block_size - remain
                    padding_value = w.betensor[:, -1:]
                    padding_tw = padding_value.repeat(1, padding_num)
                    padded_w = torch.cat((w.betensor,padding_tw),dim=1)
                    wb.betensor = padded_w.reshape(-1, w.block_size)
                else:
                    wb.betensor = w.betensor.reshape(-1, w.block_size)
                wb.key_axis = 0
                statistic_and_calibration(wb, self.attrs, is_constant_tensor=True)
                w.scale, w.zerop, w.qmin, w.qmax, w.dtype = get_linear_quant_params_from_tensor(wb, QuantMode.to_per_channel(q_mode_weight), w.qbits, is_signed=True)
                self.params['weight_block_size'] = w.block_size
            else:
                w.scale, w.zerop, w.qmin, w.qmax, w.dtype = get_linear_quant_params_from_tensor(w , q_mode_weight, w.qbits, is_signed=True)
            w.betensor = linear_quantize_clip(w.betensor, w.broadcast_scale, w.broadcast_zerop,
                                              w.qmin, w.qmax,
                                              round_func=get_round_func_according_to_dtype('ROUND_TO_EVEN', w.dtype))

        else:
            w.qbits = dtype2bits(w.ir_dtype)
            w.qmin, w.qmax = dtype2range(w.ir_dtype)
        w.qinvariant = False
        local_rescale = self.outputs[0].scale / self.inputs[0].scale
        generate_scale_shift(self, local_rescale, *args)
        bias_quantize(self, bias_scale=self.inputs[0].scale, *args)
        apply_epilogue_quantize(self)
        params_dict = {'beta0': 1.625, 'beta1': 0.98, 'alpha0': 0.75, 'alpha1': 0.5}

    elif weight_aipu_dtype == Dtype.FP4_E2M1FN:
        activation_quantize(self, *args)
        if w.ir_dtype in origin_w_dtype:
            if QuantMode.is_per_tensor(q_mode_weight):
                OPT_ERROR(
                    f"currently the op{self} quant_type[{quant_type}] can't support per_tensor weight quant_mode, now quantize_method_for_weight is {q_mode_weight}.")
                return
            fpx_weight_quantize(self, *args)
        else:
            w.qbits = dtype2bits(w.ir_dtype)
            w.qmin, w.qmax = dtype2range(w.ir_dtype)
        w.qinvariant = False
        local_rescale = self.outputs[0].scale / (self.inputs[0].scale)
        generate_scale_shift(self, local_rescale, *args)
        bias_quantize(self, bias_scale=self.inputs[0].scale, *args)
        apply_epilogue_quantize(self)
        params_dict = {'beta0' : 1.525, 'beta1' : 1.125, 'alpha0' : 0.625, 'alpha1' : 0.875}
    else:
        OPT_ERROR(
            f"currently the op{self} quant_type{quant_type} can't support.")
        return
    if 'scale0' not in self.constants or 'scale1' not in self.constants:
        scale0, scale1 = convert_fp16_scales_to_fp8_scales(self.constants["weights"],
                                                           self.constants["weights"].scale,
                                                           self.constants["weights"].zerop, act_aipu_dtype, params_dict)
        self.constants['scale0'] = PyTensor('scale0', scale0.half(), act_aipu_dtype)
        self.constants['zp0'] = PyTensor('zp0', self.constants["weights"].zerop.reshape([scale0.shape[0], scale0.shape[1]]), weight_aipu_dtype)
        self.constants['scale1'] = PyTensor('scale1', scale1, Dtype.FP16)
    #Currently, weight_block_size is not a required parameter, so it will not be added to IR for now.
    # if 'weight_block_size' not in self.params:
    #     self.params['weight_block_size'] = torch.numel(self.constants["weights"].betensor) // torch.numel(self.constants["weights"].scale)



@quant_register(OpType.FullyConnected)
def fc_quantize(self, *args):
    quant_type = self.attrs.get('quant_type')
    #some global calibration such as smooth_quant may set inputs[0].key_axis
    is_per_tensor_activation = all(ka is None for ka in [self.inputs[0].key_axis, self.outputs[0].key_axis]) or QuantMode.is_per_tensor(self.attrs["q_mode_activation"])

    if is_per_tensor_activation:
        if quant_type != 'disable' and str2dtype(QuantType.activation_type(quant_type)) in [Dtype.FP8_E4M3FN, Dtype.FP8_E5M2, Dtype.FP16, Dtype.BFP16] and QuantType.weight_bits(quant_type) == 4:
            w4fp8_quantize(self, *args)
        else: #w8a8/w16a16/wfp8afp8
            conv2d_quantize(self, *args)
            # OPT_ERROR(
            #     f"currently the op{self} can't suppport per-channel activation quantization when quant_type = {quant_type}")
    else:
        per_channel_activation_quantize(self, *args)

