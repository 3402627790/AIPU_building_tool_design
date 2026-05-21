import re
from ..engine import register_gt, register_gen_input, register_gen_weight
from .gt_util.quant_utils import clamp, _tpc_lut, linear_requantize, linear_dequantize
from .gt_util.aiff import _aiff_lut, _aiff_lookup_float_index_lut
from AIPUBuilder.logger import DEBUG
from AIPUBuilder.core import OpType
import torch
import numpy as np
from AIPUBuilder.core import Target
from typing import Union


@register_gt(OpType.Sine)
@register_gt(OpType.Cosine)
@register_gt(OpType.Sinh)
@register_gt(OpType.Cosh)
@register_gt(OpType.Asin)
@register_gt(OpType.Acos)
@register_gt(OpType.Asinh)
@register_gt(OpType.Acosh)
@register_gt(OpType.Atan)
@register_gt(OpType.Atanh)
@register_gt(OpType.Tan)
def Cosine(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float:
        is_perf_mode = self.params.get('is_perf_mode', False)
        lut = self.constants["lut"].betensor if 'lut' in self.constants else None
        is_aiff = True
        if 'is_aiff' in kwargs.keys() and not kwargs['is_aiff']:
            is_aiff = False
        if is_aiff and is_perf_mode and not lut is None:
            if self.type.name == 'Cosine':
                # cos(x) = sin(pi/2 + x)
                inp_tensor = self.inputs[0].betensor.float() + torch.pi / 2
                inter = (inp_tensor * (1 / (2 * torch.pi))).int()
                Fractional = inp_tensor - inter * 2 * torch.pi
                Fractional = Fractional.to(torch.float16)
                outp = _aiff_lookup_float_index_lut(Fractional, lut,
                                                    self.params['index_scale_value'],
                                                    self.params['index_offset_value'],
                                                    mirror_mode=self.params.get("lut_mode", "") == "MIRROR",
                                                    value_offset_for_mirror_mode=self.params['value_offset_value'])
                self.outputs[0].betensor = outp.half()

            elif self.type.name == 'Sine':
                inp_tensor = self.inputs[0].betensor.float()
                inter = (inp_tensor * (1 / (2 * torch.pi))).int()
                Fractional = inp_tensor - inter * 2 * torch.pi
                Fractional = Fractional.to(torch.float16)
                outp = _aiff_lookup_float_index_lut(Fractional, lut,
                                                    self.params['index_scale_value'],
                                                    self.params['index_offset_value'],
                                                    mirror_mode=True,
                                                    value_offset_for_mirror_mode=self.params['value_offset_value'])
                self.outputs[0].betensor = outp.half()

            if self.type.name in ["Asin", "Acos", "Sinh", "Asinh", "Acosh", "Atan", "Tanh", "Atanh"]:
                inp = self.inputs[0].betensor
                index_offset_value = self.params.get("index_offset_value", 0)
                index_scale_value = self.params.get("index_scale_value", 1)
                mirror_mode = self.params.get("lut_mode", "") == "MIRROR"
                value_offset_value = self.params.get("value_offset_value", 0)
                outp = _aiff_lookup_float_index_lut(
                    inp,
                    lut,
                    index_scale_value,
                    index_offset_value,
                    mirror_mode,
                    value_offset_value,
                )
                self.outputs[0].betensor = outp.to(torch.float16)

            elif self.type.name == "Cosh":
                inp_tensor = self.inputs[0].betensor.float()
                lut = self.constants["lut"].betensor
                mirror_mode = self.params.get("lut_mode", "") == "MIRROR"
                sinh_out = _aiff_lookup_float_index_lut(inp_tensor * 0.5, lut,
                                                        self.params['index_scale_value'],
                                                        self.params['index_offset_value'],
                                                        mirror_mode=mirror_mode,
                                                        value_offset_for_mirror_mode=self.params['value_offset_value'],
                                                        is_trunc=True)
                outp = 2 * sinh_out * sinh_out + 1
                self.outputs[0].betensor = outp.half()
            return self.outputs[0].betensor
        else:
            return lut_float(self)

    is_aiff = True
    if 'is_aiff' in kwargs.keys() and not kwargs['is_aiff']:
        is_aiff = False
    if not is_aiff or np.prod((self.constants['lut'].shape)) != 256:
        print('run tpc lut...')
        inp = self.inputs[0]
        q_bits_activation = inp.dtype.bits
        if q_bits_activation > 8 and self.type in [OpType.Sine, OpType.Cosine]:
            x = inp.betensor.long() + inp.quantization.offset
            lut = self.constants["lut"].betensor
            out = trigonometric_forward(self, x, lut)
        else:
            lut_in_bits = 8 if '8' in str(self.inputs[0].dtype) else 16
            in_is_signed = True if str(self.inputs[0].dtype).startswith('int') else False
            out = _tpc_lut(self.inputs[0].betensor, self.constants['lut'].betensor, lut_in_bits, in_is_signed)
            out = clamp(out, self.outputs[0].dtype)
        self.outputs[0].irtensor = out
        return self.outputs[0].betensor

    print('run aiff lut...')
    out = _aiff_lut(self.inputs[0].irtensor.cpu().numpy().astype(self.inputs[0].dtype.np),
                    self.constants['lut'].betensor.cpu().numpy().astype(self.constants['lut'].dtype.np),
                    target=Target().name).astype(self.outputs[0].dtype.np)

    self.outputs[0].irtensor = out
    return self.outputs[0].betensor


def trigonometric_forward(node, inp, lut):
    scale0 = node.params["scale_value"]
    shift0 = node.params["shift_value"]
    lut_size = lut.numel()

    mask = torch.tensor(lut_size - 1, dtype=torch.int32, device=inp.device)
    index = linear_requantize(inp, scale0, shift0, 'int32', 0).to(torch.int32)
    index = index & mask

    index_large_mask = index >= lut_size
    index_less_mask = index < -lut_size
    index[index_large_mask] = index[index_large_mask] - lut_size
    index[index_less_mask] = index[index_less_mask] + lut_size

    out = lut[index.long()]
    return out


def lut_float(self):
    if self.type.name == 'Sine':
        self.outputs[0].betensor = torch.sin(self.inputs[0].betensor)
    elif self.type.name == 'Cosine':
        self.outputs[0].betensor = torch.cos(self.inputs[0].betensor)
    elif self.type.name == 'Asin':
        outp = torch.asin(self.inputs[0].betensor)
        nan_mask = torch.isnan(outp)
        if True in nan_mask:
            outp[nan_mask] = 65504
            DEBUG("there are nan value in the output, please check the input range.Currently, nan is overridden with a maximum value 65504, but this may make the result abnormal!")
        self.outputs[0].betensor = outp.half()
    elif self.type.name == 'Acos':
        self.outputs[0].betensor = torch.acos(self.inputs[0].betensor)
    elif self.type.name == 'Cosh':
        self.outputs[0].betensor = np.cosh(self.inputs[0].betensor.float())  # torch -> np CP-24165
    elif self.type.name == 'Sinh':
        self.outputs[0].betensor = np.sinh(self.inputs[0].betensor.float())
    elif self.type.name == 'Asinh':
        self.outputs[0].betensor = torch.asinh(self.inputs[0].betensor)
    elif self.type.name == 'Acosh':
        self.outputs[0].betensor = torch.acosh(self.inputs[0].betensor)
    elif self.type.name == 'Atan':
        self.outputs[0].betensor = torch.atan(self.inputs[0].betensor)
    elif self.type.name == 'Tan':
        self.outputs[0].betensor = torch.tan(self.inputs[0].betensor)
    elif self.type.name == 'Tanh':
        self.outputs[0].betensor = torch.tanh(self.inputs[0].betensor)
    elif self.type.name == 'Atanh':
        outp = torch.atanh(self.inputs[0].betensor)
        self.outputs[0].betensor = outp
    else:
        raise NotImplementedError
    return self.outputs[0].betensor


@register_gen_input(OpType.Cosine)
def gen_input_data(self):
    in_shape = list(self.inputs[0].betensor.shape)
    if self.inputs[0].dtype.is_float:
        is_perf_mode = self.params.get('is_perf_mode', False)
        lut = self.constants["lut"].betensor if 'lut' in self.constants else None
        lut_mode = self.params.get('lut_mode', None)
        if is_perf_mode and not lut is None:
            index_scale = self.params['index_scale_value']
            index_offset = self.params['index_offset_value']
            clip_min = (-511.0+index_offset)/index_scale - np.pi/2
            clip_max = (511.0+index_offset)/index_scale - np.pi/2
            input_0 = np.random.uniform(clip_min, clip_max, size=in_shape)
            input0 = np.float16(input_0)
            index_0 = np.where(input0 < clip_min)
            index_1 = np.where(input0 > clip_max)
            input0[index_0] = np.float16((clip_min+clip_max)/2)
            input0[index_1] = np.float16((clip_min+clip_max)/2)
        else:
            input0 = np.float16(np.random.uniform(-1000, 1000, size=in_shape))
        input0 = input0.astype(self.inputs[0].dtype.np)
    else:
        input0 = np.random.randint(np.iinfo(self.inputs[0].dtype.np).min, np.iinfo(self.inputs[0].dtype.np).max,
                                   size=self.inputs[0].shape, dtype=self.inputs[0].dtype.np)
    self.inputs[0].betensor = np.int64(input0) if not self.inputs[0].dtype.is_float else input0
    input_data = [input0]
    return input_data


@register_gen_input(OpType.Sine)
@register_gen_input(OpType.Sinh)
@register_gen_input(OpType.Cosh)
@register_gen_input(OpType.Asinh)
@register_gen_input(OpType.Acosh)
@register_gen_input(OpType.Tan)
@register_gen_input(OpType.Atanh)
@register_gen_input(OpType.Atan)
@register_gen_input(OpType.Asin)
@register_gen_input(OpType.Acos)
def gen_input_data(self):
    if self.inputs[0].dtype.is_float:
        in_shape = list(self.inputs[0].betensor.shape)
        is_perf_mode = self.params.get('is_perf_mode', False)
        lut_mode = self.params.get('lut_mode', None)
        lut = self.constants["lut"].betensor if 'lut' in self.constants else None
        if is_perf_mode and not lut is None:
            index_scale = self.params['index_scale_value']
            index_offset = self.params['index_offset_value']
            if lut_mode:
                clip_min = (-511.0+index_offset)/index_scale
                clip_max = (511.0+index_offset)/index_scale
                input_0 = np.random.uniform(clip_min, clip_max, size=in_shape)
                input0 = np.float16(input_0)
                index_0 = np.where(input0 < clip_min)
                index_1 = np.where(input0 > clip_max)
                input0[index_0] = np.float16((clip_min+clip_max)/2)
                input0[index_1] = np.float16((clip_min+clip_max)/2)

            else:
                clip_min = (0.0+index_offset)/index_scale
                clip_max = (511.0+index_offset)/index_scale
                input_0 = np.random.uniform(clip_min, clip_max, size=in_shape)
                input0 = np.float16(input_0)
                index_0 = np.where(input0 < clip_min)
                index_1 = np.where(input0 > clip_max)
                input0[index_0] = np.float16((clip_min+clip_max)/2)
                input0[index_1] = np.float16((clip_min+clip_max)/2)

        else:
            if self.type in [OpType.Asin, OpType.Acos]:
                input0 = np.float16(np.random.uniform(-1, 1, size=in_shape))
            else:
                input0 = np.float16(np.random.uniform(-1000, 1000, size=in_shape))
    else:
        input0 = np.random.randint(np.iinfo(self.inputs[0].dtype.np).min, np.iinfo(self.inputs[0].dtype.np).max,
                                   size=self.inputs[0].shape, dtype=self.inputs[0].dtype.np)
    self.inputs[0].betensor = np.int64(input0) if not self.inputs[0].dtype.is_float else input0
    input_data = [input0]

    return input_data


@register_gen_weight(OpType.Cosine)
@register_gen_weight(OpType.Sine)
@register_gen_weight(OpType.Sinh)
@register_gen_weight(OpType.Cosh)
@register_gen_weight(OpType.Asinh)
@register_gen_weight(OpType.Acosh)
@register_gen_weight(OpType.Tan)
@register_gen_weight(OpType.Atanh)
@register_gen_weight(OpType.Atan)
@register_gen_weight(OpType.Asin)
@register_gen_weight(OpType.Acos)
def gen_random_weight(self):
    if 'lut' in self.constants.keys():
        if self.inputs[0].dtype.is_float:
            if self.params.get('is_perf_mode', False):
                index_scale = self.params['index_scale_value']
                index_offset = self.params['index_offset_value']
                lut_items_in_bits = 9
                lut = linear_dequantize(torch.arange(0, 2**lut_items_in_bits), index_scale, index_offset)
            else:
                rng = np.random.default_rng()
                param_type = str(self.constants['lut'].dtype)
                param_shape = list(self.constants['lut'].shape)
                lut = rng.uniform(low=-100, high=100,
                                  size=param_shape).astype(param_type)
        else:
            param_type = str(self.constants['lut'].dtype)
            param_shape = list(self.constants['lut'].shape)
            lut = np.random.randint(np.iinfo(param_type).min, np.iinfo(param_type).max,
                                    dtype=param_type, size=param_shape)
        self.constants['lut'].betensor = lut
