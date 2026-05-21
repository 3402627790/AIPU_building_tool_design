from ..engine import register_gt, register_gen_weight, register_gen_input, _GEN_WEIGHTS, _OP_METRICS, register_op_metric
from AIPUBuilder.core import OpType
import torch
import numpy as np
from .gt_util.aiff import _aiff_lut, clip_to_bit, _aiff_lookup_float_index_lut, _x3_aiff_exp_approximation, x3_aiff_fp32_2_fpx
from .gt_util.quant_utils import clamp, _tpc_lut, linear_dequantize
from .gt_util.metric import relative_diff, ulp_diff, cos_distance, check_fp_data
from .gt_util.dtype_utils import torch_to_numpy
from AIPUBuilder.core import Target
from .gt_util.math_utils import fp32_to_fp24, fexp_tpc

lut_methods = [
    "CELU",
    "ELU",
    "GELU",
    "HARDSWISH",
    "HARDSIGMOID",
    "HARD_SWISH",
    "MISH",
    "SELU",
    "SHRINK",
    "SIGMOID",
    "SILU",
    "SOFTPLUS",
    "SOFTSIGN",
    "SWISH",
    "TANH",
    "THRESHOLDEDRELU",
    "UNKNOWN",
]


def SILU(x, dtype):
    """
    silu(x) = x/(1+e^(-x))
    -----------------------
    This is silu_lut implementatation


    LUT_Generation:
    ---------------
    - lut_x[-10,10]
    - dytpe = "float16"
    - pad the lut_table with the last element
    ```
    def gen_lut(lut_len):
        x = np.linspace(-10, 10, lut_len - 1)
        lut = SiLU(x)
        lut.resize(lut_len)
        lut[-1] = 10
        return lut
    ```
    """

    # keep the lut value same as tpc lib
    def gen_silu_lut(lut_len, lut_edge, dtype):
        x = np.linspace(-lut_edge, lut_edge, lut_len - 1)
        n = lut_len - 1
        lut = np.zeros((512,), dtype=dtype)
        for i in range(n):
            lut[i] = x[i] / (1 + np.exp(-x[i]))
        lut[-1] = lut_edge
        return lut
    nan_index = torch.where(torch.isnan(x))
    x[nan_index] = 0

    lut = torch.from_numpy(gen_silu_lut(512, 10, dtype.np))
    left_edge, right_edge = -10, 10
    lut_size = 512
    x = x.to(torch.float32)
    xx = x.clip(left_edge, right_edge).to(torch.float32)

    LUT_INVERSE_DELTA = (lut_size - 2) / (right_edge - left_edge)
    index_ = (xx-left_edge) * LUT_INVERSE_DELTA
    index = torch.floor((xx-left_edge) * LUT_INVERSE_DELTA).to(torch.long)
    y = lut[index]
    y1 = lut[index+1]
    diff = y1.to(torch.float32) - y.to(torch.float32)
    frac = index_ - index
    iner_value = (diff * frac).to(dtype.torch)
    y = y + iner_value
    y[index >= 510] = x.to(dtype.torch)[index >= 510]
    y[index < 0] = 0
    gty = torch.nn.functional.silu(x).to(dtype.torch)
    # check the lut result is all close to the result of nn.silu
    assert torch.allclose(gty, y, 1e-3, 1e-3)
    y[nan_index] = torch.tensor(float('nan')).half()
    return y


def crelu_float(self, x):
    relu_x = torch.nn.functional.relu(x)
    relu_neg_x = torch.nn.functional.relu(-x)
    output = torch.cat((relu_x, relu_neg_x), dim=self.params['axis'])
    return output


def clip_float(self, x):
    out_max = min(self.params['clip_max'], torch.finfo(self.outputs[0].betensor.dtype).max)
    out_min = max(self.params['clip_min'], torch.finfo(self.outputs[0].betensor.dtype).min)
    return torch.clip(x.to(torch.float32), out_min, out_max)


def shrink_float(self, x):
    lambd, bias = self.params.get('lambd', 0.5), self.params.get('bias', 0)
    x = x.to(torch.float32)
    out = torch.zeros_like(x)
    out[x < -lambd] = x[x < -lambd] + bias
    out[x > lambd] = x[x > lambd] - bias
    return out


def prelu_float(self, x, is_aiff):
    negative_slope = self.constants["negative_slope"].betensor.to(torch.float32)
    if is_aiff:
        negative_slope = fp32_to_fp24(negative_slope, True)
    out = fp32_to_fp24(torch.where(x < 0, negative_slope * x, x)
                       ) if is_aiff else torch.where(x < 0, negative_slope * x, x)
    return out


def leaky_relu_float(self, x, is_aiff):
    negative_slope = self.params["negative_slope_value"]
    negative_slope = torch.tensor(negative_slope, dtype=torch.float32)
    if is_aiff:
        negative_slope = fp32_to_fp24(negative_slope)
    return fp32_to_fp24(torch.where(x < 0, negative_slope * x, x)) if is_aiff else torch.where(x < 0, negative_slope * x, x)


def thresholdrelu_float(self, x):
    return torch.where(x > self.params["alpha"], x, self.params.get("value", 0))


def hardsigmoid(self, x, is_aiff):
    beta = self.params.get("beta", 0)
    alpha = self.params.get('alpha', 1)
    clip_min = self.params.get('clip_min', 0)
    clip_max = self.params.get('clip_max', 1)
    alpha = torch.tensor(alpha, dtype=torch.float32)
    if is_aiff:
        alpha = fp32_to_fp24(alpha)
        output = torch.maximum(torch.tensor(clip_min), torch.minimum(
            fp32_to_fp24(alpha * x) + beta, torch.tensor(clip_max))).to(self.outputs[0].dtype.torch)
    else:
        x = x.to(torch.float32)
        output = torch.maximum(torch.tensor(clip_min), torch.minimum(
            (alpha * x) + beta, torch.tensor(clip_max))).to(self.outputs[0].dtype.torch)
    return output


def hardswish(x, is_aiff):
    if is_aiff:
        alpha = fp32_to_fp24(torch.tensor(1.0 / 6, dtype=torch.float32))
        out0 = fp32_to_fp24(alpha * x) + 0.5
        out1 = torch.maximum(torch.tensor(0), torch.minimum(out0, torch.tensor(1)))
        out2 = fp32_to_fp24(out1, True)
        output = fp32_to_fp24(out2 * x)
    else:
        x = x.to(torch.float32)
        output = torch.nn.functional.hardswish(x)
    return output


def swish_float(self, x):
    alpha = self.params.get('alpha', 1.0)
    output = x * torch.nn.functional.sigmoid(x * alpha)
    return output


def gelu_float(x):
    return torch.nn.functional.gelu(x, approximate="tanh")


def selu_float(self, x):
    return self.params.get('gamma', 0) * torch.nn.functional.elu(x.to(torch.float32), self.params.get("alpha", 1))


def mish_float(self, x):
    x = x.to(torch.float32).numpy()
    log_x = np.log(1 + np.exp(x))
    softplus_x = np.where(x > 20, x, log_x)
    output = x * np.tanh(softplus_x)
    return torch.from_numpy(output)


def softplus_float(self, x):
    x = x.to(torch.float32).numpy()
    log_x = np.log(1 + np.exp(x))
    softplus_x = np.where(x > 20, x, log_x)
    return torch.from_numpy(softplus_x)


def celu_float(self, x):
    x = x.to(torch.float32).numpy()
    alpha = self.params.get('alpha', 1)
    alpha_part = alpha * (fexp_tpc(x / alpha) - 1)
    output = np.where(x > 0, x, alpha_part)
    return torch.from_numpy(np.where(x > 0, x, alpha_part))


activation_func_dict = {
    "TANH": torch.nn.functional.tanh,
    "SIGMOID": torch.nn.functional.sigmoid,
    "HARDSIGMOID": hardsigmoid,
    "HARDSWISH": hardswish,
    "SOFTPLUS": softplus_float,
    "ELU": torch.nn.functional.elu,  # alpha
    "SELU": selu_float,
    "SOFTSIGN": torch.nn.functional.softsign,
    "SILU": torch.nn.functional.silu,
    "MISH": mish_float,
    "CELU": celu_float,
    "GELU": gelu_float,
    "RELU": torch.nn.functional.relu,  # alpha
    "CRELU": crelu_float,
    "CLIP": clip_float,
    "LEAKYRELU": leaky_relu_float,
    "PRELU": prelu_float,
    "RELU6": torch.nn.functional.relu6,
    "SWISH": swish_float,
    "SHRINK": shrink_float,
    "THRESHOLDEDRELU": thresholdrelu_float,
}


@register_gt(OpType.Activation)
def activation_all(self, *args, **kwargs):
    method = self.params['method']

    input_data = self.inputs[0].irtensor

    if self.inputs[0].dtype.is_float:
        is_aiff = False
        if 'is_aiff' in kwargs.keys():
            is_aiff = kwargs['is_aiff']
        if is_aiff:
            input_data = fp32_to_fp24(input_data)
        else:
            input_data = input_data.to(torch.float32)
        args = [input_data]
        if method in ["CRELU", "PRELU", "LEAKYRELU", "SHRINK", "CLIP", "THRESHOLDEDRELU", "HARDSIGMOID", "SWISH", "SELU", "SOFTPLUS", "MISH", "CELU"]:
            args = [self, input_data]
            if method in ['HARDSIGMOID', 'LEAKYRELU', 'PRELU']:
                args.append(is_aiff)
        elif method in ["ELU"]:
            args = [input_data, self.params.get("alpha", 1)]
        elif method in ["HARDSWISH"]:
            args = [input_data, is_aiff]
        output = activation_func_dict[method](*args)
        if is_aiff:
            output = x3_aiff_fp32_2_fpx(output, is_aiff, fpx=self.outputs[0].dtype)
        else:
            output = output.to(self.outputs[0].dtype.torch)
    else:
        fuse = kwargs.get("fuse", False)
        if OpType.BatchNorm in [p.type for p in self.parents] and method in ["LEAKYRELU", "PRELU"] and not self.inputs[0].dtype.is_signed:
            fuse = False
        input_data = input_data.cpu().numpy()
        if self.inputs[0].dtype.is_int and not fuse:
            input_data = np.clip(input_data, np.iinfo(self.inputs[0].dtype.np).min, np.iinfo(
                self.inputs[0].dtype.np).max).astype(self.inputs[0].dtype.np)

        # lut method
        if method in lut_methods:
            if self.inputs[0].dtype.is_int:
                input_data = np.clip(input_data, np.iinfo(self.inputs[0].dtype.np).min, np.iinfo(
                    self.inputs[0].dtype.np).max).astype(self.inputs[0].dtype.np)
            # lut
            if "lut" in self.constants:
                lut_data = self.constants['lut'].betensor.cpu().numpy()
                if self.inputs[0].dtype.is_int:
                    lut_in_bits = 8 if '8' in str(self.inputs[0].dtype) else 16
                    in_is_signed = True if str(self.inputs[0].dtype).startswith('int') else False
                    output = _tpc_lut(input_data, self.constants['lut'].betensor, lut_in_bits, in_is_signed)
                    output = clamp(output, self.outputs[0].dtype)
                else:
                    output = activation_func_dict[method](self.inputs[0].betensor.to(
                        torch.float32)).to(self.outputs[0].dtype.torch)

            else:
                output = activation_func_dict[method](self.inputs[0].betensor.to(
                    torch.float32)).to(self.outputs[0].dtype.torch)

        elif method in ['RELU', 'RELU6', 'RELU1']:
            output = RELU(self, input_data, fuse=fuse)

        elif method == "CLIP":
            output = CLIP(self, input_data, fuse=fuse)

        elif method == 'LEAKYRELU':
            output = LEAKYRELU(self, input_data, fuse=fuse)

        elif method == 'PRELU':
            output = PRELU(self, input_data, fuse=fuse)

        elif method == 'CRELU':
            output = CRELU(self, input_data)
        else:
            raise TypeError('method of activation error')

    self.outputs[0].irtensor = output
    return output


@register_gt(OpType.Activation, target='Z1_0701')
def activation(self, *args, **kwargs):
    target_name = 'Z1_0701'
    method = self.params['method']
    input_data = self.inputs[0].irtensor.cpu().numpy().astype(np.int64)
    input_data = np.clip(input_data, np.iinfo(self.inputs[0].dtype.np).min, np.iinfo(
        self.inputs[0].dtype.np).max).astype(self.inputs[0].dtype.np)

    if method in lut_methods:
        lut_data = self.constants['lut'].betensor.cpu().numpy()
        output = _aiff_lut(input_data, lut_data, input_type=str(self.inputs[0].dtype),
                           target=target_name).astype(
            self.outputs[0].dtype.np)

    elif method in ['RELU', 'RELU6', 'RELU1']:
        output = RELU(self, input_data)

    elif method == "CLIP":
        output = CLIP(self, input_data)

    elif method == 'LEAKYRELU':
        output = LEAKYRELU(self, input_data)

    elif method == 'PRELU':
        output = PRELU(self, input_data)

    elif method == 'CRELU':
        output = CRELU(self, input_data)
    else:
        raise TypeError('method of activation error')

    self.outputs[0].irtensor = output
    return output


@register_gt(OpType.Activation, target='Z1_0701_P')
def activation(self, *args, **kwargs):
    target_name = 'Z1_0701_P'
    method = self.params['method']
    input_data = self.inputs[0].irtensor.cpu().numpy().astype(np.int64)
    input_data = np.clip(input_data, np.iinfo(self.inputs[0].dtype.np).min, np.iinfo(
        self.inputs[0].dtype.np).max).astype(self.inputs[0].dtype.np)

    if method in lut_methods:
        lut_data = self.constants['lut'].betensor.cpu().numpy()
        output = _aiff_lut(input_data, lut_data, input_type=str(self.inputs[0].dtype), target=target_name).astype(
            self.outputs[0].dtype.np)

    elif method in ['RELU', 'RELU6', 'RELU1']:
        output = RELU(self, input_data)

    elif method == "CLIP":
        output = CLIP(self, input_data)

    elif method == 'LEAKYRELU':
        output = LEAKYRELU(self, input_data)

    elif method == 'PRELU':
        output = PRELU(self, input_data)

    elif method == 'CRELU':
        output = CRELU(self, input_data)
    else:
        raise TypeError('method of activation error')

    self.outputs[0].irtensor = output
    return output


def RELU(layer, input_data, fuse=False):
    max_clamp_value = layer.params['max_clamp_value'] if 'max_clamp_value' \
        in layer.params.keys() and layer.params['method'] == 'RELU' else None
    min_clamp_value = layer.params['min_clamp_value'] if 'min_clamp_value' \
        in layer.params.keys() and layer.params['method'] == 'RELU' else None

    output = clip_to_bit(input_data, 32, True)
    if not fuse:
        output = clamp(input_data, dtype=layer.inputs[0].dtype).astype(np.int64)
    output += layer.inputs[0].quantization.offset

    if layer.params['method'] == 'RELU':
        output[output < 0] = 0
    elif layer.params['method'] in ['RELU6', 'RELU1']:
        asym_quant = False
        if layer.inputs[0].quantization.offset != 0 or layer.outputs[0].quantization.offset != 0:
            asym_quant = True
        if asym_quant:
            output[output < 0] = 0
        else:
            if max_clamp_value is None and min_clamp_value is None:
                output[output < 0] = 0
            else:
                output = clamp(output, dtype=layer.outputs[0].dtype,
                               data_range=[min_clamp_value, max_clamp_value])

    output = np.round(output * layer.params['scale_value'] * (0.5 ** layer.params['shift_value'])).astype(np.int64)
    output -= layer.outputs[0].quantization.offset
    output = clamp(output, dtype=layer.outputs[0].dtype)

    return output


def CLIP(layer, input_data, fuse=False):
    max_clamp_value = layer.params['clip_max']
    min_clamp_value = layer.params['clip_min']
    scale_value = layer.params.get("scale_value", 1)
    shift_value = layer.params.get("shift_value", 0)
    final_shift = 8
    output = clip_to_bit(input_data, 32, True)
    if not fuse:
        output = clamp(input_data, dtype=layer.inputs[0].dtype).astype(np.int64)
    output += layer.inputs[0].quantization.offset

    if str(layer.outputs[0].dtype) in ["int8", 'uint8']:
        if shift_value >= final_shift:
            output = np.round(output * scale_value * (0.5 ** (shift_value - final_shift))).astype(np.int64)
        else:
            scale_value = scale_value * 2 ** (final_shift - shift_value)
            while scale_value > (0x8000 - 1):
                final_shift -= 1
                scale_value = scale_value * 0.5
            output = output.astype('int64') * scale_value
        output = np.clip(output, min_clamp_value, max_clamp_value)
        output = np.round(output * (0.5 ** final_shift)).astype(np.int64)
    else:
        output = np.round(output * scale_value * (0.5 ** shift_value)).astype(np.int64)
        output = np.clip(output, min_clamp_value, max_clamp_value)
    output -= layer.outputs[0].quantization.offset
    output = clamp(output, dtype=layer.outputs[0].dtype)

    return output


def LEAKYRELU(layer, input_data, fuse=False):
    output = clip_to_bit(input_data, 32, True)
    if not fuse:
        output = clamp(input_data, layer.inputs[0].dtype).astype(np.int64)
    output += layer.inputs[0].quantization.offset

    negative_slope_shift = layer.outputs[0].dtype.bits
    if 'negative_slope_shift' in layer.params.keys():
        negative_slope_shift = layer.params['negative_slope_shift']

    output[output < 0] = np.round(output[output < 0] *
                                  layer.params['negative_slope_value'] * 0.5 ** negative_slope_shift)

    output = np.round(output * layer.params['scale_value'] * (0.5 ** layer.params['shift_value'])).astype(np.int64)
    output -= layer.outputs[0].quantization.offset
    output = clamp(output, dtype=layer.outputs[0].dtype)

    return output


def PRELU(layer, input_data, fuse=False):
    output = clip_to_bit(input_data, 32, True)
    if not fuse:
        output = clamp(input_data, dtype=layer.inputs[0].dtype).astype(np.int64)
    output += layer.inputs[0].quantization.offset
    if np.prod(layer.constants['negative_slope'].shape) == layer.constants['negative_slope'].shape[-1]:   # [1, 1, c] ->[c]
        negative_slope_data = layer.constants['negative_slope'].betensor.cpu().numpy().flatten()
    else:
        negative_slope_data = layer.constants['negative_slope'].betensor.cpu().numpy()
    if len(negative_slope_data.shape) == 1:
        for i in range(input_data.shape[-1]):
            output[..., i][output[..., i] < 0] = np.round(output[..., i][output[..., i] < 0] *
                                                          negative_slope_data[i] * 0.5 ** layer.params['negative_slope_shift'])
    else:
        if len(negative_slope_data.shape) < len(input_data.shape):
            negative_slope_data = np.expand_dims(negative_slope_data, axis=0)
        negative_slope_data = np.repeat(negative_slope_data, layer.outputs[0].shape[0], axis=0)
        output[output < 0] = np.round(output[output < 0] * negative_slope_data[output < 0]
                                      * 0.5 ** layer.params['negative_slope_shift'])

    output = np.round(output * layer.params['scale_value'] * (0.5 ** layer.params['shift_value'])).astype(np.int64)
    output -= layer.outputs[0].quantization.offset
    output = clamp(output, dtype=layer.outputs[0].dtype)

    return output


def CRELU(layer, input_data):

    input_data = clamp(input_data, dtype=layer.inputs[0].dtype).astype(np.int64)
    input_data = input_data + layer.inputs[0].quantization.offset

    axis = layer.params['axis'] if 'axis' in layer.params.keys() else -1

    if axis not in [0, 3, -1]:
        raise TypeError('Only support axis is 0/3/-1')

    output_i = input_data * 1
    output_i[output_i < 0] = 0
    output_i = np.round(output_i * layer.params['scale_value'] * (0.5 ** layer.params['shift_value'])).astype(np.int64)
    output_i -= layer.outputs[0].quantization.offset
    output_i = clamp(output_i, dtype=layer.outputs[0].dtype)

    n_output_i = input_data * -1
    n_output_i[n_output_i < 0] = 0
    n_output_i = np.round(n_output_i * layer.params['scale_value'] *
                          (0.5 ** layer.params['shift_value'])).astype(np.int64)
    n_output_i -= layer.outputs[0].quantization.offset
    n_output_i = clamp(n_output_i, dtype=layer.outputs[0].dtype)
    output_i = [output_i, n_output_i]
    output = np.concatenate(output_i, axis=axis)

    return output


@register_gt(OpType.Activation, target='X3_1304')
@register_gt(OpType.Activation, target='X3_1302')
@register_gt(OpType.Activation, target='X3_1204')
@register_gt(OpType.Activation, target='X3_1202')
@register_gt(OpType.Activation, target='X3P_1304')
@register_gt(OpType.Activation, target='X3P_1302')
@register_gt(OpType.Activation, target='X3P_1204')
@register_gt(OpType.Activation, target='X3P_1202')
@register_gt(OpType.Activation, target='X3S_1304')
def activation_x3(self, *args, **kwargs):
    is_aiff = kwargs.get("is_aiff",False)
    lut = self.constants["lut"].betensor if 'lut' in self.constants else None
    if self.inputs[0].dtype.is_float and is_aiff and not lut is None:
        inp = self.inputs[0].irtensor
        inp = inp.to(torch.float32)
        index_offset_value = self.params.get("index_offset_value", 0)
        index_scale_value = self.params.get("index_scale_value", 1)
        mirror_mode = self.params.get("lut_mode", "") == "MIRROR"
        value_offset_value = self.params.get("value_offset_value", 0)
        if self.params["method"] in ["GELU", 'MISH', "SILU", "SWISH"]:
            sigmoid = _aiff_lookup_float_index_lut(
                inp,
                lut,
                index_scale_value,
                index_offset_value,
                mirror_mode,
                value_offset_value,
                True
            )
            outp = fp32_to_fp24(sigmoid * inp)
        elif self.params["method"] in ["CELU", "ELU", "SELU", "SOFTPLUS", "SHRINK", "SIGMOID", "SOFTSIGN", "TANH"]:
            outp = _aiff_lookup_float_index_lut(
                inp,
                lut,
                index_scale_value,
                index_offset_value,
                mirror_mode,
                value_offset_value,
            )
        output = x3_aiff_fp32_2_fpx(outp, fpx=self.outputs[0].dtype)
        self.outputs[0].irtensor = output
        return output
    else:
        return activation_all(self, *args, **kwargs)


@register_gen_weight(OpType.Activation)
def gen_random_weight(self):
    _GEN_WEIGHTS['_all'](self)
    if 'lut' in self.constants.keys():
        if self.inputs[0].dtype.is_float and self.params.get('is_perf_mode', False):
            index_scale = self.params['index_scale_value']
            index_offset = self.params['index_offset_value']
            lut_items_in_bits = 9
            lut = linear_dequantize(torch.arange(0, 2**lut_items_in_bits), index_scale, index_offset)
            self.constants['lut'].betensor = lut


@register_gen_input(OpType.Activation)
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
        if str(self.inputs[0].dtype).startswith('bfloat'):
            from ml_dtypes import bfloat16
            input0 = input0.astype(bfloat16)
    else:
        input0 = np.random.randint(np.iinfo(self.inputs[0].dtype.np).min, np.iinfo(self.inputs[0].dtype.np).max,
                                   size=self.inputs[0].shape, dtype=self.inputs[0].dtype.np)
    self.inputs[0].betensor = np.int64(input0) if not self.inputs[0].dtype.is_float else input0
    input_data = [input0]
    return input_data


@register_op_metric(OpType.Activation)
@check_fp_data
def metric(self, outputs, index=None):
    ulp_dict = {'GELU': 5, 'HARDSWISH': 2, 'HARDSIGMOID': 2}

    def call(r, u, c, t, o, gt):
        if "float" in t:
            if not self.params.get('is_perf_mode', False):  # tpc
                max_ulp = ulp_dict.get(self.params['method'], 1)
            else:
                max_ulp = 5
            if np.all(u <= max_ulp):
                return True
            if self.params.get('is_perf_mode', False):
                return c > 0.999 and c < 1.001
            return np.all(r < 0.001) or np.all(np.abs(o - gt)[np.where(r > 0.001)] < 1e-2)
        else:
            return np.all(r == 0)

    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]

    layer_dtypes = [str(o.dtype) for o in self.outputs] if index is None else [str(self.outputs[index].dtype)]

    rel_diffs = relative_diff(outputs, groundtruths)
    ulp_diffs = ulp_diff(outputs, groundtruths)
    cos_dists = cos_distance(outputs, groundtruths)
    if self.outputs[0].dtype.is_float:
        return all(call(r, u, c, t, o, gt) for r, u, c, t, o, gt in zip(rel_diffs, ulp_diffs, cos_dists, layer_dtypes, outputs, groundtruths))

    return _OP_METRICS['_all'](self, outputs, index)
