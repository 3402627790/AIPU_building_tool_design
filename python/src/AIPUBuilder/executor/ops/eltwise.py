from ..engine import register_gt, register_gen_input, _OP_METRICS, register_op_metric
from AIPUBuilder.core import OpType, Dtype
import numpy as np
import torch
from .gt_util import clip_to_bit
from .gt_util.aiff import _aiff_inner_activation, _aiff_final_output, x3_aiff_fp32_2_fpx
from .gt_util.quant_utils import clamp, fp_activation
from .gt_util.metric import relative_diff, ulp_diff, cos_distance, check_fp_data
from .gt_util.dtype_utils import torch_to_numpy
from AIPUBuilder.core import Target, ISAType


def div(input_1, input_2, lut):
    input_2_sign = torch.sign(input_2)
    input2_abs = torch.clip(torch.abs(input_2), 0, 255)
    input2_lut = lut.numpy()[input2_abs.numpy()]
    outputs = input_1 * input2_lut * input_2_sign
    return outputs


elwise_dict = {
    'ADD': lambda x, y: (x + y),
    'SUB': lambda x, y: (x - y),
    'MAX': lambda x, y: torch.where(x > y, x, y),
    'MIN': lambda x, y: torch.where(x < y, x, y),
    'MUL': lambda x, y: x * y,
    'DIV': lambda x, y, lut: div(x, y, lut)
}


@register_gt(OpType.Eltwise)
# @check_datalayout
def eltwise(self, *args, **kwargs):
    self.sgattrs = kwargs['sgattrs'] if 'sgattrs' in kwargs else None
    if self.sgattrs is None:
        self.sgattrs = dict()
    fuse = kwargs['fuse'] if 'fuse' in kwargs else False
    # self.outputs[0]._attrs['is_fuse'] = fuse  # for add_layernorm fp fuse
    is_aiff = True
    if 'is_aiff' in kwargs.keys():
        is_aiff = kwargs['is_aiff']

    if self.inputs[0].dtype.is_float:
        is_swap = self.sgattrs['is_inputs_swapped'] if 'is_inputs_swapped' in self.sgattrs else False
        is_trunc = (self.type == OpType.Sub) and (not is_swap)
        return fp_eltwise(self, fuse=fuse, is_aiff=is_aiff, is_trunc=is_trunc)

    if is_aiff or (self.inputs[0].dtype in [Dtype.INT8, Dtype.UINT8] and Target().isa >= ISAType.ZYv5):
        return aiff_eltwise(self, *args, **kwargs)
    else:
        return tpc_eltwise(self, *args, **kwargs)


def aiff_eltwise(self, *args, **kwargs):
    method = self.params["method"]
    inp0 = self.inputs[0].irtensor.long()
    inp1 = self.inputs[1].irtensor.long()
    if len(self.inputs) == 3:  # gemm op
        inp2 = self.inputs[2].irtensor.long()
    inp0 = inp0 + self.inputs[0].quantization.offset
    inp1 = inp1 + self.inputs[1].quantization.offset

    shift = self.params.get('shift_value', 0)
    if method in ['ADD', 'MAX', 'MIN', 'SUB']:
        scale_o, scale_i0, scale_i1 = self.params.get("scale_value", [1, 1, 1])
        inp0 = torch.clip(inp0 * scale_i0, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
        inp1 = torch.clip(inp1 * scale_i1, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    elif self.params["method"] in ['MUL', 'DIV']:
        scale_o = self.params.get("scale_value", 1)

    if method == 'DIV':
        params = [inp0, inp1, self.constants['lut'].betensor]
    else:
        params = [inp0, inp1]
    outputs = elwise_dict[self.params['method']](*params)
    del inp0, inp1, params
    outputs = clip_to_bit(outputs, 32)
    outputs, final_scale, final_shift = _aiff_inner_activation(self, outputs, scale_o, shift)
    # final doquant and -zp
    outputs = _aiff_final_output(self, outputs, final_scale, final_shift)
    outputs = clamp(outputs, self.outputs[0].dtype)
    self.outputs[0].irtensor = outputs
    return outputs


def tpc_eltwise(self, *args, **kwargs):
    method = self.params["method"]
    if method in ['ADD', 'MAX', 'MIN', 'SUB']:
        scale_values = self.params.get("scale_value", [1, 1, 1])
        if scale_values is None: #CP-26149
            scale_values = [1, 1, 1]
        scale_o, scale_i0, scale_i1 = scale_values
     
    elif self.params["method"] in ['MUL', 'DIV']:
        scale_o = self.params.get("scale_value", 1)
        if scale_o is None:
            scale_o = 1
    with_activation = self.params.get(self.params["with_activation"], None)
    shift = self.params.get('shift_value', 0)

    if self.params['method'].lower() == 'mul' and self.inputs[0].dtype != Dtype.INT32:  # mul always run aiff CP-4709
        return aiff_eltwise(self, *args, **kwargs)

    if self.inputs[0].dtype in [Dtype.INT8, Dtype.UINT8]:
        if self.outputs[0].dtype.bits == 16:
            # tpc i8 o16 is a special inner op
            # scale=1/shift=0/with_activation=none
            input_1 = self.inputs[0].betensor.to(torch.int32) + self.inputs[0].quantization.offset
            input_2 = self.inputs[1].betensor.to(torch.int32) + self.inputs[1].quantization.offset
            if method == 'ADD':
                output = (input_1 + input_2)
            elif method == 'SUB':
                output = (input_1 - input_2)
            else:
                raise TypeError('do not support method = {} !'.format(method))
            output -= self.outputs[0].quantization.offset
            output = clamp(output, self.outputs[0].dtype)
            self.outputs[0].betensor = output
            return output
        else:
            method = self.params["method"]
            inp0 = self.inputs[0].betensor.long()
            inp1 = self.inputs[1].betensor.long()
            inp0 = inp0 + self.inputs[0].quantization.offset
            inp1 = inp1 + self.inputs[1].quantization.offset
            inp0 = torch.round(inp0 * 0.5)
            inp1 = torch.round(inp1 * 0.5)
            shift -= 1
            if method in ['ADD', 'MAX', 'MIN', 'SUB']:
                inp0 = torch.clip(inp0 * scale_i0, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
                inp1 = torch.clip(inp1 * scale_i1, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
            if method == 'DIV':
                params = [inp0, inp1, self.constants['lut'].betensor]
            else:
                params = [inp0, inp1]
            outputs = elwise_dict[self.params['method']](*params)
            outputs = clip_to_bit(outputs, 32)
            outputs, final_scale, final_shift = _aiff_inner_activation(self, outputs, scale_o, shift)
            # final doquant and -zp
            outputs = _aiff_final_output(self, outputs, final_scale, final_shift)
            outputs = clamp(outputs, self.outputs[0].dtype)
            self.outputs[0].betensor = outputs
            return outputs
    elif self.inputs[0].dtype in [Dtype.INT16, Dtype.UINT16, Dtype.INT32]:
        if self.outputs[0].dtype.bits == 16:
            return aiff_eltwise(self, *args, **kwargs)
            # input_1 = self.inputs[0].betensor.long()
            # input_2 = self.inputs[1].betensor.long()
            # if scale_o >= 2 ** 8 or with_activation == 'PRELU':  # Judge whether is 16bit output scale or not.
            #     # Determine outputs type range consider signed or not, the sub result must be signed, therefore use i32
            #     if self.inputs[0].dtype.is_signed or self.inputs[0].dtype.is_signed or method.upper() == 'SUB':
            #         max_16bit = np.iinfo(np.int16).max
            #         min_16bit = np.iinfo(np.int16).min
            #         max_32bit = np.iinfo(np.int32).max
            #         min_32bit = np.iinfo(np.int32).min
            #         np_16bit = np.int16
            #         np_32bit = np.int32
            #     else:
            #         max_16bit = np.iinfo(np.uint16).max
            #         min_16bit = np.iinfo(np.uint16).min
            #         max_32bit = np.iinfo(np.uint32).max
            #         min_32bit = np.iinfo(np.uint32).min
            #         np_16bit = np.uint16
            #         np_32bit = np.uint32
            #
            #     if method.upper() in ['ADD', 'SUB', 'MAX', 'MIN']:
            #         out1 = input_1 * scale_i0
            #         out2 = input_2 * scale_i1
            #         if method.upper() == 'ADD':
            #             outputs = out1 + out2
            #         elif method.upper() == 'SUB':
            #             outputs = out1 - out2
            #         elif method.upper() == 'MAX':
            #             outputs = torch.where(out1 > out2, out1, out2)
            #         elif method.upper() == 'MIN':
            #             outputs = torch.where(out1 < out2, out1, out2)
            #     elif method.upper() == 'MUL':
            #         if (not self.inputs[0].dtype.is_signed) and self.inputs[1].is_signed:
            #             input_1, input_2 = input_2, input_1
            #         outputs = input_1 * input_2
            #     else:
            #         raise TypeError('do not support method = %s' % method)
            #     added = outputs.cpu().numpy()
            #     added = np.clip(outputs, min_32bit, max_32bit)
            #     # 2.1 high 16bits of `added` * os
            #     hi = np_32bit(np.clip(np_16bit(np.floor(added * (0.5 ** 16))), min_16bit, max_16bit)) * scale_o
            #     # 2.2 low 16bits of `added` * os, then shift 16(same scale with high bits before merged sum)
            #     lo = np.uint32(np.round((added & 0x0000ffff) * scale_o * (0.5 ** 16)))
            #     # 3. final shift
            #     # note: 0.5 ** 0 == 1
            #     hi_and_lo = hi + lo
            #     shifted_hi_and_lo = np.round(hi_and_lo * (0.5 ** (shift - 16)))
            #     outputs = np.clip(shifted_hi_and_lo, min_16bit, max_16bit)
            #     outputs = torch.from_numpy(outputs)
            # else:  # scale_o <= 255
            #     if method.upper() in ['ADD', 'SUB', 'MAX', 'MIN']:
            #         output1 = torch.round(input_1 * scale_o * 0.5 ** 8)
            #         output2 = torch.round(input_2 * scale_o * 0.5 ** 8)
            #         output1 *= scale_i0
            #         output2 *= scale_i1
            #         if method.upper() == 'ADD':
            #             outputs = output1 + output2
            #         elif method.upper() == 'SUB':
            #             outputs = output1 - output2
            #         elif method.upper() == 'MAX':
            #             outputs = torch.where(output1 > output2, output1, output2)
            #         elif method.upper() == 'MIN':
            #             outputs = torch.where(output1 < output2, output1, output2)
            #     elif method.upper() == 'MUL':
            #         if (not self.inputs[0].dtype.is_signed) and self.inputs[1].is_signed:
            #             input_1, input_2 = input_2, input_1
            #         outputs = input_1 * scale_o
            #         outputs = torch.round(outputs * (0.5 ** 8))
            #         outputs = outputs * input_2
            #     else:
            #         raise TypeError('do not support method = %s' % method)
            #     outputs = torch.round(outputs * (0.5 ** (shift - 8)))
            #
            # # Clamp before activation
            # outputs = torch.clip(torch.round(outputs), np.iinfo(np.int32).min, np.iinfo(np.int32).max)
            # outputs = outputs.to(torch.int32)
            #
            # # Activation: clip/leakyrelu/relu/prelu
            # if with_activation == 'CLIP':
            #     outputs = torch.clip(outputs, self.params["clip_min"], self.params["clip_max"])
            # elif with_activation == "LEAKRELU":
            #     if self.outputs[0].dtype.bits == 8 and self.inputs[0].dtype.bits == 16:
            #         pass
            #     else:
            #         outputs = torch.clip(outputs, np.iinfo(self.output_type).min,
            #                              np.iinfo(self.output_type).max).to(torch.long)
            #     outputs[outputs < 0] = torch.round(
            #         outputs[outputs < 0] * self.params["negative_slope_value"] * 0.5 ** 8)
            # elif with_activation == 'RELU':
            #     outputs[outputs < 0] = 0
            # elif with_activation == 'PRELU':
            #     # PReLU / per channel
            #     for i in range(0, outputs.shape[-1]):
            #         outputs_c = outputs[..., i]
            #         outputs_c[outputs_c < 0] = np.round(
            #             outputs_c[outputs_c < 0] * self.negative_slope[i] * (0.5 ** self.negative_slope_shift))
            #     outputs = np.clip(outputs, -2 ** 31, 2 ** 31 - 1)
            # elif with_activation == "NONE":
            #     pass
            # outputs = outputs.to(torch.long)
            # outputs -= self.outputs[0].quantization.offset
            # outputs = outputs.to(self.outputs[0].dtype.torch)
            # self.outputs[0].betensor = outputs
            # return outputs
        else:
            # tpc i16 o8 is a especial inner op
            input_1 = self.inputs[0].betensor.to(torch.int32)
            input_2 = self.inputs[1].betensor.to(torch.int32)
            params = [input_1, input_2]
            output = elwise_dict[self.params['method']](*params)
            output -= self.outputs[0].quantization.offset
            output = clamp(output, self.outputs[0].dtype)
            self.outputs[0].betensor = output
            return self.outputs[0].betensor


def fp_eltwise(self, fuse=False, is_aiff=False, is_trunc=False):
    inp0 = self.inputs[0].irtensor.to(torch.float32)
    inp1 = self.inputs[1].irtensor.to(torch.float32)

    # for ds, the two input sizes may be different, CP-14520
    inp0_size = np.prod(inp0.shape)
    inp1_size = np.prod(inp1.shape)

    if not can_broadcast(inp0, inp1) and inp1_size > inp0_size:
        inp1 = inp1.flatten()[:inp0_size].reshape(inp0.shape)
    elif not can_broadcast(inp0, inp1) and inp0_size > inp1_size:
        inp0 = inp0.flatten()[:inp1_size].reshape(inp1.shape)

    from .gt_util.math_utils import fp32_to_fp24
    if self.params['method'].upper() == "ADD":
        outputs = inp0 + inp1
    elif self.params['method'].upper() == "SUB":
        if is_aiff and is_trunc:
            outputs = fp32_to_fp24(inp0 - inp1, is_trunc)
        else:
            outputs = inp0 - inp1
    elif self.params['method'].upper() == "MUL":
        if is_aiff:
            outputs = fp32_to_fp24(inp0 * inp1)
        else:
            outputs = inp0 * inp1
    elif self.params['method'].upper() == "MAX":
        outputs = torch.maximum(inp0, inp1)
    elif self.params['method'].upper() == "MIN":
        outputs = torch.minimum(inp0, inp1)

    if is_aiff and 'with_activation' in self.params.keys():
        if self.params['with_activation'].upper() == 'RELU':
            outputs = torch.nn.functional.relu(outputs)
        elif self.params['with_activation'].upper() == 'RELU6':
            outputs = torch.nn.functional.relu6(outputs)
        elif self.params['with_activation'].upper() == 'NONE':
            outputs = outputs
        elif self.params['with_activation'].upper() == 'CLIP':
            outputs = torch.clamp(outputs, self.params['clip_min'], self.params['clip_max'])
        elif self.params['with_activation'].upper() == 'LEAKYRELU':
            negative_slope = self.params['negative_slope_value']
            negative_slope = fp32_to_fp24(negative_slope)
            outputs = fp32_to_fp24(torch.where(outputs < 0, negative_slope * outputs, outputs))
        elif self.params['with_activation'].upper() == 'PRELU':
            negative_slope = self.constants['negative_slope'].betensor
            negative_slope = fp32_to_fp24(negative_slope.to(torch.float32), True)
            outputs = fp32_to_fp24(torch.where(outputs < 0, negative_slope * outputs, outputs))
    else:
        outputs = fp_activation(outputs, self)

    from copy import deepcopy
    bak_out_dtype = deepcopy(self.outputs[0].dtype)
    if fuse and not is_aiff:
        self.outputs[0].dtype = Dtype.FP32
    else:
        outputs = x3_aiff_fp32_2_fpx(outputs, is_aiff, fpx=self.outputs[0].dtype)
    self.outputs[0].irtensor = outputs
    if fuse and not is_aiff:
        self.outputs[0].dtype = bak_out_dtype
    return outputs


def can_broadcast(arr1, arr2):
    """
    Determine whether two arrays can be broadcast.
    """
    shape1 = arr1.shape
    shape2 = arr2.shape
    max_len = max(len(shape1), len(shape2))
    # Pad the array with fewer dimensions with 1s at the front to make the lengths of the two shapes the same.
    shape1 = (1,) * (max_len - len(shape1)) + shape1
    shape2 = (1,) * (max_len - len(shape2)) + shape2

    for i in range(max_len):
        if shape1[i] != shape2[i] and shape1[i] != 1 and shape2[i] != 1:
            return False
    return True


@register_gen_input(OpType.Eltwise)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_shape1 = list(self.inputs[1].shape)
    in_type1 = str(self.inputs[1].dtype)

    if self.inputs[0].dtype.is_float:
        input0 = np.random.uniform(-1000, 1000, size=in_shape0).astype('float16')
        input1 = np.random.uniform(-1000, 1000, size=in_shape1).astype('float16')
        if in_type0.startswith('bfloat'):
            from ml_dtypes import bfloat16
            input0 = input0.astype(bfloat16)
            input1 = input1.astype(bfloat16)
    else:
        input0 = None
        input1 = None
        if isinstance(self.params['scale_value'], list) and len(self.params['scale_value']) == 3:
            if in_type0 == 'uint16':  # avoid input0*scale1+input1*scale2 exceed int32, align to opt
                bit = np.ceil(np.log2(self.params['scale_value'][1]))
                max_v = 2 ** min(16, 30 - bit)
                input0 = np.random.randint(0, max_v, size=in_shape0, dtype=in_type0)
            if in_type1 == 'uint16':
                bit = np.ceil(np.log2(self.params['scale_value'][2]))
                max_v = 2 ** min(16, 30 - bit)
                input1 = np.random.randint(0, max_v, size=in_shape1, dtype=in_type1)
        elif isinstance(self.params['scale_value'], int) or len(self.params['scale_value']) == 1:
            if in_type0 == 'uint16' and in_type1 == 'uint16':  # avoid input0*input1 exceed int32, align to opt
                input0 = np.random.randint(0, 65536, size=in_shape0, dtype=in_type0)
                input1 = np.random.randint(0, 32768, size=in_shape1, dtype=in_type1)
        if input0 is None:
            input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)
        if input1 is None:
            input1 = np.random.randint(np.iinfo(in_type1).min, np.iinfo(in_type1).max, size=in_shape1, dtype=in_type1)

    input_data = [input0, input1]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    return input_data


@register_op_metric(OpType.Eltwise)
@check_fp_data
def metric(self, outputs, index=None):
    def call(r, u, c, t):
        if "float" in t:
            if np.all(u <= 1):
                return True
            return np.all(r < 0.001)
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
        return all(call(r, u, c, t) for r, u, c, t in zip(rel_diffs, ulp_diffs, cos_dists, layer_dtypes))

    return _OP_METRICS['_all'](self, outputs, index)
