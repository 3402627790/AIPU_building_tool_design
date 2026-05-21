from ..engine import register_gt, register_gen_weight, register_gen_input, _OP_METRICS, register_op_metric
from .gt_util.quant_utils import clamp, _tpc_lut
from .gt_util.math_utils import *
from .gt_util.metric import relative_diff, ulp_diff, cos_distance, check_fp_data
from .gt_util.dtype_utils import torch_to_numpy
from .gt_util.aiff import get_ops_options
from AIPUBuilder.core import OpType
from AIPUBuilder.core.tensor import use_cuda
from AIPUBuilder.logger import FATAL, DEBUG
import torch
import numpy as np
import math


@register_gt(OpType.Softmax)
def softmax(self, *args, **kwargs):
    input_data = self.inputs[0].irtensor

    is_perf_mode = self.params.get('is_perf_mode', 'false')
    is_aiff = True if str(is_perf_mode).lower() == 'true' else False
    is_aiff = kwargs['is_aiff'] if 'is_aiff' in kwargs.keys() else is_aiff
    tpc_or_aiff = 'aiff' if is_aiff else 'tpc'
    DEBUG('run softmax: %s version !' % tpc_or_aiff)
    self.attrs['is_aiff'] = is_aiff  # for softmax's metric

    if self.inputs[0].dtype.is_float:
        fast_exp_fp16 = True if get_ops_options("FAST_EXP_FP16") == '1' else False
        faster_exp_fp16 = True if get_ops_options("FASTER_EXP_FP16") == '1' else False
        if is_aiff:
            output = softmax_aiff_float(input_data, self.params['axis'])
            self.outputs[0].irtensor = output
        elif fast_exp_fp16 or faster_exp_fp16:
            if fast_exp_fp16:
                output = fast_exp_softmax(input_data, self.params['axis'])
            else:
                output = faster_exp_softmax(input_data, self.params['axis'])
            self.outputs[0].irtensor = output
        else:
            output = forward_float(self, input_data)
            self.outputs[0].irtensor = output
    else:
        if is_aiff or ('lut' in self.constants and self.constants['lut'].dtype.is_float):
            output = softmax_aiff_int(self, input_data, self.params['axis'])
        else:
            output = forward_quantize(self)
        self.outputs[0].irtensor = output
    return self.outputs[0].betensor


def fast_exp_softmax(inp, axis):
    DEBUG('[executor] softmax fp fast_exp...')
    inp_max = inp.max(axis, keepdim=True).values
    inp_reg = torch.clamp(inp - inp_max, -85, 0).float()
    z = torch.tensor(1.4426950409).float()  # .half()
    mul_tmp = z * inp_reg
    mul_tmp = mul_tmp.float()
    ztail = torch.tensor(126.94201519)
    mul_tmp += ztail
    pow_tmp = torch.tensor(8388608).float()
    ex_lim = (pow_tmp * mul_tmp).round().int()
    # ex_lim = ex_lim >> 8  # fp32 -> fp24
    # ex_lim = ex_lim << 8
    data = ex_lim.view(torch.float32)

    data_sum = data.sum(axis, keepdim=True)
    data_int = data_sum.view(torch.int32)
    # data_int = data_int >> 8
    # data_int = data_int << 8
    data_sum = data_int.view(torch.float32)
    y = data / data_sum
    y = y.to(torch.float16)
    return y


def faster_exp_softmax(inp, axis):
    DEBUG('[executor] softmax fp faster_exp...')
    inp_max = inp.half().max(axis, keepdim=True).values
    inp_reg = torch.clamp(inp - inp_max, -10, 0).half()
    z = torch.tensor(1.4426950409).half()
    mul_tmp = z * inp_reg
    ztail = torch.tensor(14.9420151853).half()
    mul_tmp += ztail
    pow_tmp = torch.tensor(1024).half()
    ex_lim = (pow_tmp * mul_tmp).float().floor().short()
    data = ex_lim.view(torch.float16)

    in_shape = inp.shape
    tmp_len = in_shape[-1]  # only support axis=-1
    tmp1_len = np.prod(in_shape) // tmp_len
    data1 = data.reshape([tmp1_len, tmp_len])
    data_sum = torch.zeros([tmp1_len, 16]).half()
    data_sum = sum_fp16(data1, tmp_len, data_sum).float()
    data_sum = torch.sum(data_sum, dim=-1, keepdims=True).half()  # _vrpadd

    data_sum2 = data.sum(axis, keepdim=True)
    y = data1 / data_sum
    return y.half().reshape(in_shape)


def forward_float(self, input_data):
    inputs = input_data.float()
    in_dim = inputs.ndim
    axis = self.params['axis']
    axis = axis + in_dim if axis < 0 else axis

    trans_index = list(range(in_dim))
    trans_index.append(trans_index.pop(axis))
    new_shape = list(inputs.shape)
    new_shape.append(new_shape.pop(axis))

    restore_index = list(range(in_dim))
    restore_index.insert(axis, restore_index[-1])
    restore_index.pop()

    inputs = torch.permute(inputs, trans_index)  # # transpose, calc across the last dim
    inputs = torch.reshape(inputs, [-1, inputs.shape[-1]])  # n-dim -> 2-dim
    inputs -= torch.max(inputs, axis=-1, keepdim=True)[0]
    output = fexp_tpc(inputs)

    in_shape = output.shape
    tmp_len = in_shape[-1]
    tmp1_len = np.prod(in_shape) // tmp_len
    data_tmp = torch.zeros([tmp1_len, 8]).float()
    data_tmp = sum_square_fp16(output, tmp_len, data_tmp)
    sum = np.sum(data_tmp.float().cpu().numpy(), axis=-1, keepdims=True)  # _vrpadd
    sum = torch.from_numpy(sum)

    out = torch.mul(output, (1/sum))
    out = out.to(self.outputs[0].dtype.torch)

    out_shape = list(self.outputs[0].betensor.shape)
    if "_output_shapes" in self.attrs:
        out_shape = self.attrs.get("_output_shapes")[0]
    out = out.reshape(new_shape)  # 2-dim -> n-dim
    out = torch.permute(out, restore_index)  # transpose
    output = out.reshape(out_shape)

    return output


#####################################################################
# slimulate hardware behaviour to implement float softmax by LUT
#####################################################################
def softmax_aiff_float(vx, axis):
    # e^x = 2 ^(x/ln2), 1/ln2=1.442695, e^(x-max)/sum(e^(xi-max))=e^x/sum(e^xi), avoid overflow caused by big value
    vx = vx.float()
    vx = fp32_to_fp24(vx)
    max_v, _ = vx.max(axis, keepdim=True)

    tmp = fp32_to_fp24(torch.Tensor([-1 / math.log(2)]).to(vx.device))
    f_vdata = fp32_to_fp24((max_v - vx) * tmp)

    yy = fexp(f_vdata)
    y_sum = yy.sum(axis, keepdim=True)

    # convert to fp24
    # score2 = ((1 / y_sum).view(torch.int) & 0xFFFFff00).view(torch.float32)
    score = fp32_to_fp24(1 / y_sum)

    # convert yy to fp16
    yy16 = yy.half()  # middle precision always should be fp16(1/5/10) which better than bfp16(1/8/7)

    yy24 = fp32_to_fp24(yy16.float())
    score = yy24 * score
    # convert f32 softmax result to fp16 output
    score = fp32_to_fp24(score)

    f16 = score.to(vx.dtype)

    return f16.reshape(vx.shape)


def softmax_aiff_int(self, vx, axis):
    max_v, _ = vx.max(axis, keepdim=True)
    max_v = fp32_to_fp24(max_v.float())
    vx = vx.float()
    vx = fp32_to_fp24(vx)
    # e^x = 2 ^(x/ln2), 1/ln2=1.442695, e^(x-max)/sum(e^(xi-max))=e^x/sum(e^xi), avoid overflow caused by big value
    # max_v, _ = vx.max(axis, keepdim=True)
    tmp = fp32_to_fp24(torch.Tensor([-1 / (math.log(2) * self.inputs[0].quantization.scale)]))
    f_vdata = fp32_to_fp24((max_v - vx) * tmp.to(vx.device))

    yy = fexp(f_vdata)
    y_sum = yy.sum(axis, keepdim=True)
    y_sum = fp32_to_fp24(y_sum)

    # convert to fp24
    score = fp32_to_fp24(1 / y_sum)

    # convert yy to fp16
    yy16 = yy.half()

    yy24 = fp32_to_fp24(yy16.float())

    layer_top_scale = self.outputs[0].quantization.scale
    yy24 = yy24 * layer_top_scale
    score = yy24 * score
    score = torch.round(score).int()
    score -= self.outputs[0].quantization.offset
    score = clamp(score, dtype=self.outputs[0].dtype)

    return score.reshape(vx.shape)


def forward_quantize(self, *args, **kwargs):
    is_nchwc32 = False
    if self.inputs[0].shape.layout.name == 'NCHWC32':
        is_nchwc32 = True

    inputs = self.inputs[0].irtensor.float()
    qbits = self.inputs[0].dtype.bits
    in_shape = inputs.shape
    axis = self.params['axis']
    axis = axis + inputs.ndim if axis < 0 else axis
    max_v, _ = inputs.max(axis, keepdim=True)
    x = inputs

    adjust_q = self.params.get('adjust_q', -1)
    adjust_q = -1 if adjust_q is None else adjust_q
    new_alg = adjust_q >= 0

    if new_alg:
        if adjust_q > 1 and adjust_q < 8:
            FATAL("scalebits second list member value is 0,1 or 8,9,10,11,12,13,14,15")
        input_scale_value = self.params.get("input_scale_value", 3)
        input_shift_value = self.params.get("input_shift_value", 8)
        Pow2OutQ = self.params["pow2_out_q"]
        log2e_x = (x - max_v).int()
        del x
        if adjust_q >= 8:
            if (input_shift_value - adjust_q) > 0:
                log2e_x = (log2e_x * input_scale_value) >> (input_shift_value - adjust_q)
            else:
                log2e_x = (log2e_x * input_scale_value) << (adjust_q - input_shift_value)
            y = Pow2(log2e_x, adjust_q, Pow2OutQ)
        else:
            log2e_x = (log2e_x * input_scale_value) >> input_shift_value
            y = (1 << (Pow2OutQ + log2e_x)) * (
                (Pow2OutQ + log2e_x) >= 0)  # *((Pow2OutQ+log2e_x)>0) to fix torch version diff issue
        y_sum = y.sum(axis, keepdim=True, dtype=torch.long)
        del y
    else:
        lut = self.constants['lut'].betensor
        x = inputs - max_v + 2 ** qbits - 1
        x = x.contiguous().view(-1,)
        y = _tpc_lut(x, lut, qbits, False)
        y = torch.clamp(y, -2 ** 31, 2 ** 31 - 1)
        y = y.view(in_shape)
        y_sum = y.sum(axis, keepdim=True, dtype=torch.long)

    '''
    now we support two kind of softmax quantized forward, the one shape_value_in_axis < 8 and in_size % 128 == 0 case
    existed is because of in small data numbers the saved 11bits acurracy in division is faster than 31bits, the lib
    has better performance, so in order to compatiblity the lib impl, we also has two kind of quantized forward.
    '''
    if new_alg:
        qbits = self.outputs[0].dtype.bits
        y_sum = torch.clamp(y_sum, 1, (1 << 31) - 1)
        if adjust_q >= 8:
            y_sum = Log2_norm(y_sum, Pow2OutQ, adjust_q)
            out = Pow2(log2e_x - y_sum, adjust_q, qbits)
            del y_sum
        else:
            log2_sum = simple_log2(y_sum, Pow2OutQ, adjust_q)
            yy_div_sum = (((log2e_x + qbits) << adjust_q) - log2_sum) >> adjust_q
            out = (1 << yy_div_sum) * (yy_div_sum >= 0)  # *(yy_div_sum>0) to fix torch version diff issue
    else:
        # 31->30: Adaptation the lut.max=1 case in android and these case lib would overflow using base_bits=31
        base_bits = 30 if lut.max() == 1 else 31
        shape_value_in_axis = self.inputs[0].betensor.shape[axis]
        in_size = self.inputs[0].betensor.numel() / shape_value_in_axis
        if shape_value_in_axis < 8 and in_size % 128 == 0 and self.inputs[0].dtype.bits == 8:
            scale_bits = torch.ceil(torch.log2(lut.max())).long()
            enlarge_bits = base_bits - scale_bits
            numerator = (y.long() << enlarge_bits) + (y_sum >> 1)
            denominator = torch.maximum(y_sum, torch.ones_like(y_sum))
            z = torch.div(numerator, denominator, rounding_mode='trunc')
            shift_value = self.params['shift_value'] + 11
            out = torch.round(z * self.params['scale_value'] * (0.5 ** shift_value))
        else:
            denominator = torch.maximum(y_sum, torch.ones_like(y_sum))
            scale_bits = torch.ceil(torch.log2(torch.tensor(self.params['scale_value']))).long()
            enlarge_bits = base_bits - scale_bits
            enlarge_shift = self.params['shift_value'] + enlarge_bits
            enlarge_scale = self.params['scale_value'] * (2 ** enlarge_bits)
            reverse_sum = torch.div(enlarge_scale, denominator, rounding_mode='trunc')
            if is_nchwc32:
                reverse_sum, left_shift = convert_less_mbit(reverse_sum.cpu().numpy(), 2 ** 16 - 1)
                y_div_sum = (y.long() << left_shift) * reverse_sum
            else:
                y_div_sum = y.long() * reverse_sum

            out = torch.round(y_div_sum.double() * 1 * (0.5 ** enlarge_shift))

    out = out - self.outputs[0].quantization.offset
    out = clamp(out, self.outputs[0].dtype)
    return out


def convert_less_mbit(data, threshold_max):
    left_shift = np.zeros_like(data)
    while np.max(data) >= threshold_max:
        left_shift = np.where(data >= threshold_max, left_shift + 1, left_shift)
        data = np.where(data >= threshold_max, data >> 1, data)
    return data, left_shift


@register_gen_weight(OpType.Softmax)
def gen_random_weight(self):
    # lut for calc e ** i
    if 'lut' in self.constants:
        in_dtype = str(self.inputs[0].dtype)
        if self.constants['lut'].dtype.is_float:  # aiff softmax
            lut = pow2_f_lut
        else:
            is_nchwc32 = True if str(self.constants['lut'].dtype) == 'uint16' else False
            lut_bit = 16 if is_nchwc32 else 20
            mbits = lut_bit
            max_val = 2 ** lut_bit - 1
            in_scale = np.random.randint(1, 256)  # Determined the maximum value of lut
            in_zp = np.random.randint(-255, 256)  # Determine the distribution of lut
            max_inp = np.clip(np.log(max_val) * in_scale - in_zp, -2 ** 63, 2 ** 63 - 1)

            lut_items_in_bits = int(np.log2(np.prod(self.constants['lut'].shape)))
            in_qbits = 8 if str(self.inputs[0].dtype).endswith('8') else 16
            lsteps = 2 ** min(in_qbits, lut_items_in_bits)
            tmp = np.linspace(0, 2 ** in_qbits - 1, num=lsteps) + max_inp - (2 ** in_qbits - 1)
            lut = (np.asarray(tmp).flatten() + in_zp) / in_scale
            lut = np.clip(np.round(np.exp(lut)), 0, max_val)

        self.constants['lut'].betensor = np.asarray(lut, dtype=str(self.constants['lut'].dtype))
    if 'float_lut' in self.constants:
        self.constants['float_lut'].betensor = np.asarray(pow2_f_lut, dtype=str(self.constants['float_lut'].dtype))


@register_gen_input(OpType.Softmax)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_type0 = str(self.inputs[0].dtype)

    if self.inputs[0].dtype.is_float:
        # input0 = np.float16((np.random.random(size=in_shape0) - 0.5) * 100)
        input0 = np.random.uniform(-1, 1, size=in_shape0).astype(in_type0)  # softmax need small data
    else:
        input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)

    input_data = [input0]
    self.inputs[0].betensor = input_data[0]
    return input_data


@register_op_metric(OpType.Softmax)
@check_fp_data
def metric(self, outputs, index=None):
    # note: aiff's metric maybe called in layoutconvertor
    def aiff_call(r, u, c, t):
        if t in ["float16", "bfloat16"]:
            return np.all(u <= 2)  # aiff fp <=2
        return np.all(r <= 1)

    def fast_exp_call(r, u, c, t):
        return np.all(u <= 1)

    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]

    layer_dtypes = [str(o.dtype) for o in self.outputs] if index is None else [str(self.outputs[index].dtype)]

    rel_diffs = relative_diff(outputs, groundtruths)
    ulp_diffs = ulp_diff(outputs, groundtruths)
    cos_dists = cos_distance(outputs, groundtruths)
    fast_exp_fp16 = True if get_ops_options("FAST_EXP_FP16") == '1' else False
    if self.attrs.get('is_aiff', False):
        return all(aiff_call(r, u, c, t) for r, u, c, t in zip(rel_diffs, ulp_diffs, cos_dists, layer_dtypes))
    if fast_exp_fp16:
        return all(fast_exp_call(r, u, c, t) for r, u, c, t in zip(rel_diffs, ulp_diffs, cos_dists, layer_dtypes))

    return _OP_METRICS['_all'](self, outputs, index)
