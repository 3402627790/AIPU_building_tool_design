from ..engine import register_gt, register_gen_weight, register_gen_input, _OP_METRICS, register_op_metric
from .gt_util.dtype_utils import bits2range
from .gt_util.quant_utils import _tpc_lut, clamp
from .gt_util.metric import relative_diff, ulp_diff, cos_distance, check_fp_data
from .gt_util.dtype_utils import torch_to_numpy
from AIPUBuilder.core import OpType, Target
import torch
import numpy as np
from .gt_util.aiff import x3_aiff_fp32_2_fpx


@register_gt(OpType.Reduce)
def reduce(self, *args, **kwargs):
    inp = self.inputs[0].irtensor

    method = self.params['method']
    axis = self.params['axis']
    sgattrs = kwargs.get("sgattrs", {})
    if self.inputs[0].dtype.is_float:
        outp = fp_reduce(self, inp, **kwargs)
        if method.upper() not in ['ANY', 'ALL']:
            outp = outp.to(self.outputs[0].dtype.torch)
        self.outputs[0].irtensor = outp
        return self.outputs[0].betensor
    elif sgattrs.get("dequantize_int_ir", False) == True:
        inp = self.inputs[0].irtensor.float()

        scale = self.inputs[0].quantization.scale
        zp = self.inputs[0].quantization.offset
        inp = (inp + zp) / scale

        outputs = fp_reduce(self, inp, **kwargs)
        if method.upper() == 'PROD':
            has_zero = (inp == 0)
            for i in axis:
                has_zero = torch.any(has_zero, i, keepdim=True)
            outputs[has_zero] = 0

        scale = self.outputs[0].quantization.scale
        zp = self.outputs[0].quantization.offset
        if method.upper() in ['ALL', 'ANY']:
            scale = 1
            zp = 0
        outputs = outputs * scale - zp
        info = np.iinfo(self.outputs[0].dtype.np)
        outputs = np.clip(np.round(outputs.cpu()), info.min, info.max)

        self.outputs[0].irtensor = outputs.to(inp.device)
        return self.outputs[0].betensor

    inp = inp.long()

    scale_value = self.params.get('scale_value', default=1)
    shift_value = self.params.get('shift_value', default=0)

    if method.lower() == 'min':
        outp = torch.amin(inp, axis=axis, keepdims=True)
    elif method.lower() == 'max':
        outp = torch.amax(inp, axis=axis, keepdims=True)
    elif method.lower() in ['sum', 'mean']:
        inp = inp.cpu().numpy()
        inp = inp + self.inputs[0].quantization.offset
        outp = np.sum(inp, axis=tuple(axis), keepdims=True)
        outp = np.round(outp * scale_value * 0.5 ** shift_value)
        outp -= self.outputs[0].quantization.offset
        outp = clamp(outp, self.outputs[0].dtype)
    elif method.lower() == 'any':
        inp = inp + self.inputs[0].quantization.offset
        outp = inp.bool()
        for i in self.params['axis']:
            outp = torch.any(outp, i, keepdims=True)
        outp = outp.long()
    elif method.lower() == 'all':
        inp = inp + self.inputs[0].quantization.offset
        outp = inp.bool()
        for i in self.params['axis']:
            outp = torch.all(outp, i, keepdims=True)
        outp = outp.long()
    elif method.lower() == 'prod':
        inp = inp.cpu().numpy()
        inp = inp + self.inputs[0].quantization.offset
        outp = reduce_prod(self, inp)
        outp -= self.outputs[0].quantization.offset
        outp = clamp(outp, self.outputs[0].dtype)
    elif method.lower() in ['variance', 'unbiased_variance']:
        inp = inp.cpu().numpy()
        inp = inp + self.inputs[0].quantization.offset
        outp = reduce_variance(self, inp).astype('int64')
        outp -= self.outputs[0].quantization.offset
        outp = clamp(outp, self.outputs[0].dtype)
    elif method.lower() == 'l1':
        inp = inp.cpu().numpy()
        inp = inp + self.inputs[0].quantization.offset
        outp = reduce_l1(self, inp).astype('int64')
        outp -= self.outputs[0].quantization.offset
        outp = clamp(outp, self.outputs[0].dtype)
    elif method.lower() == 'l2':
        inp = inp.cpu().numpy()
        inp = inp + self.inputs[0].quantization.offset
        outp = reduce_l2(self, inp)
        # outp -= self.outputs[0].quantization.offset
        outp = clamp(outp, self.outputs[0].dtype)
    else:
        raise TypeError('do not support method = %s' % self.method)

    self.outputs[0].irtensor = outp
    return self.outputs[0].betensor


def reduce_prod(self, input):
    scale_value = self.params.get('scale_value', default=1)
    shift_value = self.params.get('shift_value', default=0)
    axis = self.params['axis']
    inp_signed = self.inputs[0].dtype.is_signed
    threshold_min, threshold_max = bits2range(16, inp_signed)
    inp_bak = input.copy()
    input_dim = input.ndim  # 4
    input_shape = list(input.shape)
    outp = None
    step = 1
    current_transpose_shape = []
    for ax in range(input_dim):
        if ax in axis:
            step *= input_shape[ax]
            input_shape[ax] = 1
        else:
            current_transpose_shape.append(ax)
    # shape[dim1,axis1,axis2,dim4] => shape[dim1,dim4,axis1,axis2]
    current_transpose_shape += axis
    inp_bak_transpose = inp_bak.transpose(current_transpose_shape).reshape(-1)
    outer_step = inp_bak.size // step
    outp_iter = np.zeros(shape=(outer_step), dtype='int64')
    for i in range(outer_step):
        start_idx = i * step
        end_idx = i * step + step
        tmp_data = 1
        left_shifts = 0
        # (Q1) * (Q2) => Q16 * (2 ** left_shift)
        for j in range(start_idx, end_idx):
            tmp_data *= inp_bak_transpose[j]
            tmp_data, left_shift = convert_less_mbit(
                self, tmp_data, threshold_min, threshold_max)
            left_shifts += left_shift
        shift_diff = min(max(shift_value - left_shifts, -32), 32)
        tmp_data = np.round(tmp_data * scale_value * 0.5 ** shift_diff)
        tmp_data = np.clip(tmp_data, -2 ** 31, 2 ** 31 - 1)
        outp_iter[i] = tmp_data
    outp = outp_iter.reshape(input_shape)
    return outp


def convert_less_mbit(self, data, threshold_min, threshold_max):
    left_shift = 0
    while (data > threshold_max) or (data < threshold_min):
        data = data >> 1
        left_shift += 1
    return data, left_shift


def reduce_variance(self, input):
    scale_value = self.params.get('scale_value', default=1)
    shift_value = self.params.get('shift_value', default=0)
    axis = self.params['axis']
    sum = np.sum(input, axis=tuple(axis), keepdims=True)
    num = 1
    for t in axis:
        num *= input.shape[t]
    mean = (sum / num).astype(input.dtype)
    tmp_square = (input - mean) ** 2
    tmp_square = np.clip(tmp_square, 0, 2**31 - 1)
    tmp_sum = np.sum(tmp_square, axis=tuple(axis), keepdims=True)
    tmp_sum = np.clip(tmp_sum, 0, 2**31 - 1)
    outp = np.round(tmp_sum * scale_value * (0.5 ** shift_value))
    return outp


def reduce_l1(self, input):
    axis = self.params['axis']
    scale_value = self.params.get('scale_value', default=1)
    shift_value = self.params.get('shift_value', default=0)
    input = np.abs(input)
    act_min, act_max = 0, 2 ** 32 - 1
    tmp_sum = np.sum(input, axis=tuple(axis), keepdims=True)
    tmp_sum = np.clip(tmp_sum, act_min, act_max)
    outp = np.round(tmp_sum * scale_value * 0.5 ** shift_value)
    return outp


def reduce_l2(self, input):
    axis = self.params['axis']
    scale_value = self.params.get('scale_value', default=1)
    shift_value = self.params.get('shift_value', default=0)
    sqrt_lut = self.constants['sqrt_lut'].betensor.cpu().numpy()
    power2_input = input * input
    sum_output = np.sum(power2_input, tuple(axis), keepdims=True)
    shape = sum_output.shape
    sum_output = clamp(sum_output, 'int32')
    sum_output = sum_output.astype(np.int64)
    # support shift < 32
    sum_output = np.round(sum_output * scale_value * 0.5 ** shift_value)
    sum_output = clamp(sum_output, 'uint16')

    sum_output_f = np.reshape(sum_output, (-1,)).astype('int32')

    sqrt = _tpc_lut(sum_output_f, sqrt_lut, 16, False)
    if '8' in str(self.outputs[0].betensor.dtype):
        out_qmin, out_qmax = bits2range(8, False)
    else:
        out_qmin, out_qmax = bits2range(16, False)
    sqrt = torch.clamp(sqrt, out_qmin, out_qmax)
    return sqrt.reshape(shape)


def fp_reduce(self, inp, **kwargs):
    target = Target()
    is_aiff = kwargs.get("is_aiff", False)
    handle_fp16_inf = False
    if is_aiff and target.name.startswith("X3"):
        handle_fp16_inf = True
    if handle_fp16_inf:
        inp = x3_aiff_fp32_2_fpx(inp, fpx=self.outputs[0].dtype).to(torch.float32)
    method = self.params['method'].upper()
    axis = self.params['axis']
    outp = torch.ones_like(self.outputs[0].betensor)
    if method == 'MIN':
        outp = torch.amin(inp, axis, keepdims=True)
    elif method == 'MAX':
        outp = torch.amax(inp, axis, keepdims=True)
    elif method == 'ANY':
        outp = inp.bool()
        for i in axis:
            outp = torch.any(outp, i, keepdims=True)
        outp = outp.long()
    elif method == 'ALL':
        outp = inp.bool()
        for i in axis:
            outp = torch.all(outp, i, keepdims=True)
        # output only is 0 or 1
        outp = outp.long()
    elif method == 'SUM':
        outp = torch.sum(inp, axis, keepdims=True)
        if handle_fp16_inf:
            outp = x3_aiff_fp32_2_fpx(outp, fpx=self.outputs[0].dtype)
    elif method == 'MEAN':
        inp = inp.float()
        outp = torch.sum(inp, axis, keepdims=True)
        if handle_fp16_inf:
            outp = x3_aiff_fp32_2_fpx(outp, fpx=self.outputs[0].dtype)
        inshape = inp.numel()
        outshape = outp.numel()
        div_dim = inshape / outshape
        outp = outp / div_dim
        # outp = torch.mean(inp, axis, keepdims=True)
    elif method == 'PROD':
        outp = inp.float()
        for i in axis:
            outp = torch.prod(outp, i, keepdim=True)
        if handle_fp16_inf:
            outp = x3_aiff_fp32_2_fpx(outp, fpx=self.outputs[0].dtype)

    elif method in ['VARIANCE', 'UNBIASED_VARIANCE']:
        input_data = inp.float()
        t_mean = torch.mean(input_data, dim=axis, keepdim=True)
        tmp_square = (input_data - t_mean) ** 2
        tmp_sum = torch.sum(tmp_square, dim=axis, keepdim=True)
        num = 1
        if handle_fp16_inf:
            tmp_sum = x3_aiff_fp32_2_fpx(tmp_sum, fpx=self.outputs[0].dtype)
        for t in axis:
            num *= inp.shape[t]
        if method == "UNBIASED_VARIANCE":
            num = num - 1
        outp = tmp_sum / num
    elif method == 'L1':
        inp = inp.float()
        abs_tensor = torch.abs(inp)
        outp = torch.sum(abs_tensor, axis, keepdim=True)
        if handle_fp16_inf:
            outp = x3_aiff_fp32_2_fpx(outp, fpx=self.outputs[0].dtype)

    elif method == 'L2':
        inp = inp.float()
        power2_input = inp * inp
        sum_output = torch.sum(power2_input, axis, keepdim=True)
        if handle_fp16_inf:
            sum_output = x3_aiff_fp32_2_fpx(sum_output, fpx=self.outputs[0].dtype)
        outp = torch.sqrt(sum_output)
    else:
        raise ValueError("Reduce method is invalid")
    return outp


@register_gen_weight(OpType.Reduce)
def gen_random_weight(self):
    if 'sqrt_lut' in self.constants:  # for L2
        sqrt_lut = np.random.randint(0, 255, 256, dtype=np.uint8)
        sqrt_lut = np.sort(sqrt_lut)
        self.constants['sqrt_lut'].betensor = np.asarray(sqrt_lut, dtype=str(self.constants['sqrt_lut'].dtype))


@register_gen_input(OpType.Reduce)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_type0 = str(self.inputs[0].dtype)
    if self.inputs[0].dtype.is_float:
        input0 = (np.random.random(in_shape0) * 1000).astype(in_type0)
        self.inputs[0].betensor = input0
        return [self.inputs[0].betensor]
    if self.params['method'].upper() == 'PROD':
        min_v = -10 if in_type0.startswith('int') else 0
        max_v = 10
    else:
        min_v = np.iinfo(in_type0).min
        max_v = np.iinfo(in_type0).max

    input0 = np.random.randint(min_v, max_v, size=in_shape0, dtype=in_type0)
    if self.params['method'].upper() == 'PROD':
        input0[input0 == 0] = 1  # no 0

    input_data = [input0]
    self.inputs[0].betensor = input_data[0]
    return input_data


@register_op_metric(OpType.Reduce)
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
