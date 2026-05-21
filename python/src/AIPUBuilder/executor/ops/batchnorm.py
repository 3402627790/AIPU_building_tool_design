from ..engine import register_gt, _OP_METRICS, register_op_metric
from AIPUBuilder.core import OpType, DataLayout
import torch
import numpy as np
from .gt_util.aiff import _aiff_merge_shifts, _aiff_final_output
from .gt_util.aiff import clip_to_bit, x3_aiff_fp32_2_fpx
from .gt_util.quant_utils import clamp
from .gt_util.math_utils import fp32_to_fp24
from .gt_util.metric import relative_diff, ulp_diff, cos_distance, check_fp_data
from .gt_util.dtype_utils import torch_to_numpy
from .gt_util import check_in_datalayout


@register_gt(OpType.BatchNorm)
def batchnorm(self, *args, **kwargs):
    fuse = True
    if not ('fuse' in kwargs and kwargs['fuse'] and
            (OpType.Eltwise in [c.type for c in self.children] or
                OpType.Activation in [c.type for c in self.children])):
        fuse = False
    is_aiff = True
    if 'is_aiff' in kwargs.keys():
        is_aiff = kwargs['is_aiff']

    if self.inputs[0].dtype.is_float:
        return fp_bn(self, fuse=fuse, is_aiff=is_aiff)
    else:
        self.sgattrs = kwargs['sgattrs'] if 'sgattrs' in kwargs else None
        if self.sgattrs is None:
            self.sgattrs = dict()

        if self.sgattrs is not None and 'bn_32bit_saturation' in self.sgattrs:
            clip = True
        else:
            clip = False
        input_tensor = self.inputs[0].betensor.long()
        weights = self.constants['weights'].betensor.long()
        biases = self.constants['biases'].betensor.type(self.constants['biases'].betensor_type)
        input_tensor = self.inputs[0].irtensor
        if 'scale' in self.constants.keys() and 'shift' in self.constants.keys():
            scale_value, shift_value = _aiff_merge_shifts(
                self.constants['scale'].betensor.long(), self.constants['shift'].betensor.long())
        else:
            scale_value = self.params['scale_value']
            shift_value = self.params['shift_value']
        # output_type = str(self.outputs[0].dtype)
        is_32bit_bias = True if self.sgattrs is not None and 'bn_32bit_bias' in self.sgattrs else False
        biases = bn_compression_bias(biases, 16, is_32bit_bias)  # hardware only support int16 bias
        outputs = input_tensor.type(torch.int64)
        axis = self.params['axis']
        if axis < 0:
            axis += input_tensor.ndim
        perm = []
        if axis != input_tensor.ndim - 1:
            orig_perm = [p for p in range(input_tensor.ndim)]
            perm = orig_perm[:axis] + orig_perm[axis+1:] + [orig_perm[axis]]
            input_tensor = torch.permute(input_tensor, perm)
        outputs = input_tensor.type(torch.double) * weights.type(torch.double) + biases
        if clip:
            outputs = torch.clip(outputs, torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max)
        outputs = _aiff_final_output(self, outputs.type(torch.int64), scale_value, shift_value, fuse=fuse)
        outputs = clip_to_bit(outputs, 32, True)
        if not fuse:
            outputs = clamp(outputs, self.outputs[0].dtype)
        if len(perm):
            orig_perm = [p for p in range(input_tensor.ndim)]
            n_perm = orig_perm[:axis] + [orig_perm[-1]] + orig_perm[axis:-1]
            outputs = torch.permute(outputs, n_perm)
        self.outputs[0].irtensor = outputs
        return self.outputs[0].betensor


def fp_bn(self, fuse=False, is_aiff=False):
    input_tensor = self.inputs[0].irtensor.to(torch.float32)
    weights = self.constants['weights'].betensor.to(torch.float32)
    biases = self.constants['biases'].betensor.to(torch.float32)
    axis = self.params['axis']

    outputs = input_tensor
    if axis != input_tensor.ndim - 1:
        if input_tensor.ndim == 5:
            [n, d, h, w, c] = input_tensor.shape
            input_tensor = input_tensor.reshape(n * d, h, w, c)
            axis = axis - 1
        weight_shape = [input_tensor.shape[axis]] + [1]*(input_tensor.ndim-1-axis)
        weights = weights.reshape(weight_shape)
        biases = biases.reshape(weight_shape)
    outputs = fp32_to_fp24(input_tensor * weights) if is_aiff else (input_tensor * weights)
    outputs += biases
    outputs = x3_aiff_fp32_2_fpx(outputs, is_aiff, fpx=self.outputs[0].dtype) if not fuse else outputs

    self.outputs[0].irtensor = outputs
    return self.outputs[0].betensor


def bn_compression_bias(a, max_shift=16, is_32bit_bias=False):
    b = a
    if is_32bit_bias:
        b = torch.clamp(b, torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max)
    elif max(b) > 32767 or min(b) < -32768:
        a_max = max(abs(b.long()))
        shift = torch.ceil(torch.log2(a_max.double() + 0.5)).to(torch.int32)
        shift -= 15
        if shift <= 0:
            return a
        elif shift > max_shift:
            shift = max_shift
        b = torch.div(b, (2 ** shift), rounding_mode='floor')  # floor
        b = torch.clamp(b, torch.iinfo(torch.int16).min, torch.iinfo(torch.int16).max)
        b = b * (2 ** shift)

    return b.long()


@register_op_metric(OpType.BatchNorm)
@check_fp_data
def metric(self, outputs, index=None):
    def call(r, u, c, t):
        if "float" in t:
            if np.all(u <= 1):
                return True
            return np.all(r < 0.0005)
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
