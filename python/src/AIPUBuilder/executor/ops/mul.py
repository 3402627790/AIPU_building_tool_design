from ..engine import register_gt, register_gen_input, _OP_METRICS, register_op_metric
from AIPUBuilder.core import OpType, Dtype
import numpy as np
import torch
from .gt_util import clip_to_bit
from .gt_util.aiff import _aiff_inner_activation, _aiff_final_output
from .gt_util.quant_utils import clamp
from .gt_util.metric import relative_diff, ulp_diff, cos_distance, check_fp_data
from . import eltwise
from .gt_util.dtype_utils import torch_to_numpy, numpy_to_torch


@register_gt(OpType.Mul)
def mul(self, *args, **kwargs):
    self.params['method'] = 'MUL'
    self.attrs["is_aiff"] = kwargs.get("is_aiff", False)  # for results compare with lib
    output = eltwise(self, *args, **kwargs)
    self.params.pop('method')
    return output


@register_gen_input(OpType.Mul)
def gen_input_data(self):
    if self.inputs[0].dtype.is_int:
        in_dtype = str(self.inputs[0].dtype)
        in_shape = list(self.inputs[0].shape)
        self.inputs[0].betensor = np.random.randint(np.iinfo(in_dtype).min,
                                                    np.iinfo(in_dtype).max,
                                                    size=in_shape,
                                                    dtype=in_dtype)
        in_dtype = str(self.inputs[1].dtype)
        in_shape = list(self.inputs[1].shape)
        self.inputs[1].betensor = np.random.randint(np.iinfo(in_dtype).min,
                                                    np.iinfo(in_dtype).max,
                                                    size=in_shape,
                                                    dtype=in_dtype)

    else:
        # input_data = np.random.randn(*in_shape) * 1000
        self.inputs[0].betensor = np.random.uniform(-100, 100,
                                                    size=list(self.inputs[0].shape)).astype(self.inputs[0].dtype.np)
        self.inputs[1].betensor = np.random.uniform(-100, 100,
                                                    size=list(self.inputs[1].shape)).astype(self.inputs[1].dtype.np)


@register_op_metric(OpType.Mul)
@check_fp_data
def metric(self, outputs, index=None):
    def call(r, u, c, t):
        if "float" in t:
            if np.all(u <= 1):
                return True
            return np.all(r < 0.001) and np.all(u <= 4)
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
