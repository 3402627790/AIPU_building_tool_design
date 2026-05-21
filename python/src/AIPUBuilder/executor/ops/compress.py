import torch
import numpy as np
from ..engine import register_gt, register_gen_input, register_op_metric
from .gt_util.metric import relative_diff, ulp_diff, cos_distance, check_fp_data
from .gt_util.dtype_utils import torch_to_numpy
from AIPUBuilder.core import OpType, Dtype


@register_gt(OpType.Compress)
def compress(self, *args, **kwargs):
    input = torch_to_numpy(self.inputs[0].betensor, self.inputs[0].dtype.np)
    condition = self.inputs[1].betensor.cpu().numpy().astype(self.inputs[1].dtype.np)
    axis = self.params['axis'] if 'axis' in self.params else None
    output = np.compress(condition, input, axis=axis)
    output = output.astype(self.outputs[0].dtype.np)
    # layout = self.outputs[0].shape.layout
    # self.outputs[0].set_numpy(output)  # output.shape could != ir.shape
    # self.outputs[0].shape.layout = layout  # restore layout after set_numpy
    self.outputs[0].betensor = output  # set_numpy() can not change betensor after aipugb


@register_gen_input(OpType.Compress)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)

    if self.inputs[0].dtype.is_float:
        input0 = np.random.uniform(-10000, 10000, size=in_shape0).astype(in_type0)
    else:
        input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)
    condition = np.random.randint(0, 2, size=in_shape1, dtype=in_type1)
    input_data = [input0, condition]

    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    return input_data


@register_op_metric(OpType.Compress)
@check_fp_data
def metric(self, outputs, index=None):
    def call(r, t):
        return np.all(r == 0)

    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]

    layer_dtypes = [str(o.dtype) for o in self.outputs] if index is None else [str(self.outputs[index].dtype)]

    if groundtruths[0].size == 0:
        print('the size of gt is 0 !')
        return True

    rel_diffs = relative_diff(outputs, groundtruths)
    return all(call(r, t) for r, t in enumerate(zip(rel_diffs, layer_dtypes)))
