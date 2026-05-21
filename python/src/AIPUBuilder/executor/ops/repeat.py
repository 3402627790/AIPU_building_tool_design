from ..engine import register_gt, register_gen_input
from AIPUBuilder.core import Target
from .gt_util.quant_utils import clamp, _tpc_lut
from .gt_util.dtype_utils import torch_to_numpy
from AIPUBuilder.core import OpType
import numpy as np
import torch
from functools import reduce


@register_gt(OpType.Repeat)
def repeat(self, *args, **kwargs):
    repeats = self.inputs[1].betensor.cpu().numpy().astype('int64')
    inp = torch_to_numpy(self.inputs[0].betensor, self.inputs[0].dtype.np)
    axis = self.params.get("axis", None)
    if not axis is None:
        output = np.repeat(inp, repeats=repeats, axis=axis)
        if output.shape[axis] < self.outputs[0].shape[axis]:
            axis_ = axis
            if axis == -1:
                axis = axis + inp.ndim
            pads = []
            for i in range(inp.ndim):
                if axis == i:
                    pad_value = [0, self.outputs[0].shape[axis] - output.shape[axis]]
                else:
                    pad_value = [0, 0]
                pads.append(pad_value)
            output = np.pad(output, pads, mode='constant')
    else:
        output = np.repeat(inp, repeats=repeats)  # axis=None
        layer_top_size = reduce(lambda x, y: x * y, self.outputs[0].shape)
        if layer_top_size > output.shape[0]:
            output = np.pad(output, (0, layer_top_size - output.shape[0]), mode='constant')
    layout = self.outputs[0].shape.layout
    output = output.astype(self.outputs[0].dtype.np)  # the output's data type should be align to ir before set_numpy()
    # self.outputs[0].set_numpy(output)  # output.shape could != ir.shape
    # self.outputs[0].shape.layout = layout
    self.outputs[0].betensor = output  # set_numpy() can not change betensor after aipugb
    pass


@register_gen_input(OpType.Repeat)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)

    max_repeat_times = 50  # modify me
    if self.inputs[0].dtype.is_float:
        input0 = np.random.uniform(-1000, 1000, size=in_shape0).astype(in_type0)
    else:
        input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)
    input1 = np.random.randint(1, max_repeat_times, size=in_shape1, dtype=in_type1)

    input_data = [input0, input1]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    return input_data
