from ..engine import register_gt, register_gen_input
from AIPUBuilder.core import OpType
import torch
import numpy as np


@register_gt(OpType.IsNaN)
def isnan(self, *args, **kwargs):

    input0 = self.inputs[0].irtensor
    output = torch.isnan(input0)

    output = output + 0  # bool -> 0,1
    output[output != 0] = 1
    output = output.to(self.outputs[0].dtype.torch)
    self.outputs[0].irtensor = output

    return self.outputs[0].betensor

@register_gen_input(OpType.IsNaN)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_type0 = self.inputs[0].dtype

    input0 = np.random.uniform(-1, 1, size=in_shape0).astype(in_type0.np)

    total_elements = input0.size
    num_nan = max(1, int(input0.size * 0.1))
    random_indices = np.random.choice(total_elements, size=num_nan, replace=False)
    input0.flat[random_indices] = np.nan
    # input0 = input0.reshape(in_shape0)

    input_data = [input0]
    self.inputs[0].betensor = input_data[0]
    return input_data
