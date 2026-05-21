import torch
from .gt_util import *
from ..engine import register_gt
from AIPUBuilder.core import OpType


@register_gt(OpType.DataStride)
def DataStride(self, *args, **kwargs):
    kernel_x = self.params['kernel_x']
    kernel_y = self.params['kernel_y']
    stride_x = self.params['stride_x']
    stride_y = self.params['stride_y']

    input = self.inputs[0].irtensor.cpu()
    input_shape = input.shape

    output = []
    for j in range(0, input_shape[1] - kernel_y + 1, stride_y):
        tmp_line = []
        for k in range(0, input_shape[2] - kernel_x + 1, stride_x):
            tmp_line.append(input[:, j:j + kernel_y, k:k + kernel_x, :])
        tmp_line = torch.cat(tmp_line, dim=2)
        output.append(tmp_line)
    output = torch.cat(output, dim=1)

    self.outputs[0].irtensor = output
    return [o.betensor for o in self.outputs]
