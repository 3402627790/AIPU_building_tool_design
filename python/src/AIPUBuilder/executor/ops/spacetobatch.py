import torch
from .gt_util import *
from ..engine import register_gt
from AIPUBuilder.core import OpType


@register_gt(OpType.SpaceToBatch)
def SpaceToBatch(self, *args, **kwargs):
    block_size_x = self.params['block_size_x']
    block_size_y = self.params['block_size_y']
    pad_left = self.params['pad_left']
    pad_right = self.params['pad_right']
    pad_top = self.params['pad_top']
    pad_bottom = self.params['pad_bottom']

    inp = self.inputs[0].irtensor
    inp = inp.double()
    paddings = (0, 0, pad_left, pad_right, pad_top, pad_bottom)
    pad_value = -self.inputs[0].quantization.offset  # pad value will be converted to inp dtype
    y = torch.nn.functional.pad(inp, paddings, value=pad_value)
    n, h, w, c = y.shape

    y = y.view(n, h // block_size_y, block_size_y, w // block_size_x, block_size_x, c)  # nhwc ->
    y = y.permute(2, 4, 0, 1, 3, 5).contiguous()  # n, block_size_y, block_size_x, h//block_size_y, w//block_size_x, c
    out = y.view(n * block_size_x * block_size_y, h // block_size_y, w // block_size_x, c)

    self.outputs[0].irtensor = out
    return [o.betensor for o in self.outputs]
