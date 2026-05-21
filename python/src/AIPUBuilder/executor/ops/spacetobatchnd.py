import torch
from .gt_util import *
from ..engine import register_gt
from AIPUBuilder.core import OpType


@register_gt(OpType.SpaceToBatchND)
def SpaceToBatchND(self, *args, **kwargs):
    block_size = self.params['block_size']
    pads = self.params['pads']

    inp = self.inputs[0].betensor
    bs_z, bs_y, bs_x = block_size

    inp = inp.permute([0, 4, 1, 2, 3])   # use pad
    paddings = (pads[2][0], pads[2][1], pads[1][0], pads[1][1], pads[0][0], pads[0][1])
    pad_value = -self.inputs[0].quantization.offset
    y = torch.nn.functional.pad(inp, paddings, value=pad_value)
    y = y.permute([0, 2, 3, 4, 1])
    n, d, h, w, c = y.shape

    y = y.view(n, d // bs_z, bs_z, h // bs_y, bs_y, w // bs_x, bs_x, c)  # ndhwc ->
    y = y.permute(2, 4, 6, 0, 1, 3, 5, 7).contiguous()  # n, bs_z, bs_y, bs_x, d//bs_z, h//bs_y, w//bs_x, c
    out = y.view(n * bs_z * bs_y * bs_x, d // bs_z, h // bs_y, w // bs_x, c)

    self.outputs[0].betensor = out
    return [o.betensor for o in self.outputs]
