import torch
from ..engine import register_gt
from AIPUBuilder.core import OpType


@register_gt(OpType.BatchToSpaceND)
def BatchToSpaceND(self, *args, **kwargs):
    block_size = self.params['block_size']
    crops = self.params['crops']
    inp = self.inputs[0].betensor
    bs_z, bs_y, bs_x = block_size

    n, d, h, w, c = inp.shape
    y = inp.view(bs_z, bs_y, bs_x, n // (bs_z * bs_y * bs_x), d, h, w, c)  # ndhwc ->
    y = y.permute(3, 4, 0, 5, 1, 6, 2, 7).contiguous()  # n//(bs_z*bs_y*bs_x), d, bs_z, h, bs_y, w, bs_x, c
    y = y.view(n // (bs_z * bs_y * bs_x), d * bs_z, h * bs_y, w * bs_x, c)
    out = y[:, crops[0][0]: d * bs_z - crops[0][1], crops[1][0]: h * bs_y - crops[1][1],
            crops[2][0]: w * bs_x - crops[2][1], :]

    self.outputs[0].betensor = out
    return [o.betensor for o in self.outputs]
