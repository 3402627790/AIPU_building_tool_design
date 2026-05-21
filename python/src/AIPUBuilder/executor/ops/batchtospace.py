import torch
from .gt_util import *
from ..engine import register_gt
from AIPUBuilder.core import OpType


@register_gt(OpType.BatchToSpace)
def BatchToSpace(self, *args, **kwargs):
    block_size_x = self.params['block_size_x']
    block_size_y = self.params['block_size_y']
    crop_left = self.params['crop_left']
    crop_right = self.params['crop_right']
    crop_top = self.params['crop_top']
    crop_bottom = self.params['crop_bottom']

    inp = self.inputs[0].irtensor

    n, h, w, c = inp.shape
    y = inp.view(block_size_y, block_size_x, n // (block_size_x * block_size_y), h, w, c)  # nhwc ->
    y = y.permute(2, 3, 0, 4, 1, 5).contiguous()  # n//(block_size_x*block_size_y), h, block_size_y, w, block_size_x, c
    y = y.view(n // (block_size_x * block_size_y), h * block_size_y, w * block_size_x, c)
    out = y[:, crop_top: h * block_size_y - crop_bottom, crop_left: w * block_size_x - crop_right, :]

    self.outputs[0].irtensor = out
    return [o.betensor for o in self.outputs]
