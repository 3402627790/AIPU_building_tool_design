import torch
from .gt_util import *
from ..engine import register_gt
from AIPUBuilder.core import OpType


@register_gt(OpType.BatchToDepth)
def batchtodepth(self, *args, **kwargs):
    block_size_ = self.params['block_size']
    inp = self.inputs[0].betensor
    out = self.outputs[0].betensor
    n, h, w, c = inp.shape
    y = inp.view(n // block_size_, block_size_, h, w, c).permute(0, 2, 3, 1, 4).contiguous()
    out = y.view(n // block_size_, h, w, c * block_size_)
    self.outputs[0].betensor = out
    return out

