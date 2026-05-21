from ..engine import register_gt
from AIPUBuilder.core import OpType
import numpy as np
import torch


@register_gt(OpType.DepthToBatch)
def DepthToBatch(self, *args, **kwargs):
    inp = self.inputs[0].betensor
    block_size = self.params['block_size']
    in_sh = self.inputs[0].shape
    ou_sh = self.outputs[0].shape
    bk_h = int(in_sh[-1] / ou_sh[-1])
    out = torch.zeros(list(ou_sh))

    for j in range(int(in_sh[0])):
        for i in range(bk_h):
            out[j * bk_h + i:(j + 1) * bk_h + i, :, :, ] = inp[j:j + 1, :, :, i * ou_sh[3]:(i + 1) * ou_sh[3]]

    self.outputs[0].betensor = out
    return self.outputs[0].betensor
