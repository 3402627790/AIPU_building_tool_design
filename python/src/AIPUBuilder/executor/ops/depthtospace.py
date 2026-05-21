from ..engine import register_gt
from AIPUBuilder.core import OpType
import torch


@register_gt(OpType.DepthToSpace)
def depthtospace(self, *args, **kwargs):
    inp = self.inputs[0].betensor
    block_size_y = self.params['block_size_y']
    block_size_x = self.params['block_size_x']

    inp = self.inputs[0].irtensor
    N, H, W, C = inp.shape
    new_channel = C // (block_size_x * block_size_y)
    if 'mode' not in self.params.keys() or self.params['mode'] == 'DCR':
        x = inp.view(N, H, W, block_size_y, block_size_x, new_channel)  # (N, H, W, bs, bs, C//bs^2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (N, H, bs, W, bs, C//bs^2)
    elif self.params['mode'] == 'CRD':
        x = inp.view(N, H, W, new_channel, block_size_y, block_size_x)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()  # (N, H, bs, W, bs, C//bs^2)
    else:
        raise ValueError("unsupported mode: %s for DepthToSpace" % self.params['mode'])
    out = x.view(N, H * block_size_y, W * block_size_x, new_channel)
    self.outputs[0].irtensor = out

    return self.outputs[0].betensor
