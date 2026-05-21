from ..engine import register_gt
from .gt_util.layout_transfer import layout_convert
from AIPUBuilder.core import OpType
import torch


@register_gt(OpType.SpaceToDepth)
def spacetodepth(self, *args, **kwargs):
    inp = self.inputs[0].betensor
    if 'block_size' in self.params.keys():
        bs = self.params['block_size']
    else:
        bs = self.params['block_size_x']
    inp = layout_convert(inp, 'NHWC', 'NCHW')
    o = space_to_depth(inp, bs)
    o = layout_convert(o, 'NCHW', 'NHWC')
    self.outputs[0].betensor = o

    return self.outputs[0].betensor


def space_to_depth(x, bs):
    N, C, H, W = x.size()
    x = x.view(N, C, H // bs, bs, W // bs, bs)       # (N, C, H//bs, bs, W//bs, bs)
    x = x.permute(0, 3, 5, 1, 2, 4).contiguous()     # (N, bs, bs, C, H//bs, W//bs)
    x = x.view(N, C * (bs ** 2), H // bs, W // bs)   # (N, C*bs^2, H//bs, W//bs)
    return x
