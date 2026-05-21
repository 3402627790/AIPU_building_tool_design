from ..engine import register_gt
from .gt_util.aiff import clip_to_bit
from AIPUBuilder.core import OpType
import torch
import numpy as np


@register_gt(OpType.ChannelShuffle)
def channelshuffle(self, *args, **kwargs):
    group = self.params['group']
    output_shape = list(self.outputs[0].shape)
    input_shape = list(self.inputs[0].shape)
    # assert output_shape[3] == group
    trans_perm = list(range(len(input_shape) + 1))
    trans_perm[-1], trans_perm[-2] = trans_perm[-2], trans_perm[-1]   # [0, 1, 2, 4, 3]
    new_shape = input_shape.copy()[:len(input_shape) - 1]
    new_shape += [group, input_shape[-1] // group]  # [is[0], is[1], is[2], group, is[3]/g]

    output = self.inputs[0].betensor.reshape(new_shape)
    output = output.permute(trans_perm)
    output = output.reshape(input_shape)
    output = torch.split(output, output.shape[-1] // self.params['splits'], dim=-1)

    for i in range(len(output)):
        self.outputs[i].betensor = output[i]

    return [o.betensor for o in self.outputs]
