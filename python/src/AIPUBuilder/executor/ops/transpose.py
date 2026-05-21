import torch
from ..engine import register_gt
from .gt_util.quant_utils import clamp
from AIPUBuilder.core import OpType


@register_gt(OpType.Transpose)
def transpose(self, *args, **kwargs):
    input_tensor = self.inputs[0].irtensor
    output = input_tensor.permute(self.params['perm'])
    self.outputs[0].irtensor = output
    return self.outputs[0].betensor
