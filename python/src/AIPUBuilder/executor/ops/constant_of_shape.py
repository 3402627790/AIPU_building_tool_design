import torch
from ..engine import register_gt
from AIPUBuilder.core import OpType


@register_gt(OpType.ConstantOfShape)
def constant_of_shape(self, *args, **kwargs):
    shape = self.inputs[0].betensor.to(torch.int)
    value = self.params.get("value", 0)
    output = torch.zeros(tuple(shape))
    output += value
    self.outputs[0].betensor = output

    return output
