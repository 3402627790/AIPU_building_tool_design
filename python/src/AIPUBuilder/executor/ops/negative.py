from ..engine import register_gt
from .gt_util.quant_utils import clamp
from AIPUBuilder.core import OpType
import torch
import numpy as np


@register_gt(OpType.Negative)
def negative(self, *args, **kwargs):
    input_tensor = self.inputs[0].betensor
    if not self.inputs[0].dtype.is_float:
        input_tensor = input_tensor.long()
        input_tensor = input_tensor + self.inputs[0].quantization.offset
        output = torch.negative(input_tensor)
        output -= self.outputs[0].quantization.offset
        output = clamp(output, self.outputs[0].dtype)
    else:
        output = torch.negative(input_tensor)
    self.outputs[0].betensor = output
    return self.outputs[0].betensor
