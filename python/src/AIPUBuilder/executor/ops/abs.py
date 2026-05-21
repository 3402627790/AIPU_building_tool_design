import torch
import numpy as np
from ..engine import register_gt
from AIPUBuilder.core import OpType, Dtype
from .gt_util.quant_utils import clamp


@register_gt(OpType.Abs)
def Abs(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float:
        output = torch.abs(self.inputs[0].betensor).to(self.inputs[0].dtype.torch)
    else:
        input_data = self.inputs[0].betensor.long()
        input_data = input_data + self.inputs[0].quantization.offset
        output = torch.abs(input_data)
        output -= self.outputs[0].quantization.offset
        output = clamp(output, self.outputs[0].dtype)
    self.outputs[0].betensor = output
