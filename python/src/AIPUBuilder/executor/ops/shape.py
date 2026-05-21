import torch
from ..engine import register_gt
from AIPUBuilder.core import OpType


@register_gt(OpType.Shape)
def shape(self, *args, **kwargs):
    input_shape = self.inputs[0].betensor.shape
    dim = self.inputs[0].betensor.dim()
    start = self.params.get("start", 0)
    end = self.params.get("end", dim)
    start = start if start >= 0 else start + dim
    end = end if end >= 0 else end + dim
    output_shape = list(input_shape)
    output = torch.Tensor(output_shape[start:end])
    self.outputs[0].betensor = output

    return output
