import torch
from .gt_util import *
from ..engine import register_gt
from AIPUBuilder.core import OpType


@register_gt(OpType.Sign)
def Sign(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float:
        output = sign_float(self)
    else:
        input = self.inputs[0].betensor.cpu().numpy().astype(np.int64)
        input = input + self.inputs[0].quantization.offset
        output = np.clip(input, -1, 1).astype(str(self.outputs[0].dtype))

    self.outputs[0].betensor = output
    return [o.betensor for o in self.outputs]


def sign_float(self):
    input = torch_to_numpy(self.inputs[0].betensor, self.inputs[0].dtype.np)
    out = np.sign(input).astype(self.outputs[0].dtype.np)
    return out
