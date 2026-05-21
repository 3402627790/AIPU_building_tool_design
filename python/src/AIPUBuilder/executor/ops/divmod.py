from ..engine import register_gt
from AIPUBuilder.core import OpType
import numpy as np
import torch
from AIPUBuilder.logger import FATAL
from .gt_util.math_utils import broadcasting_transform


@register_gt(OpType.DivMod)
def DivMod(self, *args, **kwargs):
    input0 = self.inputs[0].betensor.long()
    input1 = self.inputs[1].betensor.long()
    if input1.numel() == 1:
        input1 = torch.zeros_like(input0) + input1

    input0, input1 = broadcasting_transform(input0, input1)

    # output0, output1 = np.divmod(input0, input1)
    mode = self.params.get('mode', 'floor').lower()
    if mode not in ['floor', 'trunc']:
        FATAL("currently the op{%s} only supprot mode[floor,trunc], but now mode is {%s}." % (self.type, mode))

    out0 = torch.zeros_like(input0, device=input0.device)
    out1 = torch.zeros_like(input0, device=input0.device)

    nonzeros_mask = input1 != 0
    out0[nonzeros_mask] = torch.div(input0[nonzeros_mask], input1[nonzeros_mask], rounding_mode=mode)
    out1[nonzeros_mask] = input0[nonzeros_mask] - input1[nonzeros_mask] * out0[nonzeros_mask]

    if self.outputs[0].dtype.is_signed:
        max_v = 2 ** (self.outputs[0].dtype.bits - 1)
        out0[out0 == max_v] = -max_v  # -128/-1=128 -> -128, CTL-7226
    # out0 = clamp(out0, self.outputs[0].dtype)
    self.outputs[0].betensor = out0
    self.outputs[1].betensor = out1
    return [o.betensor for o in self.outputs]
