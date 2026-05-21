import torch
import numpy as np
from ..engine import register_gt
from AIPUBuilder.core import OpType, Dtype
from .gt_util.quant_utils import clamp


@register_gt(OpType.Bitwise)
def bitwise(self, *args, **kwargs):
    func_args = []
    input0 = self.inputs[0].betensor.cpu().numpy().astype(str(self.inputs[0].dtype))
    func_args.append(input0)
    if len(self.inputs) > 1:
        func_args.append(self.inputs[1].betensor.cpu().numpy().astype(str(self.inputs[1].dtype)))
    method_d = {
        "XOR": np.bitwise_xor,
        "AND": np.bitwise_and,
        "OR": np.bitwise_or,
        "NOT": np.bitwise_not,
    }

    output = method_d[self.params['method'].upper()](*func_args)
    self.outputs[0].betensor = torch.from_numpy(output.astype("int64"))  # torch not support uint32
    return output
