import torch
from AIPUBuilder.Optimizer.framework import *


def layernorm(self: PyNode) -> torch.Tensor:
    mask = torch.tensor(0xFFFFFF00).int()
    def to24(x): return (x.view(torch.int32) & mask).view(torch.float32)
    inp = to24(self.inputs[0].betensor.half().float())
    axis = -1
    gamma = to24(self.constants['weights'].betensor.half().float())
    beta = to24(self.constants['biases'].betensor.float())

    n = to24(torch.tensor(self.inputs[0].ir_shape[axis]).float())
    mean = to24(inp.sum(axis, keepdim=True) / n)
    diff = to24(inp - mean)
    sqrsum = to24(diff**2).sum(axis, keepdim=True)
    rsqrt = to24(sqrsum**-0.5)
    sqrtn = to24(n ** 0.5)

    norm = to24(to24(diff * rsqrt) * sqrtn)
    res = to24(norm * gamma) + beta
    self.outputs[0].betensor = res.half()
    return res
