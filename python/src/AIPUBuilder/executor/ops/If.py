from ..engine import register_gt, register_gen_input, _GEN_INPUTS
from AIPUBuilder.core import OpType
import numpy as np
from ..engine import _subgraph


@register_gt(OpType.If)
def if_node(self, *args, **kwargs):
    _subgraph("_GT", *args, kwargs)(self, *args, kwargs)


@register_gt(OpType.IfHeader)
def if_header(self, *args, **kwargs):
    for index, it in enumerate(self.inputs):
        self.outputs[index + 1].betensor = it.betensor


@register_gt(OpType.IfEnder)
def if_ender(self, *args, **kwargs):
    cond = self.inputs[1].betensor.item()
    offset = 2 if cond != 0 else 2 + len(self.outputs)
    for i in range(len(self.outputs)):
        self.outputs[i].betensor = self.inputs[i + offset].betensor


@register_gt(OpType.Dummy)
def if_dummy(self, *args, **kwargs):
    for index, it in enumerate(self.inputs):
        self.outputs[index].betensor = it.betensor
