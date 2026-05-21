import torch
import numpy as np
from ..engine import register_gt
from AIPUBuilder.core import OpType, Dtype
from .gt_util.quant_utils import clamp


@register_gt(OpType.Intersection)
def Intersection(self, *args, **kwargs):
    input0 = self.inputs[0].betensor
    input1 = self.inputs[1].betensor
    output = np.intersect1d(input0, input1)
    output = output.astype(self.outputs[0].dtype.np)
    layout = self.outputs[0].shape.layout
    self.outputs[0].set_numpy(output)  # output.shape could != ir.shape
    self.outputs[0].shape.layout = layout  # restore layout after set_numpy
    self.outputs[0].betensor = output  # set_numpy() can not change betensor after aipugb
