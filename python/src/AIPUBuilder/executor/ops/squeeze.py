from ..engine import register_gt
from .gt_util.quant_utils import clamp
from AIPUBuilder.core import OpType
import numpy as np

@register_gt(OpType.Squeeze)
def squeeze(self, *args, **kwargs):
    inp = self.inputs[0].irtensor.numpy()
    out = np.squeeze(inp, axis = tuple(self.params['axis']))
    self.outputs[0].irtensor = out
