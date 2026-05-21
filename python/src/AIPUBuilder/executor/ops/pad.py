import torch
import numpy as np
from ..engine import register_gt
from AIPUBuilder.core import OpType
from .gt_util.dtype_utils import torch_to_numpy, numpy_to_torch


@register_gt(OpType.Pad)
@register_gt(OpType.InternalPad)
def pad(self, *args, **kwargs):
    x = torch_to_numpy(self.inputs[0].irtensor, self.inputs[0].dtype.np)

    pads = self.params['pads']
    if '_pads' in self.attrs:
        pads = self.attrs['_pads']

    constant_value = self.params['constant_value'] if 'constant_value' in self.params else 0
    if 'mode' in self.params and self.params['mode'] in ['REFLECT', 'SYMMETRIC', 'EDGE', 'WRAP']:
        output = np.pad(x, pads, self.params['mode'].lower())
    else:
        output = np.pad(x, pads, 'constant', constant_values=constant_value)
    self.outputs[0].irtensor = numpy_to_torch(
        output, self.outputs[0].dtype.np) if isinstance(output, np.ndarray) else output
    return self.outputs[0].betensor
