from ..engine import register_gt
from .gt_util.quant_utils import clamp
from AIPUBuilder.core import OpType
import torch
import numpy as np


@register_gt(OpType.ArgMinMax)
def argminmax(self, *args, **kwargs):
    """
    :param: select_last_index: an param to select the last index or the first index
            if the max/min element appears more than one time.
    :return:
    """
    if self.inputs[0].dtype.is_float:
        input_data = self.inputs[0].betensor.clone().float()
    else:
        input_data = self.inputs[0].betensor.clone().long()
    axis = self.params['axis']
    axis = axis + len(input_data.shape) if axis < 0 else axis

    select_last_index = self.params['select_last_index']
    if isinstance(select_last_index, str):
        select_last_index = select_last_index.upper()
    method = self.params['method']

    # np.argmax will return the min index in many same value.
    # so we can get max index by reversing the data at first.
    if select_last_index in ['TRUE', 1]:
        input_reverse = torch.flip(input_data, dims=[axis])
        if method.upper() == 'MAX':
            output_reverse = torch.argmax(input_reverse, axis=axis)
        elif method.upper() == 'MIN':
            output_reverse = torch.argmin(input_reverse, axis=axis)
        output = self.inputs[0].betensor.shape[axis] - output_reverse - 1
        pass
    else:
        if method.upper() == 'MAX':
            output = torch.argmax(input_data, axis=axis)
        elif method.upper() == 'MIN':
            output = torch.argmin(input_data, axis=axis)
        else:
            raise TypeError('do not support method = %s' % method)
    output = output.view(self.outputs[0].betensor.shape)
    output = clamp(output, self.outputs[0].dtype)
    self.outputs[0].betensor = output

    return self.outputs[0].betensor
