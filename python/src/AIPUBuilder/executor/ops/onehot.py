import torch
import numpy as np
from ..engine import register_gt, register_gen_input
from AIPUBuilder.core import OpType, Dtype
from .gt_util.quant_utils import clamp


@register_gt(OpType.OneHot)
def OneHot(self, *args, **kwargs):
    inp = self.inputs[0].betensor.long()
    off_value = self.params['values'][0]
    on_value = self.params['values'][1]
    axis = self.params['axis']
    new_axis = axis if axis >= 0 else len(inp.shape)
    # if axis exceeds length of input shape, do not need transpose
    if axis > len(inp.shape) - 1:
        axis = new_axis = -1
    # pytorch one_hot only support last dim indice,so we need transpose inp
    # according ir definition:
    # axis =-1, shape=batch,feature,num_class
    # axis = 0, shape=num_class,batch,feature
    # axis = 1, shape=batch,num_class,feature
    # indice is negatibe and out of depth, 0 replace
    classes_number = int(self.params['depth'])

    trans_inp = torch.transpose(inp.unsqueeze(axis), axis, -1)
    invalid_mask = torch.bitwise_or(trans_inp < 0, trans_inp >= classes_number)
    trans_inp[invalid_mask] = 0
    # legal_data_flag = (trans_inp >= 0) * (trans_inp < classes_number)
    # trans_inp = trans_inp*legal_data_flag
    trans_y = torch.nn.functional.one_hot(trans_inp, classes_number)
    trans_y[invalid_mask, :] = 0
    # trans_y[..., 0] = trans_y[..., 0]*legal_data_flag
    y = torch.transpose(trans_y.squeeze(-2), new_axis, -1)
    out = y.clone().to(torch.int64) if self.outputs[0].dtype.is_int else y.clone().to(torch.float32)
    out[y == 1] = on_value
    out[y == 0] = off_value
    if self.outputs[0].dtype.is_int:
        out = clamp(out, self.outputs[0].dtype)

    self.outputs[0].betensor = out


@register_gen_input(OpType.OneHot)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_type0 = str(self.inputs[0].dtype)

    depth = self.params['depth']
    if self.inputs[0].dtype.is_float:
        indices = np.random.uniform(1, 10000, size=in_shape0).astype(in_type0)
    else:
        indices = np.random.randint(0, min(depth, np.iinfo(in_type0).max), size=in_shape0, dtype=in_type0)

    input_data = [indices]
    self.inputs[0].betensor = input_data[0]
    return input_data
