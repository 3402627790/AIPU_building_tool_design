import torch
from .gt_util import *
from ..engine import register_gt, register_gen_input
from AIPUBuilder.core import OpType


@register_gt(OpType.GatherElements)
def GatherElements(self, *args, **kwargs):
    axis = self.params['axis']
    params = self.inputs[0].betensor
    ids = self.inputs[1].betensor
    max_ids = self.inputs[0].shape[axis]
    ids = torch.where(ids < 0, ids + max_ids, ids).long()

    output = torch.gather(params, axis, ids)
    output = clamp(output, self.outputs[0].dtype)
    self.outputs[0].betensor = output
    return [o.betensor for o in self.outputs]


@register_gen_input(OpType.GatherElements)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)

    if self.inputs[0].dtype.is_float:
        input0 = (np.random.uniform(-10000, 10000, size=in_shape0)).astype(in_type0)
    else:
        input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)

    axis = self.params['axis']
    min_v = -in_shape0[axis]
    max_v = in_shape0[axis]
    max_v = min(max_v, np.iinfo(in_type1).max + 1)
    min_v = max(min_v, np.iinfo(in_type1).min)

    if 'u' in in_type1:
        input1 = np.random.randint(0, max_v, size=in_shape1, dtype=in_type1)
    else:
        input1 = np.random.randint(min_v, max_v, size=in_shape1, dtype=in_type1)

    input_data = [input0, input1]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    return input_data
