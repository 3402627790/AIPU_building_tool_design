from ..engine import register_gt
from AIPUBuilder.core import OpType
import torch


@register_gt(OpType.InternalSpliter)
@register_gt(OpType.Split)
def split(self, *args, **kwargs):
    input_data = self.inputs[0].irtensor
    axis = self.params['axis'] if 'axis' in self.params.keys() else -1
    if 'size_splits' in self.params.keys():
        splits = self.params['size_splits']
    else:
        splits = self.params['splits']

    output = list(torch.split(input_data, splits, dim=axis))
    for out_t, o in zip(self.outputs, output):
        out_t.irtensor = o
    return [self.outputs[i].betensor for i in range(len(output))]
