from ..engine import register_gt, register_gen_input
from AIPUBuilder.core import OpType
import torch


@register_gt(OpType.InTopK)
def intopk(self, *args, **kwargs):
    k = self.params['k']

    inp_betensors1 = self.inputs[0].betensor
    inp_betensors2 = self.inputs[1].betensor
    topk_v, topk_indice = torch.topk(inp_betensors1, k, -1, largest=True, sorted=True)
    out = []
    for i in range(len(inp_betensors2)):
        out.append(inp_betensors1[i, int(inp_betensors2[i])] in topk_v[i, :])
    self.outputs[0].betensor = torch.tensor(out)
    return self.outputs[0].betensor


@register_gen_input(OpType.InTopK)
def gen_input_data(self):
    import numpy as np
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)

    if self.inputs[0].dtype.is_float:
        predictions = (np.random.uniform(-10000, 10000, size=in_shape0)).astype(in_type0)
    else:
        predictions = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)
    predictions_shape = in_shape0
    targets_shape = in_shape1

    max_d = np.iinfo(in_type1).max + 1
    targets = np.random.randint(min(max_d, predictions_shape[-1]),
                                size=targets_shape, dtype=in_type1)
    input_data = [predictions, targets]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    return input_data
