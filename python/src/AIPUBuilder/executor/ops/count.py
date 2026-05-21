from ..engine import register_gt
from AIPUBuilder.core import Target
from .gt_util.quant_utils import clamp, _tpc_lut
from AIPUBuilder.core import OpType
import numpy as np
import torch


@register_gt(OpType.Count)
def count(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float:
        hist = count_fp(self)
        self.outputs[0].betensor = hist
    else:
        input_data = self.inputs[0].betensor.cpu().numpy()
        batch_size = self.inputs[0].shape[0]
        min = int(self.params["min"])
        max = int(self.params["max"])

        label_size = (input_data.size) // batch_size
        input0 = input_data.reshape(batch_size, label_size)
        stride = max - min
        hist = []

        for i in range(batch_size):
            hist += [len(np.where(input0[i] == j)[0]) for j in range(min, max+1)]

        hist = np.asarray(hist, dtype=self.outputs[0].dtype.np)
        hist = hist.reshape(batch_size, (stride + 1))
        self.outputs[0].betensor = torch.from_numpy(hist.astype('float32'))
    return hist


def count_fp(self):
    # to match current maskrcnn float32 IR, and maskrcnn must be from big to small, so exchange min and max
    if 'nbins' not in self.params:
        min_value = torch.max(self.inputs[1].betensor)
        max_value = torch.min(self.inputs[1].betensor)
        bin = (min_value - max_value + 1).int()
    else:
        bin = self.params['nbins']
        min_value = self.params['min']
        max_value = self.params['max']
    discrete = self.params.get('discrete', True)

    if discrete == False:  # nbins means interval between min and max
        # this branch, pytorch api histc don't support
        # interval is [interval[i],interval[i+1]) format
        interval = torch.linspace(min_value, max_value, steps=bin + 1)
        inp_betensors = self.inputs[0].betensor
        out = torch.zeros(list(self.outputs[0].shape), device=self.outputs[0].betensor.device)
        for batch in range(int(self.inputs[0].shape[0])):
            for i in range(bin):
                ge = torch.ge(inp_betensors[batch].flatten(), interval[i])
                lt = torch.lt(inp_betensors[batch].flatten(), interval[i + 1])
                out[batch, i] = (ge * lt).sum()
    else:
        out = torch.zeros(list(self.outputs[0].shape), device=self.outputs[0].betensor.device)
        inp_betensors = self.inputs[0].betensor
        for i in range(int(self.inputs[0].shape[0])):
            if max_value > min_value:  # from small to big
                out[i] = torch.histc(inp_betensors[i].float(), bin, min_value, max_value)
            else:
                # from big to small
                out[i] = torch.flip(torch.histc(inp_betensors[i], bin, max_value, min_value), [0])

    return out.reshape(list(self.outputs[0].shape))
