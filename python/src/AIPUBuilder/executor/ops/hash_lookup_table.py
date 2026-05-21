import numpy as np
import math
import torch
from AIPUBuilder.core import OpType
from ..engine import register_gt
import torch


@register_gt(OpType.HashLookupTable)
def hash_lookup_table(self, *args, **kwargs):
    data = self.inputs[0].betensor.cpu().numpy()   # input_data
    data2 = self.inputs[1].betensor.cpu().numpy()  # data_key
    data3 = self.inputs[2].betensor.cpu().numpy()  # search_key
    result = []
    sear_result = []
    input_shape = data.shape
    data_type = self.inputs[0].dtype.np
    miss_default_data = self.params.get("miss_default_data", 0)
    for sear in data3:
        if sear in data2:
            result.append(data[list(data2).index(sear)])
            sear_result.append(1)
        else:
            # zero = np.zeros(input_shape[1:], dtype=data_type)
            # result.append(zero)
            pad_value = np.ones(input_shape[1:], dtype=data_type)
            pad_value *= miss_default_data
            result.append(pad_value)
            sear_result.append(0)
    result = np.asarray(result)
    sear_result = np.array(sear_result, dtype='uint8')
    self.outputs[0].betensor = result
    self.outputs[1].betensor = sear_result
    return [o.betensor for o in self.outputs]
