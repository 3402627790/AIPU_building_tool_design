import torch
import numpy as np
from ..engine import register_gt
from .gt_util.quant_utils import clamp
from AIPUBuilder.core import OpType


@register_gt(OpType.ConvPre)
def convpre(self, *args, **kwargs):
    inp = self.inputs[0].betensor.cpu().numpy().flatten().astype('float16')
    inp = np.frombuffer(inp.tobytes(), dtype='uint16')
    split_num = self.params['split_num']
    dim = np.size(inp)
    outp = np.zeros([split_num, dim]).astype("int16")
    exp_offset = 4

    for i in range(dim):
        # input_inp = inp[i]
        input_sign = inp[i] >> 15
        if (input_sign == 0):
            input_sign = 1
        else:
            input_sign = -1
        input_exp = (inp[i] & 0x7fff) >> 10
        if (input_exp == 0):
            input_exp = 1
            input_mantissa = inp[i] & 0x3ff
        else:
            input_mantissa = (1 << 10) | (inp[i] & 0x3ff)

        for j in range(11):
            exp_start = j * exp_offset
            exp_end = (j + 1) * exp_offset - 1
            exp0 = input_exp
            exp1 = exp0 + 10
            if ((exp0 <= exp_start) & (exp_end <= exp1)):
                outp[j, i] = (input_mantissa & ((1 << (exp_end - exp0 + 1)) - 1)) >> (exp_start - exp0)
            elif ((exp_start <= exp1) & (exp_start >= exp0) & (exp_end > exp1)):
                outp[j, i] = (input_mantissa >> (exp_start - exp0))
            elif ((exp_end <= exp1) & (exp_end >= exp0) & (exp_start < exp0)):
                outp[j, i] = (input_mantissa & ((1 << (exp_end - exp0 + 1)) - 1)) << (exp0 - exp_start)
            elif ((exp_start <= exp0) & (exp_end >= exp1)):
                outp[j, i] = input_mantissa << (exp0 - exp_start)
            outp[j, i] *= input_sign

    for i in range(split_num):
        self.outputs[i].betensor = outp[i].reshape(list(self.outputs[i].shape))
    return [self.outputs[i].betensor for i in range(split_num)]
