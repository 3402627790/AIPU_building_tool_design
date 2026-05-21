import torch
import numpy as np
from ..engine import register_gt
from .gt_util.quant_utils import clamp
from AIPUBuilder.core import OpType
from .gt_util.quant_utils import fp_activation


@register_gt(OpType.ConvMerge)
def convmerge(self, *args, **kwargs):
    convs = [i.betensor.cpu().numpy().astype('int16') for i in self.inputs]
    split_num = self.params['split_num']
    out_shape = list(self.outputs[0].betensor.shape)
    out_dim = 1
    for i in out_shape:
        out_dim *= i
    output_mantissa = np.zeros([21, out_dim]).astype(np.int64)
    exp = np.ones([21, out_dim]).astype(np.int16) * -10000

    def bit_num(num):
        bitnum = torch.ceil(torch.log2(torch.abs(torch.tensor(num)))).int()
        return bitnum
    conv_cal = 0
    for i in range(21):
        exp_offset = 4
        j_cal = i + 1
        if i >= split_num:
            j_cal = 2 * split_num - j_cal
        for j in range(j_cal):
            output_mantissa[i] += convs[conv_cal].reshape([out_dim])
            conv_cal += 1
        indices = np.where(output_mantissa[i] != 0)
        exp[i][indices] = i * exp_offset + bit_num(output_mantissa[i][indices]) - 35
    res = np.zeros([out_dim]).astype(np.int64)
    sgn = np.zeros([out_dim]).astype(np.int32)
    exp_max = np.max(exp, axis=0)

    output_mantissa.tofile("output_mantissa.bin")
    tmp0 = np.ones([out_dim]).astype(np.int32)
    tmp1 = np.ones([out_dim]).astype(np.int32)
    exp_offset = 4
    for i in range(21):

        indices_0 = np.where((exp_max > i * exp_offset) & (exp_max - i * exp_offset < 32))
        indices_1 = np.where(exp_max <= i * exp_offset)
        indices_2 = np.where(output_mantissa[i] < 0)
        indices_3 = np.where(output_mantissa[i] >= 0)
        tmp0[indices_2] = -1
        tmp0[indices_3] = 1
        tmp2 = output_mantissa[i]
        tmp1[indices_3] = output_mantissa[i][indices_3]
        tmp1[indices_2] = -output_mantissa[i][indices_2]

        res[indices_0] += (tmp1[indices_0] >> (exp_max[indices_0] - i * exp_offset))*tmp0[indices_0]
        res[indices_1] += output_mantissa[i][indices_1] << (i * exp_offset - exp_max[indices_1])

    indices = np.where(res < 0)
    sgn[indices] = 1
    res[indices] = - res[indices]
    for i in range(out_dim):
        while (res[i] >> 11) > 0:
            exp_max[i] += 1
            res[i] = res[i] >> 1
        while (res[i] >> 10) == 0:
            exp_max[i] -= 1
            res[i] = res[i] << 1
    exp_max -= 25

    exp_max[np.where(exp_max >= 0x1f)] = 0x1f
    res[np.where(exp_max >= 0x1f)] = 0
    indices = np.where((-9 <= exp_max) & (exp_max <= 0))
    exp_max[indices] = 0
    res[indices] = res[indices] >> (1 - exp_max[indices])
    exp_max[np.where(exp_max < -9)] = 0
    res[np.where(exp_max < -9)] = 0
    res = res & 0x3ff
    output_sign = sgn
    output_exp = exp_max
    output_mantissa = res
    output_int = (sgn << 15 | exp_max << 10 | res)
    output_int = output_int.astype('int16')
    output = np.frombuffer(output_int.tobytes(), dtype='float16').reshape(list(self.outputs[0].shape))
    if self.params['with_activation'].upper() not in ['NONE', 'RELU']:
        raise ValueError('conv fp16 no support such activation method now')
    output = fp_activation(torch.from_numpy(output), self)
    output = output.to(torch.float16)
    self.outputs[0].betensor = output
    return self.outputs[0].betensor
