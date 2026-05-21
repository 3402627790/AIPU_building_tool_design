import torch
import numpy as np
from ..engine import register_gt
from .gt_util.aiff import _aiff_lut, hwa_int32_to_int16_with_round
from AIPUBuilder.core import Target
from AIPUBuilder.core import OpType


@register_gt(OpType.RnnSeq)
def rnnseq(self, *args, **kwargs):
    input_seq = self.inputs[0].betensor.cpu().numpy()
    weight_data = self.constants['weights'].numpy()
    bias_data = self.constants['biases'].numpy()
    x_size = self.inputs[0].shape[-1]
    h_size = self.outputs[0].shape[-1]
    weight_data = weight_data.transpose()
    Wxh = weight_data[0: x_size, :]
    Whh = weight_data[x_size: x_size + h_size, :]
    bias_data, bias_shift = hwa_int32_to_int16_with_round(bias_data, 31, 0)
    bias_data = np.left_shift(bias_data, bias_shift).astype('int32')
    bias_data = np.reshape(bias_data, [1, h_size])

    MAX_INT32 = 2147483647
    output = np.zeros(self.outputs[0].shape)

    for ind_t, T in enumerate(input_seq):  # for every batch
        h_q = np.zeros([1, h_size])
        for ind_x, X in enumerate(T):
            x_q = X.reshape([1, x_size]).astype(np.int64)
            x_q_w = np.matmul(x_q, Wxh)
            h_q_w = np.matmul(h_q.astype(np.int64), Whh)
            x_q_w_s = x_q_w * self.params['scale_value'][1]
            x_q_w_s = np.clip(np.round(x_q_w_s), -MAX_INT32 - 1, MAX_INT32)
            h_q_w_s = h_q_w * self.params['scale_value'][0]
            h_q_w_s = np.clip(np.round(h_q_w_s), -MAX_INT32 - 1, MAX_INT32)
            mat_sum = x_q_w_s + bias_data
            # NOTE: z1 sdp or z2 itp E_ALU can only input 32-bit
            mat_sum = np.clip(mat_sum, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
            mat_sum = mat_sum + h_q_w_s
            # NOTE: z1 sdp or z2 itp E_ALU can only input 32-bit
            mat_sum = np.clip(mat_sum, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
            mat_sum = mat_sum * self.params['scale_value'][2] * (0.5 ** self.params['shift_value'])
            mat_sum = np.clip(np.round(mat_sum), -128, 127)
            mat_sum = mat_sum.astype(np.int8)
            h_q = _aiff_lut(mat_sum, self.constants['lut'].numpy(), target=Target().name)
            h_q = h_q.astype(np.int8)
            output[ind_t][ind_x] = h_q
    output = output.astype(np.int8)
    self.outputs[0].betensor = output
