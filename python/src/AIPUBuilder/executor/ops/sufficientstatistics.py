from .gt_util import *
from ..engine import register_gt
from AIPUBuilder.core import OpType


@register_gt(OpType.SufficientStatistics)
def SufficientStatistics(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float:
        mean, var = sufficientStatistics_float(self)
    else:
        inp0 = self.inputs[0].betensor
        inp1 = self.inputs[1].betensor

        out_type0 = str(self.outputs[0].dtype)
        out_type1 = str(self.outputs[1].dtype)
        x = inp0 + self.inputs[0].quantization.offset
        shift = inp1 + self.inputs[1].quantization.offset
        axis = self.params['axis']
        out0_scale, out1_scale, input0_scale, input1_scale = self.params['scale_value']
        out0_shift, out1_shift, input0_shift, input1_shift = self.params['shift_value']

        # calculate mean
        act_qmin, act_qmax = -2 ** 31, 2 ** 31 - 1
        # In fact, the true value domain of x and shift after requantize is within the range of int15
        x = torch.clamp(torch.round(x.double() * input0_scale * (0.5 ** input0_shift)) - 0, act_qmin, act_qmax).long()
        shift = torch.clamp(torch.round(shift.double() * input1_scale * (0.5 ** input1_shift)) - 0, act_qmin,
                            act_qmax).long()
        # In fact, the true value domain of m_ss is within the range of int16
        m_ss = x - shift
        output0 = torch.sum(m_ss, axis, keepdims=True)
        # Limit the number of items to 65535
        output0 = torch.clamp(output0, act_qmin, act_qmax)
        out0 = torch.clamp(torch.div(output0 * out0_scale, 2 ** out0_shift, rounding_mode='trunc').int() - self.outputs[
            0].quantization.offset, np.iinfo(out_type0).min, np.iinfo(out_type0).max).long()
        mean = out0.cpu().numpy().astype(out_type0)

        # calculate var
        v_ss = (m_ss * m_ss).long()
        v_ss = torch.clamp(v_ss, 0, 2 ** 32 - 1)
        high16_v_ss = torch.div(v_ss, 2 ** 16, rounding_mode='floor')
        high_sum = torch.sum(high16_v_ss, axis, keepdims=True)
        high_sum = torch.clamp(high_sum, 0, 2 ** 32 - 1)
        high_output = (high_sum * out1_scale * 2 ** 16)

        low16_v_ss = v_ss - high16_v_ss * 2 ** 16
        low_sum = torch.sum(low16_v_ss, axis, keepdims=True).long()
        low_sum = torch.clamp(low_sum, 0, 2 ** 32 - 1)
        low_output = (low_sum * out1_scale)
        out1 = torch.clamp(((high_output + low_output) >> out1_shift) - self.outputs[1].quantization.offset,
                           np.iinfo(out_type1).min, np.iinfo(out_type1).max).int()
        var = out1.cpu().numpy().astype(out_type1)

    output = [mean, var]
    self.outputs[0].betensor = mean
    self.outputs[1].betensor = var
    return output


def sufficientStatistics_float(self):
    x = self.inputs[0].betensor.float()
    y = self.outputs[0].betensor.float()
    shift = self.inputs[1].betensor.float()
    axis = self.params['axis']
    m_ss = x - shift
    v_ss = m_ss * m_ss

    # out0 = torch.sum(m_ss, axis, keepdims=True).half()
    # out1 = torch.sum(v_ss, axis, keepdims=True).half()

    trans_perm = axis_to_perm(axis, x.dim())
    restore_perm = invert_perm(trans_perm)
    new_shape = [y.shape[trans_perm[i]] for i in range(x.dim())]

    in_shape = x.shape
    tmp_len = np.prod([in_shape[a] for a in self.params['axis']])
    tmp1_len = np.prod(in_shape) // tmp_len
    data1 = torch.permute(m_ss, trans_perm).numpy()  # transpose, calc across the last dim
    data1 = data1.reshape([tmp1_len, tmp_len])  # n-dim -> 2-dim
    data2 = torch.permute(v_ss, trans_perm).numpy()
    data2 = data2.reshape([tmp1_len, tmp_len])

    data_sum = np.zeros([tmp1_len, 16], dtype='float32')
    data_sum = sum_fp16(data1, tmp_len, data_sum)
    out0 = np.sum(data_sum, axis=-1, keepdims=True).astype('float16')  # _vrpadd
    out0 = out0.reshape(new_shape)  # 2-dim -> n-dim
    out0 = np.transpose(out0, restore_perm)  # transpose

    data2_sum = np.zeros([tmp1_len, 16], dtype='float32')
    data2_sum = sum_fp16(data2, tmp_len, data2_sum)
    out1 = np.sum(data2_sum, axis=-1, keepdims=True).astype('float16')  # _vrpadd
    out1 = out1.reshape(new_shape)
    out1 = np.transpose(out1, restore_perm)

    return out0, out1
