import numpy as np
import math
import torch
from AIPUBuilder.core import OpType, Dtype
from ..engine import register_gt, register_gen_weight, _GEN_WEIGHTS
from .gt_util.aiff import hwa_gru_to_int16, _aiff_lut, clip_to_bit
from .gt_util.quant_utils import clamp
from .gt_util.dtype_utils import dtype2str, str2dtype, dtype2bits


def clip(data, dtype):
    if 'int8' in dtype:
        # NOTE: clip to [-128, 127] since we use full table
        return np.clip(np.round(data), np.iinfo(np.int8).min, np.iinfo(np.int8).max)
    elif 'int16' in dtype:
        return np.clip(np.round(data), np.iinfo(np.int16).min, np.iinfo(np.int16).max)


def get_h_by_wc_v3(hm1_q, r, Wc0, dtype, mtp_trsh_c_h, bn=None):
    # the element-wise multi of reset and h_t-1
    if 'int8' in dtype:
        hprev_r = np.round(r.astype(np.float32) * hm1_q / 256)
        hprev_r = np.clip(np.round(hprev_r), np.iinfo(np.int8).min, np.iinfo(np.int8).max)
    elif 'int16' in dtype:
        hprev_r = np.round(r * hm1_q / 65536)
    h_by_wc = np.matmul(hprev_r, Wc0)
    # mtp ocvt
    h_by_wc = h_by_wc * (0.5 ** mtp_trsh_c_h)
    h_by_wc = np.clip(np.round(h_by_wc), np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    return h_by_wc


def get_h_by_wc_v1(hm1_q, r, Wc0, dtype, mtp_trsh_c_h, bn=None):
    h_by_wc = np.matmul(hm1_q, Wc0)  # 32bit
    # mtp ocvt
    h_by_wc = h_by_wc * (0.5 ** mtp_trsh_c_h)
    h_by_wc = np.clip(np.round(h_by_wc), np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    h_by_wc = h_by_wc + bn
    h_by_wc = np.clip(np.round(h_by_wc), np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    h_by_wc = r * h_by_wc
    h_by_wc = np.clip(np.round(h_by_wc), np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    return h_by_wc


def gru(self, get_h_by_wc, *args, **kwargs):

    target = kwargs['target'] if 'target' in kwargs else None
    scale_value = self.params['scale_value']
    shift_value = self.params['shift_value']
    input_type = str(self.inputs[0].dtype)
    output_type = str(self.outputs[0].dtype)
    input_bit = dtype2bits(str2dtype(input_type))
    output_bit = dtype2bits(str2dtype(output_type))
    weight_data = self.constants['weights'].betensor.cpu().numpy()
    bias_data = self.constants['biases'].betensor.cpu().numpy()

    b, t, x_size = self.inputs[0].shape
    h_size = self.inputs[1].shape[-1]

    weight_data = weight_data.transpose()
    Wr1 = weight_data[0: x_size, 0: 2 * h_size]
    Wr0 = weight_data[x_size: x_size + h_size, 0: 2 * h_size]
    Wc1 = weight_data[0: x_size, 2 * h_size:]
    Wc0 = weight_data[x_size: x_size + h_size, 2 * h_size:]
    bias_data = np.reshape(bias_data, [1, bias_data.shape[0]])
    gate_bias = bias_data[:, 0:2 * h_size]
    candidate_bias = bias_data[:, 2 * h_size: 3 * h_size:]
    bn = None
    if bias_data.shape[1] == 4 * h_size:
        bn = bias_data[:, 3 * h_size:]

    lut_rt = self.constants['lut_rt'].betensor.cpu().numpy().astype(str(self.constants['lut_rt'].dtype))
    lut_zt = self.constants['lut_zt'].betensor.cpu().numpy().astype(str(self.constants['lut_zt'].dtype))
    lut_ht = self.constants['lut_ht'].betensor.cpu().numpy().astype(str(self.constants['lut_ht'].dtype))

    input_x = self.inputs[0].betensor.long().cpu().numpy()
    if self.params['direction'] == 'reverse':
        input_x = np.flip(input_x, 1)
    output = np.zeros([t, b, h_size])

    output2 = []
    mtp_trsh_rz_h = 0
    mtp_trsh_rz_x = 0
    itp_trsh_rz_h = shift_value[0]
    itp_trsh_rz = shift_value[1]
    if 'int16' in input_type:
        if shift_value[0] > 15:
            mtp_trsh_rz_h = shift_value[0] - 15
        if shift_value[1] > 15:
            mtp_trsh_rz_x = shift_value[1] - 15
        itp_trsh_rz = min(15, shift_value[1])
        itp_trsh_rz_h = min(15, shift_value[0]) + mtp_trsh_rz_x
    gate_bias, bias_shift = hwa_gru_to_int16(gate_bias, mtp_trsh_rz_x)
    gate_bias = np.left_shift(gate_bias, bias_shift).astype(np.int32)

    mtp_trsh_c_h = 0
    mtp_trsh_c_x = 0
    itp_trsh_c_h = shift_value[2]
    itp_trsh_c = shift_value[3]
    if 'int16' in input_type:
        if shift_value[2] > 15:
            mtp_trsh_c_h = shift_value[2] - 15
        if shift_value[3] > 15:
            mtp_trsh_c_x = shift_value[3] - 15
        itp_trsh_c = min(15, shift_value[3])
        itp_trsh_c_h = min(15, shift_value[2]) + mtp_trsh_c_x
    candidate_bias, bias_shift = hwa_gru_to_int16(candidate_bias, mtp_trsh_c_x)
    candidate_bias = np.left_shift(candidate_bias, bias_shift).astype(np.int32)
    if bn is not None:
        if not bn.any():
            bn = 0
            bias_shift = 0
        else:
            bn, bias_shift = hwa_gru_to_int16(bn, mtp_trsh_c_h)
            bn = np.left_shift(bn, bias_shift).astype(np.int32)

    """
    in: [b, t, x]
         |  |  |
    out:[b, t, h]
    """
    hm1_q = self.inputs[1].betensor.long().cpu().numpy()
    input_x_ = input_x.transpose([1, 0, 2])
    for ind_x, X in enumerate(input_x_):

        x_q = X.astype('int64')
        x_by_wg = np.matmul(x_q, Wr1)
        h_by_wg = np.matmul(hm1_q, Wr0)

        if 'int8' in input_type:
            # NOTE: z1 sdp icvt or z2 mtp ocvt can only output 32-bit
            x_by_wg = np.clip(x_by_wg, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
            re_scaled_h_by_wg = h_by_wg * scale_value[0] * (0.5 ** shift_value[0])
            re_scaled_h_by_wg = np.clip(np.round(re_scaled_h_by_wg), np.iinfo(np.int32).min, np.iinfo(np.int32).max)
            # NOTE: z1 sdp or z2 itp E_ALU can only input 32-bit
            x_by_wg = np.clip(x_by_wg + gate_bias, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
            mat_sum = x_by_wg + re_scaled_h_by_wg
            rescaled_mat_sum = mat_sum * scale_value[1] * (0.5 ** shift_value[1])
            rescaled_mat_sum = clip(rescaled_mat_sum, input_type)

            sigm_lut_out = _aiff_lut(rescaled_mat_sum.astype(output_type), lut_rt, target=target)
            r, z = np.split(sigm_lut_out, 2, axis=-1)

            x_by_wc = np.matmul(x_q, Wc1)
            h_by_wc = get_h_by_wc(hm1_q, r, Wc0, input_type, mtp_trsh_c_h, bn)

            x_by_wc = np.clip(x_by_wc, np.iinfo(np.int32).min, np.iinfo(np.int32).max)

            re_scaled_h_by_wc = np.round(h_by_wc.astype(np.float64) * scale_value[2] * (0.5 ** shift_value[2]))
            re_scaled_h_by_wc = np.clip(np.round(re_scaled_h_by_wc), np.iinfo(np.int32).min, np.iinfo(np.int32).max)

            # NOTE: z1 sdp or z2 itp E_ALU can only input 32-bit
            x_by_wc = np.clip(x_by_wc + candidate_bias, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
            mat_sum2 = x_by_wc + re_scaled_h_by_wc
            rescaled_mat_sum2 = mat_sum2 * scale_value[3] * (0.5 ** shift_value[3])
            rescaled_mat_sum2 = clip(rescaled_mat_sum2, input_type)

            c = _aiff_lut(rescaled_mat_sum2.astype(output_type), lut_ht, target=target)

            c_times_1_minus_u = (127.0 - z) * c.astype(np.int32)

            rescaled_c_times_1_minus_u = c_times_1_minus_u * scale_value[4] * (0.5 ** shift_value[4])
            rescaled_c_times_1_minus_u = np.clip(np.round(rescaled_c_times_1_minus_u), np.iinfo(np.int32).min,
                                                 np.iinfo(np.int32).max)

            hm1_q = rescaled_c_times_1_minus_u + z.astype(np.int32) * hm1_q

            hm1_q = hm1_q * scale_value[5] * (0.5 ** shift_value[5])
            hm1_q = clip(hm1_q, input_type)
            output[ind_x] = hm1_q

        if 'int16' in input_type:
            # mtp ocvt
            h_by_wg = h_by_wg * (0.5 ** mtp_trsh_rz_h)
            h_by_wg = np.clip(np.round(h_by_wg), np.iinfo(np.int32).min, np.iinfo(np.int32).max)
            # itp mul
            re_scaled_h_by_wg = h_by_wg * scale_value[0] * (0.5 ** itp_trsh_rz_h)
            # itp clamp
            re_scaled_h_by_wg = np.clip(np.round(re_scaled_h_by_wg), np.iinfo(np.int32).min >> mtp_trsh_rz_x,
                                        np.iinfo(np.int32).max >> mtp_trsh_rz_x)
            # mtp ocvt
            x_by_wg = x_by_wg * (0.5 ** mtp_trsh_rz_x)
            x_by_wg = np.clip(np.round(x_by_wg), np.iinfo(np.int32).min, np.iinfo(np.int32).max)
            # itp alu
            mat_sum = np.clip(x_by_wg + gate_bias, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
            # itp alu
            mat_sum = mat_sum + re_scaled_h_by_wg
            # itp clamp and ocvt
            hi_val = math.ceil((np.iinfo(np.int16).max << itp_trsh_rz) * 1.0 / scale_value[1])
            lo_val = math.floor((np.iinfo(np.int16).min << itp_trsh_rz) * 1.0 / scale_value[1])
            mat_sum = np.clip(mat_sum, lo_val, hi_val)
            rescaled_mat_sum = mat_sum * scale_value[1] * (0.5 ** itp_trsh_rz)
            rescaled_mat_sum = np.clip(np.round(rescaled_mat_sum), np.iinfo(output_type).min, np.iinfo(output_type).max)

            sigm_lut_out = _aiff_lut(rescaled_mat_sum.astype(output_type), lut_rt, target=target)

            r, z = np.split(np.int32(sigm_lut_out), 2, axis=-1)

            x_by_wc = np.matmul(x_q, Wc1)
            h_by_wc = get_h_by_wc(hm1_q, r, Wc0, input_type, mtp_trsh_c_h, bn)

            # mtp ocvt
            x_by_wc = x_by_wc * (0.5 ** mtp_trsh_c_x)
            x_by_wc = np.clip(np.round(x_by_wc), np.iinfo(np.int32).min, np.iinfo(np.int32).max)
            # itp mul
            re_scaled_h_by_wc = np.round(h_by_wc.astype(np.float64) * scale_value[2] * (0.5 ** itp_trsh_c_h))
            re_scaled_h_by_wc = np.clip(np.round(re_scaled_h_by_wc), np.iinfo(np.int32).min, np.iinfo(np.int32).max)
            # itp alu
            mat_sum2 = np.clip(x_by_wc + candidate_bias, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
            # itp alu
            mat_sum2 = np.clip(mat_sum2 + re_scaled_h_by_wc, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
            # itp clamp and ocvt
            hi_val = math.ceil((np.iinfo(np.int16).max << itp_trsh_c) * 1.0 / scale_value[3])
            lo_val = math.floor((np.iinfo(np.int16).min << itp_trsh_c) * 1.0 / scale_value[3])
            mat_sum2 = np.clip(mat_sum2, lo_val, hi_val)
            rescaled_mat_sum2 = mat_sum2 * scale_value[3] * (0.5 ** itp_trsh_c)
            rescaled_mat_sum2 = np.clip(np.round(rescaled_mat_sum2), np.iinfo(
                output_type).min, np.iinfo(output_type).max)

            c = _aiff_lut(rescaled_mat_sum2.astype(output_type), lut_ht, target=target)

            c = np.array(c, dtype='int16')
            c_times_1_minus_u = (32767.0 - z) * c.astype(np.int32)

            rescaled_c_times_1_minus_u = c_times_1_minus_u * scale_value[4] * (0.5 ** shift_value[4])
            rescaled_c_times_1_minus_u = np.clip(np.round(rescaled_c_times_1_minus_u), np.iinfo(np.int32).min,
                                                 np.iinfo(np.int32).max)
            hm1_q = rescaled_c_times_1_minus_u + z.astype(np.int32) * hm1_q
            hm1_q = hm1_q * scale_value[5] * (0.5 ** shift_value[5])
            hm1_q = clip(hm1_q, input_type)
            output[ind_x] = hm1_q
    output2 = hm1_q
    output = output.astype(output_type).transpose([1, 0, 2])
    output2 = np.asarray(output2).astype(output_type)

    if self.params['direction'] == 'reverse':
        output = np.flip(output, 1)
    output_dict = {'H': output, 'Hn': output2}
    for index, o in enumerate(self.params['out_sequence']):
        self.outputs[index].betensor = torch.from_numpy(output_dict[o].copy())
    return [o.betensor for o in self.outputs]

    pass


@register_gt(OpType.GRUv3)
def gruv3(self, *args, **kwargs):
    return gru(self, get_h_by_wc_v3)


@register_gt(OpType.GRUv3, target='Z1_0701')
def gruv3_Z1_0701(self, *args, **kwargs):
    return gru(self, get_h_by_wc_v3, target='Z1_0701')


@register_gt(OpType.GRUv3, target='Z1_0701_P')
def gruv3_Z1_0701_P(self, *args, **kwargs):
    return gru(self, get_h_by_wc_v3, target='Z1_0701_P')


@register_gt(OpType.GRUv1)
def gruv1(self, *args, **kwargs):
    return gru(self, get_h_by_wc_v1)


@register_gt(OpType.GRUv1, target='Z1_0701')
def gruv1_Z1_0701(self, *args, **kwargs):
    return gru(self, get_h_by_wc_v1, target='Z1_0701')


@register_gt(OpType.GRUv1, target='Z1_0701_P')
def gruv1_Z1_0701_p(self, *args, **kwargs):
    return gru(self, get_h_by_wc_v1, target='Z1_0701_P')


@register_gen_weight(OpType.GRUv1)
@register_gen_weight(OpType.GRUv3)
def gen_random_weight(self):
    # fixed value
    lut = np.array(
        [0, 1, 2, 4, 5, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23,
         24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
         42, 43, 44, 45, 45, 46, 47, 47, 48, 49, 49, 50, 50, 51, 51, 52, 52, 53,
         53, 53, 54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 57, 57, 58, 58, 58, 58,
         58, 59, 59, 59, 59, 59, 59, 60, 60, 60, 60, 60, 60, 60, 61, 61, 61, 61,
         61, 61, 61, 61, 61, 61, 61, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62,
         62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 63, 63, 63, 63,
         63, 63, 0, 3, 6, 9, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 42, 45,
         48, 51, 53, 56, 58, 61, 63, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86,
         87, 89, 91, 92, 94, 95, 96, 98, 99, 100, 101, 102, 103, 105, 106, 106, 107, 108,
         109, 110, 111, 111, 112, 113, 113, 114, 115, 115, 116, 116, 117, 117, 118, 118, 119, 119,
         119, 120, 120, 120, 121, 121, 121, 122, 122, 122, 122, 123, 123, 123, 123, 123, 123, 124,
         124, 124, 124, 124, 124, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 126, 126,
         126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
         126, 126, 127, 127] * 2, dtype='int8'
    )
    sigmoid_16 = np.array(
        [66, 70, 73, 77, 81, 85, 89, 93, 98,
         103, 108, 113, 118, 124, 130, 137, 144, 151,
         158, 166, 174, 183, 192, 201, 211, 222, 233,
         244, 256, 269, 282, 296, 310, 326, 342, 358,
         376, 394, 414, 434, 455, 477, 501, 525, 551,
         578, 606, 635, 666, 699, 733, 768, 805, 844,
         885, 928, 972, 1019, 1068, 1119, 1173, 1229, 1287,
         1348, 1413, 1479, 1549, 1623, 1699, 1779, 1862, 1949,
         2039, 2134, 2233, 2336, 2443, 2555, 2671, 2793, 2919,
         3050, 3187, 3329, 3477, 3630, 3789, 3955, 4126, 4304,
         4488, 4679, 4877, 5082, 5293, 5512, 5737, 5970, 6210,
         6458, 6713, 6975, 7244, 7521, 7806, 8097, 8396, 8702,
         9015, 9335, 9662, 9995, 10335, 10681, 11032, 11390, 11752,
         12120, 12492, 12868, 13249, 13633, 14020, 14410, 14802, 15196,
         15591, 15987, 16384, 16781, 17177, 17572, 17966, 18358, 18748,
         19135, 19519, 19900, 20276, 20648, 21016, 21378, 21736, 22087,
         22433, 22773, 23106, 23433, 23753, 24066, 24372, 24671, 24962,
         25247, 25524, 25793, 26055, 26310, 26558, 26798, 27031, 27256,
         27475, 27686, 27891, 28089, 28280, 28464, 28642, 28813, 28979,
         29138, 29291, 29439, 29581, 29718, 29849, 29975, 30097, 30213,
         30325, 30432, 30535, 30634, 30729, 30819, 30906, 30989, 31069,
         31145, 31219, 31289, 31355, 31420, 31481, 31539, 31595, 31649,
         31700, 31749, 31796, 31840, 31883, 31924, 31963, 32000, 32035,
         32069, 32102, 32133, 32162, 32190, 32217, 32243, 32267, 32291,
         32313, 32334, 32354, 32374, 32392, 32410, 32426, 32442, 32458,
         32472, 32486, 32499, 32512, 32524, 32535, 32546, 32557, 32567,
         32576, 32585, 32594, 32602, 32610, 32617, 32624, 32631, 32638,
         32644, 32650, 32655, 32660, 32665, 32670, 32675, 32679, 32683,
         32687, 32691, 32695, 32698], dtype='int16'
    )
    tanh_16 = np.array(
        [-32635, -32629, -32622, -32615, -32607, -32599, -32591, -32582,
         -32573, -32563, -32553, -32542, -32531, -32519, -32507, -32494,
         -32481, -32466, -32451, -32436, -32419, -32402, -32384, -32365,
         -32345, -32325, -32303, -32280, -32256, -32231, -32204, -32176,
         -32147, -32117, -32085, -32051, -32016, -31979, -31941, -31900,
         -31858, -31813, -31766, -31718, -31666, -31613, -31556, -31497,
         -31435, -31371, -31303, -31232, -31158, -31080, -30998, -30913,
         -30824, -30730, -30632, -30530, -30423, -30311, -30194, -30071,
         -29943, -29809, -29669, -29523, -29370, -29211, -29044, -28870,
         -28689, -28500, -28302, -28097, -27882, -27658, -27426, -27183,
         -26930, -26668, -26394, -26110, -25815, -25508, -25189, -24859,
         -24516, -24160, -23791, -23409, -23014, -22605, -22182, -21745,
         -21294, -20828, -20348, -19853, -19343, -18818, -18279, -17725,
         -17156, -16573, -15975, -15363, -14737, -14097, -13444, -12777,
         -12098, -11407, -10703, -9989, -9264, -8529, -7784, -7031,
         -6271, -5503, -4728, -3949, -3165, -2377, -1586, -793,
         0, 793, 1586, 2377, 3165, 3949, 4728, 5503,
         6271, 7031, 7784, 8529, 9264, 9989, 10703, 11407,
         12098, 12777, 13444, 14097, 14737, 15363, 15975, 16573,
         17156, 17725, 18279, 18818, 19343, 19853, 20348, 20828,
         21294, 21745, 22182, 22605, 23014, 23409, 23791, 24160,
         24516, 24859, 25189, 25508, 25815, 26110, 26394, 26668,
         26930, 27183, 27426, 27658, 27882, 28097, 28302, 28500,
         28689, 28870, 29044, 29211, 29370, 29523, 29669, 29809,
         29943, 30071, 30194, 30311, 30423, 30530, 30632, 30730,
         30824, 30913, 30998, 31080, 31158, 31232, 31303, 31371,
         31435, 31497, 31556, 31613, 31666, 31718, 31766, 31813,
         31858, 31900, 31941, 31979, 32016, 32051, 32085, 32117,
         32147, 32176, 32204, 32231, 32256, 32280, 32303, 32325,
         32345, 32365, 32384, 32402, 32419, 32436, 32451, 32466,
         32481, 32494, 32507, 32519, 32531, 32542, 32553, 32563,
         32573, 32582, 32591, 32599, 32607, 32615, 32622, 32629], dtype='int16'
    )
    _GEN_WEIGHTS['_all'](self)
    if str(self.constants['weights'].dtype) == 'int8':
        self.constants['lut_rt'].betensor = lut[:256]
        self.constants['lut_zt'].betensor = lut[:256]
        self.constants['lut_ht'].betensor = lut[256:]
    elif str(self.constants['weights'].dtype) == 'int16':
        self.constants['lut_rt'].betensor = sigmoid_16
        self.constants['lut_zt'].betensor = sigmoid_16
        self.constants['lut_ht'].betensor = tanh_16

    if self.constants['biases'].shape[0] == 4 * self.inputs[1].shape[1] and np.random.randint(2) % 2 == 0:
        self.constants['biases'].betensor[3 * self.inputs[1].shape[1]:] = 0
