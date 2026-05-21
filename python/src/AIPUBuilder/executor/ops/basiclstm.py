from ..engine import register_gt, register_gen_weight, _GEN_WEIGHTS
from AIPUBuilder.core import OpType
from .gt_util import clip_to_bit
from .gt_util.aiff import _aiff_lut, lstm_bias_int32_to_int16_with_round, x3_aiff_fp32_2_fpx
from AIPUBuilder.core import Target
import numpy as np
import copy
import tqdm
from .gt_util.aiff import _aiif_ahead_shift_biases, get_ops_options, _aiff_lookup_float_index_lut
import torch
from .gt_util.math_utils import fp32_to_fp24


@register_gt(OpType.BasicLSTM)
def basiclstm_all(self, *args, **kwargs):
    self.sgattrs = kwargs['sgattrs'] if 'sgattrs' in kwargs else None
    if self.sgattrs is None:
        self.sgattrs = dict()
    self.is_32bit_bias = self.sgattrs['is_32bit_bias'] if 'is_32bit_bias' in self.sgattrs else False
    return basiclstm(self)


@register_gt(OpType.BasicLSTM, target='Z1_0701')
def basiclstm_Z1_0701(self, *args, **kwargs):
    return basiclstm(self, target='Z1_0701')


@register_gt(OpType.BasicLSTM, target='Z1_0701_P')
def basiclstm_Z1_0707_P(self, *args, **kwargs):
    return basiclstm(self, target='Z1_0701_P')


def basiclstm(layer, *args, **kwargs):
    if layer.inputs[0].dtype.is_float:
        basiclstm_fp16(layer)
        return

    target = kwargs['target'] if 'target' in kwargs else None
    input_data_x = layer.inputs[0].betensor.cpu().numpy().astype(layer.inputs[0].dtype.np)
    hm1_q_s = layer.inputs[1].betensor.cpu().numpy().astype(layer.inputs[1].dtype.np)
    c_prev_s = layer.inputs[2].betensor.cpu().numpy().astype(layer.inputs[2].dtype.np)
    lut_index_type = str(layer.inputs[0].dtype).replace('u', '')

    if layer.params['direction'] == 'reverse':
        input_data_x = np.flip(input_data_x, 1)

    scale = layer.constants['scale'].numpy()
    shift = layer.constants['shift'].numpy()

    weight_data = layer.constants['weights'].numpy()
    bias = layer.constants['biases'].numpy()
    b, t_size, x_size = layer.inputs[0].shape
    h_size = layer.outputs[0].shape[-1]

    if scale.shape[0] == 8 * t_size or scale.shape[0] == 11 * t_size:
        multi_time_step_forwarwd(layer)
        return

    weight_data = weight_data.transpose()

    wx_q = weight_data[0: x_size, :]
    wh_q = weight_data[x_size:, :]

    MAX_INT32 = 2**31-1
    MAX_INT16 = 2**15-1
    MAX_INT8 = 127

    output = np.zeros([b, t_size, h_size])  # no top shape here because top shape[0](t) could be 1
    output2 = []
    output3 = []

    """
    in: [b, t, x]
            |  |  |
    out:[b, t, h]
    """

    bias_i, bias_g, bias_f, bias_o = np.split(bias, 4, axis=0)
    bias_ifo = np.concatenate([bias_i, bias_f, bias_o], axis=0)
    ignore_sp_target = False
    if layer.inputs[0].dtype.bits == 8:
        lut_it = hwa_zy_clamp_lut(layer.constants['lut_it'].numpy())
        lut_ct = hwa_zy_clamp_lut(layer.constants['lut_ct'].numpy())
        lut_ft = hwa_zy_clamp_lut(layer.constants['lut_ft'].numpy())
        lut_ot = hwa_zy_clamp_lut(layer.constants['lut_ot'].numpy())
        if hwa_zy_lstm_check_lut_mirror(lut_it, True) and hwa_zy_lstm_check_lut_mirror(lut_ct, False) \
                and hwa_check_all_positive_shift(shift):
            ignore_sp_target = True
        if not layer.is_32bit_bias:
            bias_ifo, bshift_ifo = lstm_bias_int32_to_int16_with_round(bias_ifo, 31, 0, True)
            bias_ifo = np.left_shift(bias_ifo, bshift_ifo).astype('int32')
            bias_g, bshift_g = lstm_bias_int32_to_int16_with_round(bias_g, 31, 0, True)
            bias_g = np.left_shift(bias_g, bshift_g).astype('int32')
        bias_i, bias_f, bias_o = np.split(bias_ifo, 3, axis=0)
        bias = np.concatenate([bias_i, bias_g, bias_f, bias_o], axis=0)
    else:
        lut_it = layer.constants['lut_it'].numpy()
        lut_ct = layer.constants['lut_ct'].numpy()
        lut_ft = layer.constants['lut_ft'].numpy()
        lut_ot = layer.constants['lut_ot'].numpy()
        if not layer.is_32bit_bias:
            bias_ifo, bshift_ifo = lstm_bias_int32_to_int16_with_round(
                bias_ifo, 31, shift[1]-layer.params['lut_shift_value'], True)
            bias_ifo = np.left_shift(bias_ifo, bshift_ifo).astype('int32')
            bias_g, bshift_g = lstm_bias_int32_to_int16_with_round(
                bias_g, 31, shift[1]-layer.params['lut_shift_value'], True)
            bias_g = np.left_shift(bias_g, bshift_g).astype('int32')
        else:
            bias_ifo = np.right_shift(bias_ifo,  shift[1]-layer.params['lut_shift_value']).astype('int32')
            bias_g = np.right_shift(bias_g, shift[1]-layer.params['lut_shift_value']).astype('int32')
        bias_i, bias_f, bias_o = np.split(bias_ifo, 3, axis=0)
        bias = np.concatenate([bias_i, bias_g, bias_f, bias_o], axis=0)

    # [b, t, x] ==> [t, b, x]
    input_data_x_ = input_data_x.transpose([1, 0, 2])
    for t, X in tqdm.tqdm(enumerate(input_data_x_)):
        x_q = X.astype('int32')
        x_by_w = np.matmul(x_q, wx_q.astype(np.float64))
        h_by_w = np.matmul(hm1_q_s, wh_q.astype(np.float64))

        if 'int8' in input_data_x.dtype.name:
            req_h_by_w = np.round(h_by_w * scale[0] * (0.5 ** shift[0]))
            req_h_by_w = np.clip(np.round(req_h_by_w), -MAX_INT32-1, MAX_INT32)
            mat_sum = np.clip(x_by_w.astype(np.float64) + bias.astype(np.float64), -
                              MAX_INT32-1, MAX_INT32) + req_h_by_w.astype(np.float64)
            mat_sum = clip_to_bit(mat_sum, 32, True)
            i_tmp, g_tmp, f_tmp, o_tmp = np.split(mat_sum, 4, axis=1)
            i_in = np.round(i_tmp * scale[1] * (0.5 ** shift[1]))
            i_in = np.clip(np.round(i_in), -MAX_INT8-1, MAX_INT8)
            g_in = np.round(g_tmp * scale[2] * (0.5 ** shift[2]))
            g_in = np.clip(np.round(g_in), -MAX_INT8-1, MAX_INT8)
            f_in = np.round(f_tmp * scale[1] * (0.5 ** shift[1]))
            f_in = np.clip(np.round(f_in), -MAX_INT8-1, MAX_INT8)
            o_in = np.round(o_tmp * scale[1] * (0.5 ** shift[1]))
            o_in = np.clip(np.round(o_in), -MAX_INT8-1, MAX_INT8)
            i = _aiff_lut(i_in.astype(lut_index_type), lut_it, target=target, ignore_sp_target=ignore_sp_target)
            g = _aiff_lut(g_in.astype(lut_index_type), lut_ct, target=target, ignore_sp_target=ignore_sp_target)
            f = _aiff_lut(f_in.astype(lut_index_type), lut_ft, target=target, ignore_sp_target=ignore_sp_target)
            o = _aiff_lut(o_in.astype(lut_index_type), lut_ot, target=target, ignore_sp_target=ignore_sp_target)
            f_times_c_prev = f.astype(np.float64) * c_prev_s.astype(np.float64)
            res_f_times_c_prev = np.round(f_times_c_prev * scale[5 + 2 * t] * (0.5 ** shift[5 + 2 * t]))
            res_f_times_c_prev = np.clip(np.round(res_f_times_c_prev), -MAX_INT32-1, MAX_INT32)
            c_tmp = i.astype(np.float64) * g.astype(np.float64) + res_f_times_c_prev.astype(np.float64)
            c_tmp = clip_to_bit(c_tmp, 32, True)
            re_scaled_c_tmp = np.round(c_tmp * scale[3] * (0.5 ** shift[3]))
            re_scaled_c_tmp = np.clip(np.round(re_scaled_c_tmp), -MAX_INT8, MAX_INT8)
            hm1_q_tmp = o.astype(np.float64) * _aiff_lut(re_scaled_c_tmp.astype(lut_index_type),
                                                         layer.constants['lut_h'].numpy(), target=target, ignore_sp_target=ignore_sp_target).astype(np.float64)
            hm1_q_s = np.round(hm1_q_tmp * scale[4] * (0.5 ** shift[4]))
            hm1_q_s = np.clip(np.round(hm1_q_s), -MAX_INT8, MAX_INT8)
            c_prev_s = np.round(c_tmp * scale[5 + 2 * t + 1] * (0.5 ** shift[5 + 2 * t + 1]))
            c_prev_s = np.clip(np.round(c_prev_s), -MAX_INT8, MAX_INT8)
        elif 'int16' in input_data_x.dtype.name:
            x_by_w = Requantize(x_by_w, 1, shift[1]-layer.params['lut_shift_value'], precison=32)
            h_by_w = Requantize(h_by_w, 1, shift[0]-layer.params['lut_shift_value'], precison=32)
            req_h_by_w = np.round(h_by_w * scale[0] * (0.5 ** shift[1]))  # int8 is diff
            req_h_by_w = np.clip(np.round(req_h_by_w), -MAX_INT32-1, MAX_INT32)
            mat_sum = np.clip(x_by_w.astype(np.float64) + bias.astype(np.float64), -
                              MAX_INT32-1, MAX_INT32) + req_h_by_w.astype(np.float64)
            mat_sum = clip_to_bit(mat_sum, 32, True)
            i_tmp, g_tmp, f_tmp, o_tmp = np.split(mat_sum, 4, axis=1)
            i_in = Requantize(i_tmp, scale[1], layer.params['lut_shift_value'])
            g_in = Requantize(g_tmp, scale[2], int(shift[2])-int(shift[1]) + layer.params['lut_shift_value'])
            f_in = Requantize(f_tmp, scale[1], layer.params['lut_shift_value'])
            o_in = Requantize(o_tmp, scale[1], layer.params['lut_shift_value'])
            i = _aiff_lut(i_in.astype(lut_index_type), lut_it, target=target, ignore_sp_target=ignore_sp_target)
            g = _aiff_lut(g_in.astype(lut_index_type), lut_ct, target=target, ignore_sp_target=ignore_sp_target)
            f = _aiff_lut(f_in.astype(lut_index_type), lut_ft, target=target, ignore_sp_target=ignore_sp_target)
            o = _aiff_lut(o_in.astype(lut_index_type), lut_ot, target=target, ignore_sp_target=ignore_sp_target)
            f_times_c_prev = f.astype(np.float64) * c_prev_s.astype(np.float64)
            res_f_times_c_prev = np.round(f_times_c_prev * scale[5 + 2 * t] * (0.5 ** shift[5 + 2 * t]))
            res_f_times_c_prev = np.clip(np.round(res_f_times_c_prev), -MAX_INT32-1, MAX_INT32)
            c_tmp = i.astype(np.float64) * g.astype(np.float64) + res_f_times_c_prev.astype(np.float64)
            c_tmp = clip_to_bit(c_tmp, 32, True)
            re_scaled_c_tmp = Requantize(c_tmp, scale[3],  shift[3])
            hm1_q_tmp = o.astype(np.float64) * _aiff_lut(re_scaled_c_tmp.astype(lut_index_type),
                                                         layer.constants['lut_h'].numpy(), target=target, ignore_sp_target=ignore_sp_target).astype(np.float64)
            hm1_q_s = Requantize(hm1_q_tmp, scale[4], shift[4])
            hm1_q_s = np.clip(hm1_q_s, -MAX_INT16, MAX_INT16)
            c_prev_s = Requantize(c_tmp, scale[5 + 2 * t + 1], shift[5 + 2 * t + 1])
        output[:, t] = hm1_q_s
    output2.append(hm1_q_s)
    output3.append(c_prev_s)

    if layer.params['direction'] == 'reverse':
        output = np.flip(output, 1)

    output = output.astype(layer.outputs[0].dtype.np)
    output2 = np.asarray(output2).astype(layer.outputs[0].dtype.np)
    output3 = np.asarray(output3).astype(layer.outputs[0].dtype.np)
    output_dict = {'Y': output, 'H': output2, 'C': output3}
    outputs = []
    for o in layer.params['out_sequence']:
        outputs.append(output_dict[o])
    for i in range(len(layer.outputs)):
        layer.outputs[i].betensor = outputs[i].reshape(layer.outputs[i].shape)
    return outputs


def basiclstm_fp16(layer):
    input_data_x = layer.inputs[0].betensor.to(torch.float32)
    hm1_q_s = layer.inputs[1].betensor.to(torch.float32)
    c_prev_s = layer.inputs[2].betensor.to(torch.float32)

    if layer.params['direction'] == 'reverse':
        input_data_x = np.flip(input_data_x, 1)

    weight_data = layer.constants['weights'].betensor.to(torch.float32)
    bias = layer.constants['biases'].betensor.to(torch.float32)
    b, t_size, x_size = layer.inputs[0].shape
    h_size = layer.outputs[0].shape[-1]

    weight_data = weight_data.transpose(0, 1)

    wx_q = weight_data[0: x_size, :]
    wh_q = weight_data[x_size:, :]

    # no top shape here because top shape[0](t) could be 1
    lstm_type = layer.inputs[0].dtype
    output = torch.zeros([b, t_size, h_size], dtype=lstm_type.torch)
    output2 = torch.zeros([b, h_size], dtype=lstm_type.torch)
    output3 = torch.zeros([b, h_size], dtype=lstm_type.torch)

    """
    in: [b, t, x]
            |  |  |
    out:[b, t, h]
    """

    bias_i, bias_g, bias_f, bias_o = torch.split(bias, h_size, dim=0)
    bias_ifo = torch.concat([bias_i, bias_f, bias_o], dim=0)

    bias_i, bias_f, bias_o = torch.split(bias_ifo, h_size, dim=0)
    bias = torch.concat([bias_i, bias_g, bias_f, bias_o], dim=0)

    lut_it = layer.constants['lut_it'].betensor.to(torch.float32)
    lut_ct = layer.constants['lut_ct'].betensor.to(torch.float32)
    lut_ft = layer.constants['lut_ft'].betensor.to(torch.float32)
    lut_ot = layer.constants['lut_ot'].betensor.to(torch.float32)
    lut_ht = layer.constants['lut_h'].betensor.to(torch.float32)

    # [b, t, x] ==> [t, b, x]
    input_data_x_ = input_data_x.transpose(0, 1)
    for t, X in tqdm.tqdm(enumerate(input_data_x_)):
        x_q = X.to(torch.float32)
        hm1_q_s = hm1_q_s.to(torch.float32)
        c_prev_s = c_prev_s.to(torch.float32)
        # x_by_w = torch.from_numpy(np.matmul(x_q.numpy(),wx_q.numpy()))
        # h_by_w = torch.from_numpy(np.matmul(hm1_q_s.numpy(),wh_q.numpy()))
        x_by_w = torch.matmul(x_q, wx_q)
        h_by_w = torch.matmul(hm1_q_s, wh_q)

        req_h_by_w = h_by_w
        mat_sum = x_by_w + bias + req_h_by_w
        i_tmp, g_tmp, f_tmp, o_tmp = torch.split(mat_sum, h_size, dim=1)
        i_in = i_tmp
        g_in = g_tmp
        f_in = f_tmp
        o_in = o_tmp

        ifo_index_scale = layer.params['lut_it_index_scale_value']
        ifo_index_offset = layer.params['lut_it_index_offset_value']
        ifo_value_offset = layer.params['lut_it_value_offset_value']
        c_index_scale = layer.params['lut_ct_index_scale_value']
        c_index_offset = layer.params['lut_ct_index_offset_value']
        c_value_offset = layer.params['lut_ct_value_offset_value']
        mirror_mode = True

        i = _aiff_lookup_float_index_lut(
            i_in, lut_it, ifo_index_scale, ifo_index_offset, mirror_mode, ifo_value_offset)
        g = _aiff_lookup_float_index_lut(
            g_in, lut_ct, c_index_scale, c_index_offset, mirror_mode, c_value_offset)
        f = _aiff_lookup_float_index_lut(
            f_in, lut_ft, ifo_index_scale, ifo_index_offset, mirror_mode, ifo_value_offset)
        o = _aiff_lookup_float_index_lut(
            o_in, lut_ot, ifo_index_scale, ifo_index_offset, mirror_mode, ifo_value_offset)

        i = x3_aiff_fp32_2_fpx(i, fpx=lstm_type)
        g = x3_aiff_fp32_2_fpx(g, fpx=lstm_type)
        f = x3_aiff_fp32_2_fpx(f, fpx=lstm_type)
        o = x3_aiff_fp32_2_fpx(o, fpx=lstm_type)

        i = i.to(torch.float32)
        g = g.to(torch.float32)
        f = f.to(torch.float32)
        o = o.to(torch.float32)

        f_times_c_prev = f * c_prev_s
        res_f_times_c_prev = fp32_to_fp24(f_times_c_prev)
        c_tmp = fp32_to_fp24(i*g) + res_f_times_c_prev
        re_scaled_c_tmp = c_tmp
        hm1_q_tmp = o * _aiff_lookup_float_index_lut(
            re_scaled_c_tmp, lut_ht, c_index_scale, c_index_offset, mirror_mode, c_value_offset)

        hm1_q_s = fp32_to_fp24(hm1_q_tmp)
        hm1_q_s = x3_aiff_fp32_2_fpx(hm1_q_s, fpx=lstm_type)
        c_prev_s = fp32_to_fp24(c_tmp)
        c_prev_s = x3_aiff_fp32_2_fpx(c_prev_s, fpx=lstm_type)
        output[:, t] = hm1_q_s
    output2 = hm1_q_s
    output3 = c_prev_s

    if layer.params['direction'] == 'reverse':
        output = torch.flip(output, 1)

    output_dict = {'Y': output, 'H': output2, 'C': output3}
    outputs = []
    for o in layer.params['out_sequence']:
        outputs.append(output_dict[o])
    for i in range(len(layer.outputs)):
        layer.outputs[i].betensor = outputs[i]
    return outputs


def Requantize(src, scale, shift, precison=16):
    max0 = 2.0 ** (precison - 1) - 1
    min0 = - 2.0 ** (precison - 1)
    quantized_down = np.round(src * scale * (0.5 ** shift))
    return np.clip(quantized_down, min0, max0)


def hwa_zy_lstm_check_lut_mirror(lut, is_sigmoid):
    # when 8bit and return true, lib of z1 will use tpc branch
    for i in range(128):
        index = i+128
        mirror_index = 128-i
        offset = 0
        if is_sigmoid:
            offset = ((lut[255]-lut[1]) >> 1) + lut[1]
        temp0 = lut[index]-offset
        temp1 = -(lut[mirror_index]-offset)
        if((temp0 != temp1) or (temp0 < -128) or (temp0 > 127)):
            return False
    return True


def hwa_zy_clamp_lut(lut):
    lut_new = copy.deepcopy(lut)
    lut_new[0] = lut_new[1]
    return lut_new


def hwa_check_all_positive_shift(shift):
    neg_count = sum(shift < 0)
    if neg_count > 0:
        return False
    return True


def gen_random_weight_float16(self):
    # lut_type = str(self.constants['lut_it'].dtype)
    # luts = ((np.random.random(size=[512]) - 0.5) * 1000).astype(lut_type)
    luts = torch.sin(torch.from_numpy(np.random.random(size=[1024])))
    self.constants['lut_it'].betensor = luts[:512]
    self.constants['lut_ft'].betensor = luts[:512]
    self.constants['lut_ot'].betensor = luts[:512]
    self.constants['lut_ct'].betensor = luts[512:]
    self.constants['lut_h'].betensor = luts[512:]


@register_gen_weight(OpType.BasicLSTM)
def gen_random_weight(self):
    _GEN_WEIGHTS['_all'](self)
    in_dtype = str(self.inputs[0].dtype)

    if in_dtype.endswith('float16'):
        gen_random_weight_float16(self)
        return

    lut_temp_half1 = np.random.randint(-63, 63, size=[127], dtype='int8')
    lut_temp_half2 = -np.flip(lut_temp_half1, axis=0)
    temp0 = np.random.randint(-128, 127, size=[1], dtype='int8')
    temp1 = np.array([0], dtype='int8')
    lut_temp = np.concatenate([temp0, lut_temp_half2, temp1, lut_temp_half1], axis=0)
    lut_sigmoid = lut_temp + 64

    lut_temp_half1 = np.random.randint(-127, 127, size=[127], dtype='int8')
    lut_temp_half2 = -np.flip(lut_temp_half1, axis=0)
    temp0 = np.random.randint(-128, 127, size=[1], dtype='int8')
    temp1 = np.array([0], dtype='int8')
    lut_temp = np.concatenate([temp0, lut_temp_half2, temp1, lut_temp_half1], axis=0)
    lut_tanh = lut_temp
    luts_8bit = np.concatenate([lut_sigmoid, lut_tanh], axis=0)
    if in_dtype.endswith('16'):
        luts_8bit = np.int16(luts_8bit * 256)
    self.constants['lut_it'].betensor = luts_8bit[:256]
    self.constants['lut_ft'].betensor = luts_8bit[:256]
    self.constants['lut_ot'].betensor = luts_8bit[:256]
    self.constants['lut_ct'].betensor = luts_8bit[256:]
    self.constants['lut_h'].betensor = luts_8bit[256:]

    scale_type = str(self.constants['scale'].dtype)
    shift_type = str(self.constants['shift'].dtype)
    weights_type = str(self.constants['weights'].dtype)
    if '16' in scale_type:  # hardware just have 15bit scale
        self.constants['scale'].betensor = np.random.randint(0, 2 ** 15 - 1,
                                                             size=self.constants['scale'].shape, dtype=scale_type)
    self.constants['shift'].betensor = np.random.randint(0, 21, self.constants['shift'].shape, shift_type)
    if weights_type == 'int16':
        shift = self.constants['shift'].numpy()
        lut_shift_value = self.params['lut_shift_value']
        shift[0:2] = np.random.randint(lut_shift_value, lut_shift_value + 16,
                                       2)  # first 3 shift value must >= 16 for 16bit IR
        shift[2] = np.random.randint(max(shift[1] - lut_shift_value, lut_shift_value),
                                     max(shift[1] - lut_shift_value,
                                         lut_shift_value) + 16, 1)
        self.constants['shift'].betensor = shift


def multi_time_step_forwarwd(layer):
    # function to support new quant method
    input_data_x = layer.inputs[0].betensor.long().cpu().numpy()
    hm1_q_s = layer.inputs[1].betensor.long().cpu().numpy()
    c_prev_s = layer.inputs[2].betensor.long().cpu().numpy()

    weight_data = layer.constants['weights'].betensor.long().cpu().numpy()
    bias = layer.constants['biases'].betensor.long().cpu().numpy()

    b, t_size, x_size = layer.inputs[0].shape
    h_size = layer.outputs[0].shape[-1]
    weight_data = weight_data.transpose()

    wx_q = weight_data[0: x_size, :]
    wh_q = weight_data[x_size:, :]

    time_step = layer.params['time_steps']

    lut_ot = layer.constants['lut_ot'].betensor.long().cpu().numpy().reshape([time_step, -1])
    lut_it = layer.constants['lut_it'].betensor.long().cpu().numpy().reshape([time_step, -1])
    lut_ft = layer.constants['lut_ft'].betensor.long().cpu().numpy().reshape([time_step, -1])
    lut_gt = layer.constants['lut_ct'].betensor.long().cpu().numpy().reshape([time_step, -1])
    lut_h = layer.constants['lut_h'].betensor.long().cpu().numpy().reshape([time_step, -1])

    scales = layer.constants['scale'].betensor.long().cpu().numpy().reshape([time_step, -1])
    shifts = layer.constants['shift'].betensor.long().cpu().numpy().reshape([time_step, -1])
    zerops = layer.constants['zerop'].betensor.long().cpu().numpy().reshape([time_step, -1])

    MAX_INT32 = 2**31-1
    MAX_INT16 = 2**15-1
    MAX_INT8 = 127

    index_max = 255

    output = np.zeros([b, time_step, h_size])  # no top shape here because top shape[0](t) could be 1
    output2 = np.zeros([b, time_step, h_size])  # no top shape here because top shape[0](t) could be 1

    def linear_requantize(x, scale, shift, zerop, qmin, qmax):
        if isinstance(x, np.ndarray):
            dtype = x.dtype
            if shift < 0:
                assert(False)
            y = np.clip(np.round(x.astype(np.float32) * scale * (0.5 ** shift)) - zerop, qmin, qmax)
            y = y.astype(dtype)
        else:
            y = np.clip(np.round(x * scale * (0.5 ** shift)) - zerop, qmin, qmax)
        return y

    scale_start = 3
    shift_start = 3

    if layer.params['direction'] == 'reverse':
        input_data_x = np.flip(input_data_x, 1)

    for batch, data_batch in enumerate(input_data_x):  # for every batch
        c_prev = c_prev_s[batch, :]
        hm1_q = hm1_q_s[batch, :].astype(np.int32)
        for ts in range(time_step):
            scale = scales[ts]
            shift = shifts[ts]
            zerop = zerops[ts]
            x_q = data_batch[ts:ts+1]
            x_q = x_q.reshape([1, x_size]).astype(np.int32)

            c_prev_zerop = layer.inputs[2].quantization.offset if ts == 0 else zerops[ts-1][6]
            c_prev = c_prev.astype(np.int32) + c_prev_zerop

            hw_do_scale, hw_do_shift = scale[0], shift[0]
            fc_do_scale, fc_do_shift = scale[1], shift[1]
            hout_do_scale, hout_do_shift = scale[2], shift[2]

            x_by_w = np.matmul(x_q, wx_q.astype(np.int32))
            h_by_w = np.matmul(hm1_q, wh_q.astype(np.int32))

            req_h_by_w = linear_requantize(h_by_w, hw_do_scale, hw_do_shift, 0, -MAX_INT32-1, MAX_INT32)
            mat_sum = linear_requantize(x_by_w + bias + req_h_by_w, 1.0, 0, 0, -MAX_INT32-1, MAX_INT32)

            rescaled_mat_sum = linear_requantize(mat_sum, fc_do_scale, fc_do_shift, zerop[0], -MAX_INT8-1, MAX_INT8)
            i_in, g_in, f_in, o_in = np.split(rescaled_mat_sum, 4, axis=1)

            if isinstance(layer.params['is_forget_bias_absorb'], str) and layer.params['is_forget_bias_absorb'].lower() == 'false':
                # fb_do_scale, scale0, scale1 = scale[scale_start: scale_start+3]
                fb_do_scale = scale[scale_start]
                scale0 = scale[scale_start + 1]
                scale1 = scale[scale_start + 2]
                fb_do_shift = shift[shift_start]
                fc_add_b0 = linear_requantize(f_in + zerop[0], scale0, 0, 0, -MAX_INT32 - 1, MAX_INT32)
                fb_b1 = linear_requantize(
                    float(layer.params['forget_bias']) - layer.params['forget_bias_zp'][0], scale1, 0, 0, -MAX_INT32 - 1, MAX_INT32)
                res_forget_bias = linear_requantize(fc_add_b0 + fb_b1, fb_do_scale,
                                                    fb_do_shift, zerop[1], -MAX_INT8-1, MAX_INT8)
                f_in = res_forget_bias

                scale_start += 3
                shift_start += 1

            i = np.asarray([lut_it[ts][np.clip(np.floor(c).astype(np.int32), 0, index_max)] for c in (i_in + 128)])
            g = np.asarray([lut_gt[ts][np.clip(np.floor(c).astype(np.int32), 0, index_max)] for c in (g_in + 128)])
            f = np.asarray([lut_ft[ts][np.clip(np.floor(c).astype(np.int32), 0, index_max)] for c in (f_in + 128)])
            o = np.asarray([lut_ot[ts][np.clip(np.floor(c).astype(np.int32), 0, index_max)] for c in (o_in + 128)])

            f_times_c_prev = (f.astype(np.int32) + zerop[2]) * c_prev
            i_times_g = (i.astype(np.int32) + zerop[3]) * (g.astype(np.int32) + zerop[5])

            ig_b_do_scale, fcprev_b_do_scale, scale0, scale1, ts_b_do_scale = scale[scale_start: scale_start+5]
            ig_b_do_shift, fcprev_b_do_shift, ts_b_do_shift = shift[shift_start: shift_start+3]

            res_i_times_g = linear_requantize(i_times_g, ig_b_do_scale, ig_b_do_shift, 0, -MAX_INT32-1, MAX_INT32)
            res_i_times_g_times_scale0 = linear_requantize(res_i_times_g, scale0, 0, 0, -MAX_INT32-1, MAX_INT32)
            res_f_times_c_prev = linear_requantize(
                f_times_c_prev, fcprev_b_do_scale, fcprev_b_do_shift, zerop[10], -MAX_INT8-1, MAX_INT8)
            c_tmp = linear_requantize((res_i_times_g_times_scale0 + (res_f_times_c_prev +
                                      zerop[10]) * scale1), ts_b_do_scale, ts_b_do_shift, 0, -MAX_INT32-1, MAX_INT32)

            re_scaled_c_tmp = linear_requantize(
                c_tmp, 1., 0, zerop[6], -MAX_INT8-1, MAX_INT8)  # c_lut no need consider zp
            c_prev = re_scaled_c_tmp

            c_lut_out = np.asarray([lut_h[ts][np.clip(np.floor(c).astype(np.int32), 0, index_max)]
                                   for c in (re_scaled_c_tmp + 128)])
            hm1_q_tmp = (o.astype(np.int32) + zerop[4]) * (c_lut_out.astype(np.int32) + zerop[7])
            h_prev_new = linear_requantize(hm1_q_tmp, hout_do_scale, hout_do_shift, zerop[8], -MAX_INT8 - 1, MAX_INT8)
            hm1_q = h_prev_new

            output[batch][ts] = h_prev_new
            output2[batch][ts] = re_scaled_c_tmp

    if layer.params['direction'].lower() == 'reverse':
        output = np.flip(output, 1)
    output = output.astype(layer.outputs[0].dtype.np)
    output3 = np.asarray(output2[:, -1, :]).astype(layer.outputs[0].dtype.np)
    output2 = np.asarray(output[:, -1, :]).astype(layer.outputs[0].dtype.np)
    output_dict = {'Y': output, 'H': output2, 'C': output3}
    outputs = []
    for o in layer.params['out_sequence']:
        outputs.append(output_dict[o])
    for i in range(len(layer.outputs)):
        layer.outputs[i].betensor = outputs[i]
