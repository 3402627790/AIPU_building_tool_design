from ..engine import register_gt, register_gen_weight, _OP_METRICS
from AIPUBuilder.core import OpType
from AIPUBuilder.logger import ERROR
from .gt_util import *
import numpy as np
import torch

MOVE_BITS = 30


@register_gt(OpType.Normalization)
@register_gt(OpType.Normalization)
def normalization(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float:
        is_aiff = kwargs.get('is_aiff', False)
        tpc_or_aiff = 'aiff' if is_aiff else 'tpc'
        print('run normalization fp: %s version !' % tpc_or_aiff)
        self.attrs['is_aiff'] = is_aiff  # for metric
        if is_aiff:
            outputs = normalization_float_aiff(self)
        else:
            outputs = normalization_float(self)
        self.outputs[0].irtensor = outputs
    else:
        inp = self.inputs[0].betensor.cpu().numpy()
        input_type = str(self.inputs[0].dtype)
        output_type = str(self.outputs[0].dtype)
        input_shape = list(self.inputs[0].shape).copy()
        output_shape = list(self.outputs[0].shape)
        method = self.params['method'].lower()

        input_array = inp.astype(np.int64)
        input_array = input_array + self.inputs[0].quantization.offset

        if method == 'l1':
            outputs = l1_norm(self, input_array)
        elif method == 'l2':
            outputs = l2_norm(self, input_array)
        else:
            raise ValueError('do not support method == %s !' % method)

        outputs -= self.outputs[0].quantization.offset
        outputs = np.clip(outputs, np.iinfo(output_type).min, np.iinfo(output_type).max)
        outputs = np.reshape(outputs, output_shape)
        self.outputs[0].betensor = torch.from_numpy(outputs.astype('int64'))


def l1_norm(self, input_array):
    shift_value = self.params['shift_value']
    scale_value = self.params['scale_value']
    axis = self.params['axis'].copy() if isinstance(self.params['axis'], list) else self.params['axis']

    l1_sum = np.sum(np.abs(input_array), axis=tuple(axis), keepdims=True)
    for a in axis:
        l1_sum = np.repeat(l1_sum, repeats=list(self.inputs[0].shape)[a], axis=a)

    do_shift = shift_value
    if do_shift < 0:
        l1_sum = l1_sum >> (0 - do_shift)
        do_shift = 0

    output_array = np.trunc(input_array * scale_value / l1_sum)
    output_array[l1_sum == 0] = 0
    output_array = np.round(output_array * 0.5 ** do_shift)
    return output_array


def l2_norm(self, input_array):
    shift_value = self.params['shift_value']
    scale_value = self.params['scale_value']
    axis = self.params['axis'].copy() if isinstance(self.params['axis'], list) else self.params['axis']
    pre_shift = self.params.get('reciprocal_shift_value', 0)
    pre_shift = 0 if pre_shift == None else pre_shift
    lut = self.constants['lut'].numpy()

    q16_min, q16_max = np.iinfo('int16').min, np.iinfo('int16').max
    act_qmin, act_qmax = np.iinfo('int32').min, np.iinfo('int32').max

    axis = [axis] if isinstance(axis, int) else axis
    sum = np.sum(input_array ** 2, axis=tuple(axis), keepdims=True)

    hi, lo = divide_into_2parts(self, sum.astype('int64'))
    out32, shift_need = combine_31bit_data(self, lo, hi)
    shift_need_zero = np.zeros_like(shift_need)
    shift_need = np.where(sum >= (1 << MOVE_BITS), shift_need, shift_need_zero)
    sum = np.where(sum >= (1 << MOVE_BITS), out32, sum)  # data should < 30(MOVE_BITS) bit to use fast sqrt method

    rev_sqrt = rsqrt(self, lut, sum.astype('int64'))  # 1 / sqrt(sum(x ** 2)),
    byte_width, sqrt_out = D_UTIL_norm_l(rev_sqrt)

    had_shift_zero = np.zeros_like(byte_width)
    rsqnorm_limit_bits = 15
    had_shift = np.where(byte_width > rsqnorm_limit_bits, byte_width - rsqnorm_limit_bits, had_shift_zero)

    shift_all = 31 - had_shift + shift_need + shift_value - pre_shift

    rev_sqrt = np.clip(rev_sqrt >> had_shift, q16_min, q16_max)

    input_array = np.clip(input_array * scale_value, act_qmin, act_qmax)
    input_array = np.clip(input_array >> pre_shift, q16_min, q16_max)

    output_array = np.clip(input_array * rev_sqrt, act_qmin, act_qmax)
    output_array = np.round(output_array * 0.5 ** shift_all)
    return output_array


def rsqrt(self, lut, x):
    # t = int(self.D_UTIL_inverse_sqrt(x) / 0x10000)
    t = D_UTIL_inverse_sqrt(lut, x)
    return t


def combine_31bit_data(self, lo, hi):
    shift, temp = D_UTIL_norm_l(hi)
    shift = np.where(shift >= (31 - MOVE_BITS), MOVE_BITS + shift - 31, 0)
    shift = np.where(shift % 2, shift + 1, shift)
    out = (hi << (MOVE_BITS - shift)) + (lo >> shift)
    shift = shift >> 1
    return out, shift


def divide_into_2parts(self, input):
    hi = (input >> MOVE_BITS)
    lo = (input & ((1 << MOVE_BITS) - 1))
    return hi, lo


def normalization_float(self):
    eps = self.params.get('epsilon', 1e-12)
    eps = 1e-12 if eps == None else eps
    axis = self.params['axis']
    if self.params['method'].lower() == 'l1':
        p = 1
    elif self.params['method'].lower() == 'l2':
        p = 2
    else:
        ERROR('normalization only support l1 and l2!')
    x = self.inputs[0].betensor.float()
    # x_pow = torch.abs(x) ** p
    # norm = torch.sum(x_pow, dim=axis, keepdim=True).float()
    # norm[norm < eps] = eps
    # norm = torch.pow(norm, 1 / p)
    # output = x / norm

    trans_perm = axis_to_perm(axis, x.dim())
    restore_perm = invert_perm(trans_perm)
    new_shape = [x.shape[trans_perm[i]] for i in range(x.dim())]

    in_shape = x.shape
    tmp_len = np.prod([in_shape[a] for a in self.params['axis']])
    tmp1_len = np.prod(in_shape) // tmp_len
    data1 = torch.permute(x, trans_perm)  # transpose, calc across the last dim
    data1 = data1.reshape([tmp1_len, tmp_len])  # n-dim -> 2-dim
    data_tmp = torch.zeros([tmp1_len, 8]).float()
    data_tmp_square = torch.zeros([tmp1_len, 8]).float()
    data_tmp, data_tmp_square = sum_square_fp16(torch.abs(data1), tmp_len, data_tmp, data_tmp_square, merged=True)

    if self.params['method'].lower() == 'l1':
        data_tmp = np.sum(data_tmp.float().cpu().numpy(), axis=-1, keepdims=True)  # _vrpadd
        data_tmp[data_tmp < eps] = eps
        output = data1 / data_tmp
    else:
        data_tmp_square = np.sum(data_tmp_square.float().cpu().numpy(), axis=-1, keepdims=True)  # _vrpadd
        data_tmp_square[data_tmp_square < eps] = eps
        norm = f32_rsqrt(data_tmp_square)
        output = data1 * norm

    output = output.reshape(new_shape)  # 2-dim -> n-dim
    output = torch.permute(output, restore_perm)  # transpose
    return output


def normalization_float_aiff(self):
    eps = self.params.get('epsilon', 1e-12)
    eps = 1e-12 if eps == None else eps
    input = self.inputs[0].irtensor.float()

    input = fp32_to_fp24(input)
    if self.params['method'].lower() == 'l1':
        power2 = fp32_to_fp24(np.abs(input))
    elif self.params['method'].lower() == 'l2':
        power2 = fp32_to_fp24(input ** 2)
    else:
        ERROR('normalization only support l1 and l2!')

    power2_sum = aiff_itp_acc_channel(power2)
    power2_sum[power2_sum < eps] = eps

    if self.params['method'].lower() == 'l1':
        norm = 1 / power2_sum
    elif self.params['method'].lower() == 'l2':
        norm = f32_rsqrt(power2_sum)

    norm = fp32_to_fp24(norm)
    output = input * norm
    output = fp32_to_fp24(output)
    return output


@register_gen_weight(OpType.Normalization)
def gen_random_weight(self):
    if 'lut' in self.constants:
        self.constants['lut'].betensor = np.array(
            [32767, 32515, 32268, 32026, 31790, 31558, 31332, 31111, 30894,
             30682, 30474, 30270, 30070, 29874, 29682, 29494, 29309, 29127,
             28949, 28774, 28602, 28434, 28268, 28105, 27945, 27787, 27632,
             27480, 27330, 27183, 27038, 26895, 26755, 26617, 26481, 26346,
             26214, 26084, 25956, 25830, 25705, 25583, 25462, 25342, 25225,
             25109, 24994, 24882, 24770, 24660, 24552, 24445, 24339, 24235,
             24132, 24031, 23930, 23831, 23733, 23637, 23541, 23447, 23354,
             23262, 23170, 23080, 22992, 22904, 22817, 22731, 22646, 22562,
             22479, 22396, 22315, 22235, 22155, 22077, 21999, 21922, 21845,
             21770, 21695, 21621, 21548, 21476, 21404, 21333, 21263, 21193,
             21124, 21056, 20988, 20921, 20855, 20789, 20724, 20660, 20596,
             20533, 20470, 20408, 20346, 20285, 20225, 20165, 20106, 20047,
             19988, 19930, 19873, 19816, 19760, 19704, 19649, 19594, 19539,
             19485, 19431, 19378, 19326, 19273, 19221, 19170, 19119, 19068,
             19018, 18968, 18919, 18870, 18821, 18773, 18725, 18677, 18630,
             18583, 18536, 18490, 18444, 18399, 18354, 18309, 18264, 18220,
             18176, 18133, 18090, 18047, 18004, 17962, 17920, 17878, 17837,
             17795, 17755, 17714, 17674, 17634, 17594, 17554, 17515, 17476,
             17438, 17399, 17361, 17323, 17285, 17248, 17211, 17174, 17137,
             17100, 17064, 17028, 16992, 16957, 16921, 16886, 16851, 16817,
             16782, 16748, 16714, 16680, 16646, 16613, 16579, 16546, 16514,
             16481, 16448, 16416, 16384], dtype='int16')


@register_op_metric(OpType.Normalization)
@check_fp_data
def metric(self, outputs, index=None):
    # note: aiff's metric maybe called in layoutconvertor
    def aiff_call(r, u, c, t):
        if "float" in t:
            return np.all(u <= 1)
        return np.all(r == 0)

    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]

    layer_dtypes = [str(o.dtype) for o in self.outputs] if index is None else [str(self.outputs[index].dtype)]

    if self.attrs.get('is_aiff', False):
        rel_diffs = relative_diff(outputs, groundtruths)
        ulp_diffs = ulp_diff(outputs, groundtruths)
        cos_dists = cos_distance(outputs, groundtruths)
        return all(aiff_call(r, u, c, t) for r, u, c, t in zip(rel_diffs, ulp_diffs, cos_dists, layer_dtypes))
    return _OP_METRICS['_all'](self, outputs, index)
