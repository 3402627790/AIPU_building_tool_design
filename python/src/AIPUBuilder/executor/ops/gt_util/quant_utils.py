import torch
import numpy as np
from collections import namedtuple
from collections import defaultdict
from .dtype_utils import *


def linear_requantize(x, multiplier, shift_bits, dtype, zero_point=0):
    is_tensor = isinstance(x, torch.Tensor)
    x_t = x if is_tensor else torch.tensor(np.int64(x))
    x_type = str2dtype(str(dtype))
    clamp_min, clamp_max = dtype2range(x_type)
    y = torch.clamp(torch.round(x_t.double() * multiplier * (0.5 ** shift_bits)) - zero_point, clamp_min,
                    clamp_max).double()
    y = torch.where(torch.isnan(y), torch.zeros_like(y, device=y.device), y)
    y = y if is_tensor else y.cpu().numpy().astype(str(dtype))
    return y


def clamp(x, dtype, data_range=None):
    is_tensor = isinstance(x, torch.Tensor)
    if is_tensor:
        x_t = x
    elif 'float' in str(x.dtype):
        x_t = torch.tensor(x)
    else:
        x_t = torch.tensor(np.int64(x))
    if data_range is None:
        if str(dtype).startswith(('float', 'bfloat')):
            # if data_range is None, it means user just want to saturate input as dest dtype
            # if input is all nan, the 'torch.where(torch.isnan(x_t), torch.zeros_like(x_t, device=x_t.device), x_t)' will return all 0
            return x_t if is_tensor else x_t.detach().cpu().numpy().astype(dtype.np)
        if isinstance(dtype, str):
            if 'int' in dtype:
                clamp_min, clamp_max = np.iinfo(dtype).min, np.iinfo(dtype).max
            else:
                clamp_min, clamp_max = np.finfo(dtype).min, np.finfo(dtype).max
        else:
            if dtype.is_int:
                clamp_min, clamp_max = np.iinfo(dtype.np).min, np.iinfo(dtype.np).max
            else:
                clamp_min, clamp_max = np.finfo(dtype.np).min, np.finfo(dtype.np).max
    else:
        clamp_min, clamp_max = data_range
    x_t = torch.clamp(x_t, clamp_min, clamp_max).double()
    x_t = torch.where(torch.isnan(x_t), torch.zeros_like(x_t, device=x_t.device), x_t)
    x_t = x_t if is_tensor else x_t.detach().cpu().numpy().astype(str(dtype))
    return x_t


def _tpc_lut(inputs, lut, lut_in_bits, in_is_signed, is_float=False, align_mode=None):
    """the lut is size of 2^N, use the inputs high Nbits as the lut index and the low bits for interplate.
    :param inputs: inputs data as lut indexes
    :param lut: lookup table
    :param lut_in_bits: indexes range used bits
    :param in_is_signed: indexes sign
    :is_float: Whether the table element is of type float
    :param align_mode: right align and left align
    :return:
    """
    from .dtype_utils import bits2range
    import math
    if not isinstance(lut, torch.Tensor):
        lut = torch.from_numpy(lut)
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.from_numpy(inputs.astype(np.float64))
    table_size = lut.numel()
    table_size_bits = int(math.log2(table_size))
    if 2 ** table_size_bits != table_size:
        print('tpc_lut function can only support lut which has 2**N items.')
    in_qmin, _ = bits2range(lut_in_bits, in_is_signed)
    lutp = torch.nn.functional.pad(lut, (0, 1), value=lut[-1])
    lut_long = lutp.to(torch.float) if is_float else lutp.to(torch.long)
    inputs_long = inputs.to(torch.long)
    indexes = (inputs_long - in_qmin)
    ibits = 0
    lut_index = indexes
    interp_index = 0
    if lut_in_bits > table_size_bits:
        ibits = lut_in_bits - table_size_bits
        if align_mode is not None:
            if align_mode == 'right_align':
                diff = indexes.max() - (table_size - 1)
                lidx = torch.clamp(indexes - diff, 0, table_size - 1)
                lut_index = lidx << ibits
            elif align_mode == 'left_align':
                diff = indexes.min() - 0
                lidx = torch.clamp(indexes - diff, 0, table_size - 1)
                lut_index = lidx << ibits
            else:
                print("tpc_lut align_mode=default.")
        lut_index = (indexes >> ibits).long()
        interp_index = indexes - (lut_index << ibits)
    interp_index = torch.tensor(interp_index) if isinstance(interp_index, int) else interp_index
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    interp_index = interp_index.to(device)
    lut_long = lut_long.to(device)
    inter_value = (lut_long[lut_index + 1] - lut_long[lut_index]) * interp_index
    if is_float:
        inter_value = inter_value / (2 ** ibits)
    else:
        inter_value = inter_value >> ibits
    out = lut_long[lut_index] + inter_value
    del lut_long
    return out


def D_UTIL_norm_l(L_var1):
    L_var1 = L_var1.copy()
    var_out = np.zeros_like(L_var1, dtype='int32')
    for k in range(0, 30):
        indices = np.where(L_var1 < 0x40000000)
        L_var1[indices] = (L_var1[indices] << 1)
        var_out[indices] += 1
    var_out = 31 - var_out
    return var_out, L_var1


def D_UTIL_normalised_inverse_sqrt(lut, frac, exp):
    D_ROM_isqrt = np.int64(lut)
    frac = np.int64(frac)

    indices = np.where((exp & 0x1) == 1)
    frac[indices] = (frac[indices] >> 1)

    exp = ((exp - 1) >> 1)
    frac = frac >> 7
    i = frac >> 16
    i = i - 64

    frac = frac >> 1
    a = frac
    a = a & 0x7fff

    frac = D_ROM_isqrt[i]
    frac = frac << 16
    tmp = D_ROM_isqrt[i] - D_ROM_isqrt[i + 1]
    frac = frac - ((tmp * a) << 1)
    return frac, exp


def D_UTIL_inverse_sqrt(lut, L_x):
    exp, L_x = D_UTIL_norm_l(L_x)
    L_x, exp = D_UTIL_normalised_inverse_sqrt(lut, L_x, exp)
    L_y = L_x >> (exp)
    return L_y


def rsqrt(lut, x):
    '''
    Calculate square root
    '''
    t = np.int32(D_UTIL_inverse_sqrt(lut, x) / 0x10000)
    return t


def tpc_bias_compact(original, max_shift, pre_shift, compact_bit):
    """
    Compressed bias only retains the valid high compact_bit(31) bits
    :param original:
    :param max_shift:
    :param pre_shift:
    :param compact_bit:
    :return:
    """
    pre_shift = max(0, pre_shift)
    shift = 0
    d = np.int64(original.copy().flatten())
    elt_cnt = len(d)
    max_v = np.iinfo(np.int64).min
    max_ori = np.iinfo(np.int64).min
    min_ori = np.iinfo(np.int64).max
    for i in range(elt_cnt):
        tmp = d[i]
        if tmp > max_v:
            max_v = abs(d[i])
    max_ori = max(d)
    min_ori = min(d)
    max_compact = (1 << compact_bit) - 1
    min_compact = - (1 << compact_bit)
    if max_ori > max_compact or min_ori < min_compact:
        shift = int(np.round(np.log2(max_v + 0.5)))
        if shift == compact_bit:
            dd = np.clip(d >> pre_shift, min_compact, max_compact)
        elif shift >= max_shift:
            shift = max_shift
            dd = np.clip(d >> (shift + pre_shift - compact_bit), min_compact, max_compact)
            print('[W] the scope of the input bias is too big')
        else:
            dd = np.clip(d >> (shift + pre_shift - compact_bit), min_compact, max_compact)
        shift -= compact_bit
    else:
        dd = np.clip(d >> pre_shift, min_compact, max_compact)
    return dd, shift


def get_scale_approximation_params(fp32_scale_value, mult_bits, limit=False, mult_bits_ceil=15, shift_bits_ceil=31, force_shift_positive=False):
    fp32_scale = fp32_scale_value if isinstance(
        fp32_scale_value, torch.Tensor) else torch.tensor(fp32_scale_value, dtype=torch.float32)
    mbits = mult_bits if isinstance(mult_bits, torch.Tensor) else torch.tensor(mult_bits)
    # AIFF max support multiplier value is 32767, so we limit mbits to less equal than 15
    mbits = torch.minimum(mult_bits_ceil*torch.ones_like(mbits), mbits)
    shift_bits = torch.log2((2.0 ** mbits - 1.0) / fp32_scale).floor()
    if limit:
        shift_bits = torch.minimum(mbits, shift_bits)
    # AIFF max support shift value is 31, so we limit shift to less equal than 31
    shift_bits = torch.minimum(shift_bits, torch.ones_like(shift_bits) * shift_bits_ceil)
    multiplier = (fp32_scale * (2.0 ** shift_bits) + 0.5).floor()
    if force_shift_positive:
        shift_less0_mask = shift_bits < 0
        shift_bits[shift_less0_mask] = 0
        multiplier[shift_less0_mask] = min(2.0**mult_bits-1.0, 2.0**mult_bits_ceil-1.0)
    q_bits = 8 if mult_bits <= 8 else 16
    multiplier_type = bits2dtype(q_bits, is_signed=False)
    _, shiftbits_type = range2dtype(shift_bits.min(), shift_bits.max(), force_int=True)
    if (multiplier - 2.0**shift_bits).abs().max().item() < torch.finfo(torch.float32).eps:
        multiplier = torch.ones_like(multiplier)
        shift_bits = torch.zeros_like(shift_bits)
    if fp32_scale.dim() < 1:
        return multiplier.item(), multiplier_type, shift_bits.item(), shiftbits_type
    else:
        return multiplier, multiplier_type, shift_bits, shiftbits_type


# calculate 1/sqrt(x)
def isqrt_forward(x):
    xq = x.to(torch.int64)
    overflow_bits = torch.zeros_like(x).int()
    for k in range(31, 64):
        overflow_bits = torch.where((xq >> k) == 0, overflow_bits, overflow_bits + 1)
    overflow_bits = torch.where((overflow_bits % 2).to(torch.bool), overflow_bits + 1, overflow_bits)
    yq = calculate_inverse_sqrt(xq >> overflow_bits).to(torch.int64)
    rsqnorm = yq.long()
    rsqnorm_bits = torch.ceil(torch.log2(rsqnorm)).int()
    rsqnorm_limit_bits = 15
    rsqnorm_shift = torch.where(rsqnorm_bits > rsqnorm_limit_bits, rsqnorm_bits -
                                rsqnorm_limit_bits, torch.zeros_like(rsqnorm_bits, device=rsqnorm_bits.device).int())
    rsqnorm = torch.clamp(rsqnorm >> rsqnorm_shift, 0, 2**rsqnorm_limit_bits-1)
    return rsqnorm, rsqnorm_shift - torch.div(overflow_bits, 2, rounding_mode='trunc')


def calculate_inverse_sqrt(x):
    inverse_sqrt_table = [32767, 32515, 32268, 32026, 31790, 31558, 31332, 31111, 30894,
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
                          16481, 16448, 16416, 16384]

    def normalize_to31bit(var):
        var = var.to(torch.int64)
        var_out = torch.zeros_like(var)
        for k in range(0, 30):
            cond = var < 0x40000000
            var = torch.where(cond, var << 1, var)
            var_out = torch.where(cond, var_out + 1, var_out)
        var_out = 31 - var_out
        return var_out, var

    def normalised_inverse_sqrt(frac, exp):
        frac = torch.where(exp & 0x1 == 1, frac >> 1, frac)
        exp = (exp-1) >> 1
        frac = frac >> 7
        i = (frac >> 16) - 64
        i = torch.maximum(i, torch.zeros_like(i))
        frac = frac >> 1
        a = frac & 0x7fff
        isqrt = torch.tensor(inverse_sqrt_table, dtype=torch.int64, device=frac.device)
        tmp1 = torch.gather(isqrt, 0, i.reshape(-1))
        frac = tmp1 << 16
        tmp = tmp1 - torch.gather(isqrt, 0, (i+1).reshape(-1))
        frac = frac - (torch.multiply(tmp, a) << 1)
        return frac, exp
    vshape = x.shape
    vtype = x.dtype
    exp, x = normalize_to31bit(x.reshape(-1))
    x, exp = normalised_inverse_sqrt(x, exp)
    y = x >> exp
    return y.reshape(vshape).to(vtype)


def fp_activation(data, layer):
    data = data.to(torch.float32)
    if 'with_activation' in layer.params.keys():
        if layer.params['with_activation'].upper() == 'RELU':
            data = torch.nn.functional.relu(data)
        elif layer.params['with_activation'].upper() == 'RELU6':
            data = torch.nn.functional.relu6(data)
        elif layer.params['with_activation'].upper() == 'NONE':
            data = data
        elif layer.params['with_activation'].upper() == 'CLIP':
            data = torch.clamp(data, layer.params['clip_min'], layer.params['clip_max'])
        elif layer.params['with_activation'].upper() == 'LEAKYRELU':
            data = torch.nn.functional.leaky_relu(data, layer.params['negative_slope_value'])
        elif layer.params['with_activation'].upper() == 'PRELU':
            # Cannot use torch.nn.functional.prelu:
            #   torch is NCHW and 1D negative_slope should be same with C
            #   but Eltwise in aipu is NHWC and in torch is still NCHW
            #   we cannot swap output.permute and fp_activation in op using fp_activation, otherwise Eltwise may raise error
            negative_slope = layer.constants['negative_slope'].betensor
            for i in range(0, data.shape[-1]):
                outputs_c = data[..., i]
                outputs_c[outputs_c < 0] = outputs_c[outputs_c < 0] * negative_slope[i]
    return data


def lookup_lut_powerof2(inputs, lut, lut_in_bits, in_is_signed, lut_out_bits, out_is_signed, align_mode=None):
    """the lut is size of 2^N, use the inputs high Nbits as the lut index and the low bits for interplate.
    :param inputs: inputs data as lut indexes
    :param lut: lookup table
    :param lut_in_bits: indexes range used bits
    :param in_is_signed: indexes sign
    :param lut_out_bits: lut value range used bits
    :param out_is_signed: lut value sign
    :param align_mode: right align and left align
    :return:
    """
    from AIPUBuilder.logger import DEBUG, FATAL
    import math
    table_size = lut.numel()
    table_size_bits = int(math.log2(table_size))
    if 2 ** table_size_bits != table_size:
        FATAL('lookup_lut_powerof2 function can only support lut which has 2**N items.')

    in_qmin, _ = bits2range(lut_in_bits, in_is_signed)
    out_qmin, out_qmax = bits2range(lut_out_bits, out_is_signed)
    lutp = torch.nn.functional.pad(lut, (0, 1), value=lut[-1])
    lut_long = lutp.to(torch.long)
    inputs_long = inputs.to(torch.long)
    indexes = (inputs_long - in_qmin)
    ibits = 0
    lut_index = indexes
    interp_index = 0
    if lut_in_bits > table_size_bits:
        ibits = lut_in_bits - table_size_bits
        if align_mode is not None:
            if align_mode == 'right_align':
                diff = indexes.max() - (table_size - 1)
                lidx = torch.clamp(indexes - diff, 0, table_size - 1)
                lut_index = lidx << ibits
            elif align_mode == 'left_align':
                diff = indexes.min() - 0
                lidx = torch.clamp(indexes - diff, 0, table_size - 1)
                lut_index = lidx << ibits
            else:
                DEBUG("lookup_lut_power2 align_mode=default.")
        lut_index = (indexes >> ibits).long()
        interp_index = indexes - (lut_index << ibits)
    inter_value = (lut_long[lut_index+1] - lut_long[lut_index]) * interp_index
    inter_value = inter_value >> ibits
    out = lut_long[lut_index] + inter_value
    out = torch.clamp(out, out_qmin, out_qmax).long()
    return out

def linear_dequantize(x, scale, zero_point, key_axis=None):
    dev = x.device if isinstance(x, torch.Tensor) else None
    x_t, scale_t, zero_point_t = batch_construct_torch_tensor([x, scale, zero_point], device=dev)
    if key_axis is not None:
        scale_shape = [-1 if s == key_axis else 1 for s in range(x_t.dim())]
        scale_t = scale_t.reshape(scale_shape)
        zero_point_t = zero_point_t.reshape(scale_shape)
    return (x_t + zero_point_t) * (1.0 / scale_t)

def batch_construct_torch_tensor(vars, device=None):
    if not isinstance(vars, (list, tuple)):
        vars = [vars]
    constructed_t = []
    for var in vars:
        constructed_t.append(construct_torch_tensor(var, device=device))
    return constructed_t[0] if len(constructed_t) == 1 else constructed_t


def is_torch_tensor(var):
    return True if isinstance(var, torch.Tensor) else False

def construct_torch_tensor(var, device=None):
    dev = device if device is not None else "cpu"
    if var is None:
        constructed_t = var
    elif is_torch_tensor(var):
        constructed_t = var.to(device=dev)
    else:
        constructed_t = torch.tensor(var, device=dev)
    return constructed_t
