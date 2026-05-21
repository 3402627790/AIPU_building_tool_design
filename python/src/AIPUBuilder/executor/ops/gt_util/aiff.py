import torch
import numpy as np
import math
from AIPUBuilder.core import OpType, Target, BuilderConfig, Dtype, ISAType
from AIPUBuilder.logger import DEBUG
from .math_utils import fp32_to_fp24

numpy_to_torch_dtype_map = {
    'uint16': 'int32',
    'uint32': 'int64',
}


def wrap_for_np(func):
    def f(*args, **kwargs):
        arg = []
        is_np = False
        for a in args:
            if isinstance(a, np.ndarray):
                is_np = True
                a = torch.tensor(a) if str(a.dtype) not in numpy_to_torch_dtype_map.keys() else torch.tensor(
                    a.astype(numpy_to_torch_dtype_map[str(a.dtype)]))
            arg.append(a)
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                is_np = True
                kwargs[k] = torch.tensor(v) if str(v.dtype) not in numpy_to_torch_dtype_map.keys() else torch.tensor(
                    v.astype(numpy_to_torch_dtype_map[str(v.dtype)]))
        args = tuple(arg)
        r = func(*args, **kwargs)
        if is_np:
            if isinstance(r, tuple):
                return tuple([a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a for a in r])
            if isinstance(r, torch.Tensor):
                return r.detach().cpu().numpy()
        return r
    return f


def parse_ops_options(ops_options):
    if ops_options != None and not isinstance(ops_options, dict):
        tmp = ops_options.split(',')
        ops_options = {}
        for t in tmp:
            _tmp = t.split('=')
            ops_options[_tmp[0]] = _tmp[1]
    return ops_options


def get_ops_options(key, default=''):
    cfg = BuilderConfig
    if "op_plugin_options" in cfg.keys():
        v = cfg.get("op_plugin_options")
        vs = parse_ops_options(v)
        if key in vs:
            return vs[key]
        return default
    return default


@wrap_for_np
def _aiff_compression_bias(a):
    b = a
    if max(b) > 32767 or min(b) < -32768:
        a_max = max(abs(b.long()))
        shift = torch.ceil(torch.log2(a_max.double() + 0.5)).to(torch.int32)
        shift -= 15
        if shift <= 0:
            return a
        b = torch.div(b, (2 ** shift), rounding_mode='floor')  # floor
        b = b * (2 ** shift)

    return b.long()


def _aiff_compression_bias_combin(a):
    is_32b_bias = True if get_ops_options("AIPU_32BIT_BIAS", '1') == '1' else False
    # 64bit biases still needs compression due to pre-shift when 16bit fmp x 16bit weight
    if Target().isa >= ISAType.ZYv5:
        # only X2 has 32bit ALU
        if a.dtype == torch.int32 and is_32b_bias:
            # print("setting 32bit biases, max {}, min {}".format(np.max(a), np.min(a)))
            return a.long()
        else:
            return _aiff_compression_bias(a)
    elif a.dtype in [torch.float16, torch.float32, torch.float64]:
        return a
    else:
        return _aiff_compression_bias(a)


@wrap_for_np
def _aiff_merge_shifts(scales, shifts):
    scales = scales.long()
    shifts = shifts.long()  # if shifts is 8bit, 2**shifts is still 8bit thus overflow
    m_shift = shifts.max()
    shifts = m_shift - shifts
    s = scales * (2**shifts)
    while s.max() > (1 << 15) - 1:
        s = s >> 1
        m_shift -= 1
    return s, m_shift


def _aiif_ahead_shift_biases(self, data, origin_shift, biases=None):
    '''

    :param self:
    :param data: the middle data out from aiff mtp
    :param origin_shift:
    :param biases:
    :return: the data added bias and the remain shift
    '''
    i_type = self.inputs[0].dtype
    if self.type not in [OpType.MatMul, OpType.Gemm]:
        w_type = self.constants['weights'].dtype
    else:
        w_type = self.inputs[1].dtype

    remain_shift = origin_shift
    remain_shift_tmp = origin_shift
    if i_type.is_int and w_type.is_int:
        # fp or int4/int12 does not need pre-shift up to now
        if i_type.bits == 16 and w_type.bits == 16:
            remain_shift_tmp = self.params.get('remain_shift', 20)  # 16
        elif (i_type.bits == 16 and w_type.bits == 8) or (w_type.bits == 16 and i_type.bits == 8):
            remain_shift_tmp = self.params.get('remain_shift', 10)  # 8
        if origin_shift - remain_shift_tmp >= 0:
            data = torch.round(data.double() * (0.5 ** (origin_shift - remain_shift_tmp))).long()
            if not biases is None:
                biases = biases >> (origin_shift - remain_shift_tmp)
                if not isinstance(biases, int):
                    biases = biases.long()
            remain_shift = remain_shift_tmp
    data = clip_to_bit(data, 32, True, 'int64').cpu()
    if not biases is None:
        data += biases.to(data.device)
    return data, remain_shift


def _aiif_ahead_shift_biases_for_merge(self, data0, data1, origin_shift, biases):
    i_type = self.inputs[0].dtype
    if self.type not in [OpType.MatMul, OpType.Gemm]:
        w_type = self.constants['weights'].dtype
    else:
        w_type = self.inputs[1].dtype

    remain_shift = origin_shift
    remain_shift_tmp = origin_shift
    if i_type.is_int and w_type.is_int:
        if i_type.bits == 16 and w_type.bits == 16:
            remain_shift_tmp = self.params.get('remain_shift', 20)  # 16
        elif (i_type.bits == 16 and w_type.bits == 8) or (w_type.bits == 16 and i_type.bits == 8):
            remain_shift_tmp = self.params.get('remain_shift', 10)  # 8
        if origin_shift - remain_shift_tmp >= 0:
            data0 = torch.round(data0.double() *
                                (0.5 ** (origin_shift - remain_shift_tmp)))
            data1 = torch.round(data1.double() *
                                (0.5 ** (origin_shift - remain_shift_tmp)))
            biases = biases >> (origin_shift - remain_shift_tmp)
            remain_shift = remain_shift_tmp

    data0 = clip_to_bit(data0, 32, True, 'int64')
    data1 = clip_to_bit(data1, 32, True, 'int64')

    if not self.add_bias_after_merge:
        data1 += biases

    data1 = clip_to_bit(data1, 32, True, 'int64')
    data1 += data0
    data1 = clip_to_bit(data1, 32, True, 'int64')

    if self.add_bias_after_merge:
        data1 += biases
        data1 = clip_to_bit(data1, 33, True, 'int64')

    return data1, remain_shift


def _aiff_inner_activation(layer, data: np.ndarray, scale_value, shift_value, using_ptp=False):
    if isinstance(scale_value, list):
        scale_value = [s.item() for s in scale_value]

    if 'with_activation' in layer.params.keys():
        with_activation = layer.params['with_activation']
    else:
        with_activation = 'None'

    if with_activation.upper() in ['RELU', 'RELU6']:
        data, final_scale, final_shift = inner_relu(
            layer, data, scale_value, shift_value, using_ptp)
    elif with_activation.upper() == 'LEAKYRELU':
        # leaky ReLU / per layer
        data, final_scale, final_shift = inner_leakyrelu(
            layer, data, scale_value, shift_value, using_ptp=False)
    elif with_activation.upper() == 'PRELU':
        # PReLU / per channel
        data, final_scale, final_shift = inner_prelu(
            layer, data, scale_value, shift_value, using_ptp=False)
    elif with_activation.upper() == 'CLIP':
        data, final_scale, final_shift = inner_clip(
            layer, data, scale_value, shift_value, using_ptp)
    elif with_activation.upper() == 'NONE':
        final_scale = scale_value
        final_shift = shift_value
        if not using_ptp:
            data = clip_to_bit(data, 33, True)
    return data, final_scale, final_shift


@wrap_for_np
def inner_relu(layer, data: np.ndarray, scale_value, shift_value, using_ptp=False):
    if not using_ptp:
        data = clip_to_bit(data, 33, sign=True)
    data[data < 0] = 0
    return data, scale_value, shift_value


def inner_leakyrelu(layer, data: np.ndarray, scale_value, shift_value, using_ptp=False):
    negative_slope_shift = layer.outputs[0].dtype.bits
    if 'negative_slope_shift' in layer.params.keys():
        negative_slope_shift = layer.params['negative_slope_shift']
    if 'negative_slope' in layer.constants.keys():
        data[data < 0] = torch.round(data[data < 0].double() * layer.constants['negative_slope'] * (0.5 ** negative_slope_shift)).\
            type(data.dtype)
    else:
        data[data < 0] = torch.round(data[data < 0].double() * layer.params['negative_slope_value'] * (0.5 ** negative_slope_shift)).\
            type(data.dtype)
    data = clip_to_bit(data, 32, True)

    return data, scale_value, shift_value


def inner_prelu(layer, data: np.ndarray, scale_value, shift_value, using_ptp=False):
    negative_slope = layer.constants['negative_slope'].betensor
    for i in range(0, data.shape[-1]):
        outputs_c = data[..., i]
        outputs_c[outputs_c < 0] = torch.round(
            outputs_c[outputs_c < 0].double() * negative_slope[i] * (0.5 ** layer.params['negative_slope_shift'])).\
            type(outputs_c.dtype)
    data = clip_to_bit(data, 32, True)

    return data, scale_value, shift_value


def inner_clip(layer, data: np.ndarray, scale_value, shift_value, using_ptp=False):
    final_scale = 1
    o_type = str(layer.outputs[0].dtype)
    if o_type in ['int16', 'uint16']:
        final_shift = 0
        data = torch.round(data.double() * torch.tensor(scale_value).expand(data.shape)
                           * (0.5 ** shift_value)).type(data.dtype)
    else:
        final_shift = 8
        if shift_value >= final_shift:
            data = torch.round(data.double() * torch.tensor(scale_value).expand(data.shape)
                               * 0.5 ** (shift_value - final_shift)).type(data.dtype)
        else:
            if isinstance(scale_value, int):
                scale_value = scale_value * 2 ** (final_shift - shift_value)
                while scale_value > (0x8000 - 1):
                    final_shift -= 1
                    scale_value = scale_value * 0.5
            else:
                # per-channel
                scale_value = [s * (1 << (final_shift - shift_value))
                               for s in scale_value]
                max_scale = max(scale_value)
                ss = 0
                while max_scale > (0x8000 - 1):
                    ss += 1
                    max_scale = max_scale * 0.5
                scale_value = [s >> ss for s in scale_value]
                final_shift = final_shift - ss
            data = (data.double() * scale_value).type(data.dtype)
    data = torch.clip(data, layer.params['clip_min'], layer.params['clip_max'])
    return data, final_scale, final_shift


@wrap_for_np
def _aiff_final_output(layer, data: np.ndarray, final_scale=None, final_shift=None, fuse=False):
    legacy_asym_output = False
    if 'asym_ocvt_bias' in layer.sgattrs and layer.sgattrs['asym_ocvt_bias']:
        legacy_asym_output = True
    zp = torch.tensor(layer.outputs[0].quantization.offset).to(data.device)
    if isinstance(final_scale, torch.Tensor):
        final_scale = final_scale.to(data.device)
    if isinstance(final_shift, torch.Tensor):
        final_shift = final_shift.to(data.device)
    target = Target().name.upper()

    if fuse:
        data = torch.round(data.double() * final_scale *
                           (0.5 ** final_shift)).long()
        data -= zp
    else:
        if target in ['Z2_1104', 'Z2_1002', 'Z2_0901', 'Z3_1104', 'Z3_1002', 'Z3_0901', 'X1_1204'] and legacy_asym_output:
            # print("%s using aipu asym output method!!!" % layer.name)
            zp = zp * 2 ** final_shift
            if isinstance(final_scale, int):
                zp = torch.round(zp / torch.tensor(final_scale).double()).long()
            else:
                zp = torch.round(zp / final_scale.double()).long()
            zp = torch.clip(zp, -2147483648, 2147483647).to(torch.int32)
            data -= zp
            data = torch.round(data.double() * final_scale *
                               (0.5 ** final_shift)).long()
        else:
            data = torch.round(data.double() * final_scale *
                               (0.5 ** final_shift)).long()
            if target in ['Z1_0904', 'Z1_1002', 'Z1_0701', 'Z1_0701_P'] and zp != 0:
                raise Exception("%s does not support asym quant!" % (target))
            else:
                data -= zp

    return data


@wrap_for_np
def clip_to_bit(data, bit_width, sign=True, to_type=None):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    dtype = Dtype.get(data.dtype)
    if dtype.bits >= bit_width or dtype.is_float:
        data = torch.clamp(data, -2 ** (bit_width - 1), 2 ** (bit_width - 1) - 1) if sign \
            else torch.clamp(data, 0, 2 ** bit_width - 1)
    if to_type is not None:
        data = data.to(Dtype.get(to_type).torch)
    return data.double()


def use_ptp_or_mtp(layer, target_name, fuse=False):
    kernel_x = layer.params['kernel_x']
    kernel_y = layer.params['kernel_y']
    stride_x = layer.params['stride_x']
    stride_y = layer.params['stride_y']
    dilation_x = layer.params['dilation_x']
    dilation_y = layer.params['dilation_y']
    ic = layer.inputs[0].shape.c_total
    # NHWC is c, NCHWCx is C * Cx, NCHWCxTx is C * Cx + Tx
    oc = layer.outputs[0].shape.c_total
    group = layer.params['group']
    with_activation = layer.params['with_activation'].upper()

    if layer.inputs[0].dtype.bits == layer.outputs[0].dtype.bits:
        mixed_pre = False
    else:
        mixed_pre = True

    ptp_not_sopport_relu = False
    output_zp = True if layer.outputs[0].quantization.offset != 0 else False
    if with_activation == 'CLIP':
        ptp_not_sopport_relu = True
        output_type = layer.outputs[0].dtype.np
        clip_min = layer.params['clip_min'] if 'clip_min' in layer.params else -32768
        clip_max = layer.params['clip_max'] if 'clip_max' in layer.params else 32767
        if output_zp and clip_min == np.iinfo(output_type).min and clip_max == np.iinfo(output_type).max:
            ptp_not_sopport_relu = False
        if not output_zp and clip_min == np.iinfo(output_type).min and clip_max == np.iinfo(output_type).max and layer.outputs[0].dtype.bits == 16:
            ptp_not_sopport_relu = False
    ptp_not_sopport_relu = True if with_activation == 'LEAKYRELU' else ptp_not_sopport_relu
    ptp_not_sopport_relu = True if with_activation == 'PRELU' else ptp_not_sopport_relu

    if target_name.upper() in ['Z3_0901', 'Z3_1002', 'Z2_0901', "Z2_1002"] and not mixed_pre\
            and not ptp_not_sopport_relu and not output_zp:
        if kernel_x == 3 and kernel_y == 3 and dilation_x == dilation_y and \
                stride_x != 3 and stride_y != 3 and oc == group and \
                (not fuse or layer.children[0].type == OpType.Convolution) and \
                layer.inputs[0].shape.c_total <= 2048 and \
                layer.outputs[0].shape.c_total <= 2048:
            # just single dwconv or dwconv_pwconv or dwconv_pwconv_eltwise in specific target and limited spec
            use_ptp = True
        else:
            use_ptp = False
    else:
        use_ptp = False

    return use_ptp


def _aiff_lut(input_data: np.ndarray, lut_data, input_type=None, target=None, ignore_sp_target=False):
    '''
        aiff lut uinify function,follow the action of itp.
    :param input_data: input to index the lut
    :param lut_data:
    :param target: Z1_0701_P will has little difference
    :return:
    '''

    filter_list = ['Z1_0701', 'Z1_0701_P']
    Global_SDP_LUT_BIAS = 0
    Global_SDP_LUT_PRSH = 0
    Global_SDP_LUT_MAX_SLOPE = 0
    Global_SDP_LUT_MIN_SLOPE = 0
    one_64b = 1
    max_slope_8b = Global_SDP_LUT_MAX_SLOPE
    min_slope_8b = Global_SDP_LUT_MIN_SLOPE

    if lut_data.size < 256:
        print('Warning!!! LUT size must be no less than 256')
        return np.array([], dtype=lut_data.dtype)
    if lut_data.size == 256:
        lut_data = np.append(lut_data, lut_data[255])

    din = np.zeros_like(input_data, dtype='int64')
    input_type = input_type if input_type is not None else input_data.dtype.name
    if input_type == 'int8':
        din = (np.int64(input_data) + 128) * 256
    elif input_type == 'int16':
        din = np.int64(input_data) + 32768
    elif input_type == 'uint8':
        din = np.int64(input_data) * 256
    elif input_type == 'uint16':
        din = np.int64(input_data)

    lut_sub = np.int64(din) - np.int64(Global_SDP_LUT_BIAS)
    lut_rsh = lut_sub >> Global_SDP_LUT_PRSH
    pos_out_of_range = lut_rsh > ((one_64b << 16) - 1)
    neg_out_of_range = lut_rsh < 0

    if target and target.upper() in filter_list and not ignore_sp_target:
        lut_idx = (lut_rsh >> 8) & 0xfe
        lut_delta = (lut_rsh & 0x1fe) >> 1
    else:
        lut_idx = (lut_rsh >> 8) & 0xff
        lut_delta = lut_rsh & 0xff

    pos_out_of_range_clamp_en = (
        lut_rsh - (one_64b << 16)) > ((one_64b << 16) - 1)
    neg_out_of_range_clamp_en = lut_rsh < (0 - ((one_64b << 16) - 1))
    pos_out_of_range_delta = np.zeros(shape=input_data.shape, dtype='int64')
    pos_out_of_range_delta[pos_out_of_range_clamp_en == True] = (
        one_64b << 16) - 1
    pos_out_of_range_delta[pos_out_of_range_clamp_en == False] = lut_rsh[pos_out_of_range_clamp_en == False] - (
        one_64b << 16)

    neg_out_of_range_delta = np.zeros(shape=input_data.shape, dtype='int64')
    neg_out_of_range_delta[neg_out_of_range_clamp_en == True] = (
        one_64b << 16) - 1
    neg_out_of_range_delta[neg_out_of_range_clamp_en == False] = lut_rsh[neg_out_of_range_clamp_en == False] - (
        one_64b << 16)

    dout = np.zeros(shape=input_data.shape, dtype='int64')
    dout[pos_out_of_range == True] = pos_out_of_range_delta[pos_out_of_range ==
                                                            True] * max_slope_8b + lut_data[256]
    dout[neg_out_of_range == True] = neg_out_of_range_delta[neg_out_of_range ==
                                                            True] * max_slope_8b + lut_data[0]
    if target and target.upper() in filter_list and not ignore_sp_target:
        dout[(pos_out_of_range == False) & (neg_out_of_range == False)] = \
            (np.int64(lut_data[lut_idx[(pos_out_of_range == False) & (neg_out_of_range == False)] + 2]) -
             np.int64(lut_data[lut_idx[(pos_out_of_range == False) & (neg_out_of_range == False)]])) * \
            lut_delta[(pos_out_of_range == False) & (neg_out_of_range == False)] + \
            (np.int64(lut_data[lut_idx[(pos_out_of_range == False) & (
                neg_out_of_range == False)]]) << 8)
        dout[(pos_out_of_range == False) & (neg_out_of_range == False)] >>= 8

    else:
        dout[(pos_out_of_range == False) & (neg_out_of_range == False)] = \
            (np.int64(lut_data[lut_idx[(pos_out_of_range == False) & (neg_out_of_range == False)] + 1]) -
             np.int64(lut_data[lut_idx[(pos_out_of_range == False) & (neg_out_of_range == False)]])) * \
            lut_delta[(pos_out_of_range == False) & (neg_out_of_range == False)] + \
            (np.int64(lut_data[lut_idx[(pos_out_of_range == False) & (neg_out_of_range == False)]]) << 8)

        dout[(pos_out_of_range == False) & (neg_out_of_range == False)] >>= 8

    output = dout.astype(lut_data.dtype.name)
    return output


def c_round(x):
    # Similar to ROUND_HALF_UP in Decimal
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    def c_round_lambda(z): return (int(z > 0) - int(z < 0)) * int(abs(z) + 0.5)
    if isinstance(x, (np.ndarray, np.generic)):
        return np.vectorize(c_round_lambda)(x)
    else:
        return c_round_lambda(x)


def to_int16_data(x, shift):
    return np.clip(c_round(x * (0.5 ** shift)), np.iinfo(np.int16).min, np.iinfo(np.int16).max)


def hwa_gru_to_int16(old_biases, pre_shift=0):
    pre_shift = max(0, pre_shift)
    # abs_biases = [abs(i) for i in list(old_biases[0])]
    abs_biases = np.abs(old_biases)
    max_abs_val = np.max(abs_biases)
    if max_abs_val == 0:
        return old_biases, 0
    bits = c_round(math.log2(max_abs_val) + 0.5) + 1
    if bits - pre_shift > 16:
        new_biases = to_int16_data(old_biases, bits - 16)
        shift = bits - pre_shift - 16
    else:
        new_biases = [to_int16_data(i, pre_shift) for i in list(old_biases[0])]
        shift = 0
    return new_biases, shift


def hwa_int32_to_int16_with_round(array, maxlshift=31, pre_shift=0):
    import math
    input_dtype = array.dtype
    max_ori = max(list(array))
    min_ori = min(list(array))
    abs_ = [abs(i) for i in list(array)]
    max_ = max(abs_)
    shift = 0
    # clamp_s16

    def clamp(x):
        if x > 0x7fff:
            x = 0x7fff
        elif x < (-0x7fff - 1):
            x = (-0x7fff - 1)
        return x

    if max_ori > 0x7fff or min_ori < (-0x7fff - 1):
        shift = round(math.log2(max_) + 0.5)
        if shift == 15:
            array = [clamp(i >> pre_shift) for i in array]
        elif shift >= (maxlshift + 15):
            shift = maxlshift + 15
            array = [clamp(round(i >> (shift + pre_shift - 15)))
                     for i in array]
        else:
            array = [clamp(round(i >> (shift + pre_shift - 15)))
                     for i in array]
        shift -= 15
    else:
        array = [clamp(round(i >> (pre_shift))) for i in array]
    return np.array(array, dtype=input_dtype), shift


def lstm_bias_int32_to_int16_with_round(array, maxlshift=31, pre_shift=0, pre_shift_before=False):
    import math
    input_dtype = array.dtype
    max_ori = max(list(array))
    min_ori = min(list(array))
    abs_ = [abs(i) for i in list(array)]
    max_ = max(abs_)
    shift = 0
    # clamp_s16

    def clamp(x):
        if x > 0x7fff:
            x = 0x7fff
        elif x < (-0x7fff - 1):
            x = (-0x7fff - 1)
        return x

    if max_ori > 0x7fff or min_ori < (-0x7fff - 1):
        shift = round(math.log2(max_) + 0.5)
        if pre_shift_before:
            shift = max(shift - pre_shift, 0)
        if shift <= 15:
            array = [clamp(i >> pre_shift) for i in array]
        elif shift >= (maxlshift + 15):
            shift = maxlshift + 15
            array = [clamp(round(i >> (shift + pre_shift - 15)))
                     for i in array]
        else:
            array = [clamp(round(i >> (shift + pre_shift - 15)))
                     for i in array]
        shift = max(shift - 15, 0)
    else:
        array = [clamp(round(i >> (pre_shift))) for i in array]
    return np.array(array, dtype=input_dtype), shift


def pad_input(self, input_tensor):
    pad_left = self.params['pad_left']
    pad_right = self.params['pad_right']
    pad_top = self.params['pad_top']
    pad_bottom = self.params['pad_bottom']
    input_tensor = input_tensor.numpy()
    constant_value = 0 - \
        self.params['layer_top_zp'] if 'layer_top_zp' in self.params else 0
    # if 'conv_method' in self.params and self.params['conv_method'] == 'winograd_conv':
    #     bottom_n, bottom_h, bottom_w, bottom_c = input_tensor.shape
    #     # top_c = self.layer_top_shape[0][3]

    #     pr_ = pad_right + ((bottom_w + pad_left + pad_right) % 2)  #
    #     pb_ = pad_bottom
    #     ow_ = (bottom_w - 3 + pad_left + pr_) // 1 + 1
    #     oh_ = (bottom_h - 3 + pad_top + pb_) // 1 + 1
    #     loop_w = ow_ // (3 - 1)
    #     loop_h = oh_

    #     input_tensor = np.pad(input_tensor, ((0, 0), (pad_top, pb_), (pad_left, pr_), (0, 0)), 'constant',
    #                 constant_values=(constant_value, constant_value))
    if 'pads_mode' in self.params:
        n, h, w, c = input_tensor.shape
        temp_tensor = np.zeros(
            [n, h + pad_top + pad_bottom, w + pad_left + pad_right, c], dtype=input_tensor.dtype)
        if self.params['pads_mode'] == 0:
            # per-layer pad
            constant_value = np.array([self.params['pads_value']] * c)
        elif self.params['pads_mode'] == 1:
            # per-channel pad
            constant_value = self.constants['pads'].betensor.numpy()
        else:
            raise Exception(f"{self.params['pads_mode']} not supported!!")
        for cc in range(c):
            temp_tensor[:, :, :, cc] = np.pad(input_tensor[:, :, :, cc], ((0, 0),
                                                                          (pad_top,
                                                                           pad_bottom),
                                                                          (pad_left, pad_right)), 'constant',
                                              constant_values=(constant_value[cc], constant_value[cc]))
        input_tensor = temp_tensor.copy()
    else:
        input_tensor = np.pad(input_tensor,
                              ((0, 0),
                               (pad_top, pad_bottom),
                                  (pad_left, pad_right),
                                  (0, 0)),
                              mode='constant',
                              constant_values=(constant_value, constant_value))
    return torch.Tensor(input_tensor)


def aiff_itp_acc_channel(data: torch.Tensor):
    # data = data.numpy()
    c = data.shape[-1]
    c16_tail = 16 - c % 16
    data = torch.nn.functional.pad(data, [0, c16_tail])

    for j in range(4):
        c = data.shape[-1]
        sum_8_list = []
        for i in range(0, c, 2):
            sum_8_list.append(torch.sum(data[..., i:i+2], axis=-1, keepdims=True))
        data = torch.cat(sum_8_list, dim=-1)
    data = torch.sum(data, axis=-1, keepdims=True)
    return data


def _aiff_lookup_float_index_lut(
        t,
        lut,
        index_scale,
        index_offset,
        mirror_mode=False,
        value_offset_for_mirror_mode=0.0,
        is_trunc=False):

    def lookup_lut(x):
        y = x
        table = fp32_to_fp24(lut).flatten()
        table_size = table.numel()
        left_slope = fp32_to_fp24(table[1] - table[0])
        right_slope = fp32_to_fp24(table[-1] - table[-2])
        left_mask = x < 0
        right_mask = x > (table_size - 1)

        y_left = torch.where(left_mask, table[0] - left_slope * (0 - x), x)
        y_right = torch.where(right_mask, table[-1] + right_slope * (x - (table_size - 1)), x)
        index = torch.clamp(x, 0, table_size - 1).floor().long()
        # if nan in index, nan is clamp to -9223372036854775808, it is out of index size
        index = torch.clamp(index, 0, table_size - 1)
        lutp = torch.nn.functional.pad(table, (0, 1), value=table[-1])
        delta = torch.round((x - x.int()) * (2 ** 16))
        y_middle = lutp[index] + fp32_to_fp24(fp32_to_fp24(lutp[index + 1] - lutp[index]) * delta) * (0.5 ** 16)
        y_middle = fp32_to_fp24(y_middle)
        y = torch.where(left_mask, y_left, y_middle)
        y = torch.where(right_mask, y_right, y)

        return y

    t = fp32_to_fp24(t)
    index_scale = fp32_to_fp24(index_scale)
    index_offset = torch.tensor(index_offset, dtype=torch.float32)
    value_offset_for_mirror_mode = torch.tensor(value_offset_for_mirror_mode, dtype=torch.float32)
    tt = fp32_to_fp24(t * index_scale)
    # swap lut/e_pipe
    x_t = fp32_to_fp24(tt - index_offset, True)

    if mirror_mode:
        x_t = torch.clamp(x_t, -1023.0, 1023.0)
        y_pos = lookup_lut(x_t)
        y_neg = lookup_lut(x_t.negative()).negative()
        lut_out = fp32_to_fp24(torch.where(x_t >= 0.0, y_pos, y_neg))
    else:
        x_t = torch.clamp(x_t, -512.0, 1023.0)
        lut_out = fp32_to_fp24(lookup_lut(x_t))
    if is_trunc:
        # swap e_alu/e_mul
        out = fp32_to_fp24(lut_out - value_offset_for_mirror_mode, True)
    else:
        out = lut_out - value_offset_for_mirror_mode
    return out


def _x3_aiff_exp_approximation(f_vdata: torch.Tensor, pow2_f_lut: torch.Tensor) -> torch.Tensor:
    # hardware clamp
    f_vdata = fp32_to_fp24(f_vdata)
    f_vdata = torch.clamp(f_vdata, -128.0, 128.0)
    integer = f_vdata.floor()

    # get 16bit decimal, high 9 bits for index, low 7 bits extends to 16 bits for interpolation
    decimal = torch.round((f_vdata - integer) * (2 ** 16)).long()
    index = decimal >> 7
    delta = decimal & 0x7F
    delta <<= 9

    lutp = fp32_to_fp24(pow2_f_lut.to(torch.float32))
    upper_index = torch.clamp(index + 1, 0, 512)
    post_lut = lutp[index] + fp32_to_fp24(fp32_to_fp24(lutp[upper_index] - lutp[index]) * delta) * (0.5 ** 16)
    post_lut = fp32_to_fp24(post_lut)

    result = post_lut * (2.0 ** integer)
    result = fp32_to_fp24(result.to(torch.float32))

    return result


def x3_aiff_fp32_2_fpx(fp32_data, is_saturation=True, fpx=Dtype.FP16):
    DEBUG(f'target in x3_aiff_fp32_2_fpx: {Target().name.upper()}')  # debug CP-14921
    DEBUG(f'is_saturation: {is_saturation}, fpx: {str(fpx)}')
    assert isinstance(fpx, Dtype)
    is_scalar = False
    if isinstance(fp32_data, float):
        is_scalar = True
        fp32_data = torch.Tensor([fp32_data]).to(torch.float32)
    is_tensor = isinstance(fp32_data, torch.Tensor)
    fp32_data = torch.from_numpy(fp32_data) if not is_tensor else fp32_data

    if is_saturation and Target().isa == ISAType.ZYv6:
        from ml_dtypes import bfloat16, float8_e4m3fn, float8_e5m2
        max_value = 0
        if fpx.np == np.float16:
            max_value = 65504
        elif fpx.np == bfloat16:
            max_value = 3.3895313892515354759047080037148786688e38
        elif fpx.np == float8_e5m2:
            max_value = 57344
        elif fpx.np == float8_e4m3fn:
            max_value = 448
        min_value = -max_value
        neg_condition = (fp32_data > -float('inf')) & (fp32_data < min_value)
        fp32_data[torch.where(neg_condition)] = min_value
        pos_condition = (fp32_data > max_value) & (fp32_data < float('inf'))
        fp32_data[torch.where(pos_condition)] = max_value
    fpx_data = fp32_data.to(fpx.torch)
    if is_scalar:
        return fpx_data.item()
    if not is_tensor:
        if str(fpx) not in ['float16']:
            fpx_data = fpx_data.float()
        fpx_data = fpx_data.cpu().numpy().astype(fpx.np)

    return fpx_data


def _aiff_wt_dequant(fp32_wt, constants, oc, ic, kh, kw):
    # wt dequant zp
    if constants['weights'].quantization.offsets is not None:
        weights_zp = np.array(constants['weights'].quantization.offsets).astype(
            np.float32).flatten()  # [oc*group_num] or [oc] or [1] or []
        if 'zp0' in constants:
            # [oc*group_num] or [oc, group_num]
            # y = ((x @ (w + zp0) * scale0) * scale1 * scale2 + bias) * scale
            weights_zp = constants['zp0'].np.astype(np.float32).flatten().tolist()
        if len(weights_zp) != 0 and constants['weights'].dtype in [Dtype.INT8, Dtype.ALIGNED_INT4, Dtype.FP4E2M1FN]:
            weights_zp = torch.tensor(weights_zp).to(torch.float32).to(fp32_wt.device)
            if len(weights_zp) > oc:  # oc, already permute
                if kh * kw != 1:
                    raise Exception("gptq only support kernel=1x1!")
                weights_zp = weights_zp.reshape((oc, -1))  # [oc, group_num]
                weights_zp = weights_zp.permute([1, 0])  # [group_num, oc]
                weights_zp = torch.repeat_interleave(weights_zp, ic // weights_zp.shape[0], dim=0)
            fp32_wt = fp32_wt + weights_zp

    # wt dequant scale
    weights_scale = np.array(constants['weights'].quantization.scales).flatten()  # [oc*group_num] or [oc] or [1] or []
    if 'scale0' in constants:
        # [oc*group_num] or [oc, group_num]
        # y = ((x @ (w + zp0) * scale0) * scale1 * scale2 + bias) * scale
        weights_scale = constants['scale0'].betensor.flatten().float().reciprocal().tolist()
    if len(weights_scale) != 0 and constants['weights'].dtype in [Dtype.INT8, Dtype.ALIGNED_INT4, Dtype.FP4E2M1FN]:
        weights_scale = torch.tensor(weights_scale).to(torch.float32).to(fp32_wt.device)
        if len(weights_scale) > oc:  # oc, already permute
            if kh * kw != 1:
                raise Exception("gptq only support kernel=1x1!")
            weights_scale = weights_scale.reshape((oc, -1))  # [oc, group_num]
            weights_scale = weights_scale.permute([1, 0])  # [group_num, oc]
            weights_scale = torch.repeat_interleave(weights_scale, ic // weights_scale.shape[0], dim=0)
        fp32_wt = fp32_wt / weights_scale

        if constants['weights'].dtype in [Dtype.FP8E4M3FN, Dtype.FP8E5M2]:    # in hardware for int4xfp8, the weight type is float8 after dequantize finish
            fp32_wt = fp32_wt.to(constants['weights'].dtype.torch).to(torch.float32)

    return fp32_wt
