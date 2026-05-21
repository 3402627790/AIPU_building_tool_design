import torch
from .gt_util import *
from .gt_util.aiff import _aiff_compression_bias, _aiff_merge_shifts
from ..engine import register_gt, register_gen_weight, _OP_METRICS, register_op_metric
from .batchnorm import bn_compression_bias
from AIPUBuilder.core import OpType
from AIPUBuilder.logger import WARN
from .layernorm import sum_square_fp16
from .layernorm import groupnorm_aiff_int
from .gt_util.dtype_utils import torch_to_numpy, numpy_to_torch


@register_gt(OpType.GroupNorm)
def GroupNorm(self, *args, **kwargs):
    is_aiff = kwargs.get('is_aiff', False)
    tpc_or_aiff = 'aiff' if is_aiff else 'tpc'
    print('run groupnorm: %s version !' % tpc_or_aiff)
    self.attrs['is_aiff'] = is_aiff  # for groupnorm's metric

    self.sgattrs = kwargs['sgattrs'] if 'sgattrs' in kwargs else None
    if self.sgattrs is None:
        self.sgattrs = dict()
    if is_aiff:
        if self.inputs[0].dtype.is_float:
            outputs = groupnorm_float(self)
        else:
            reduce_mode = "div"
            is_transpose = True
            inp = self.inputs[0].irtensor.cpu().numpy().astype('float32')
            axis = self.params["axis"]
            in_dim = inp.ndim
            axis = axis + in_dim if axis < 0 else axis

            if inp.ndim == 5:
                n, d, h, w, c = inp.shape
                if axis == 1:
                    inp = inp.reshape([n, d, h * w, c])
                elif axis == 2:
                    inp = inp.reshape([n, d, h, w * c])
                elif axis == 3:
                    inp = inp.reshape([n, d * h, w, c])
                    axis = 2
                elif axis == 4:
                    inp = inp.reshape([n, d, h * w, c])
                    axis = 3
            self.params["axis"] = axis
            group = self.params['group']
            reduce_axis = [1, 2, 3]
            reduce_num = torch.div(torch.prod(torch.Tensor([inp.shape[a] for a in reduce_axis])), group)
            outputs = groupnorm_aiff_int(self, inp, inp.shape, reduce_mode, is_transpose, reduce_num)

    else:
        if self.inputs[0].dtype.is_float:
            outputs = groupnorm_float(self)
        else:
            target = Target().name
            if target.startswith(('X2', 'X3')):
                outputs = groupnorm_tpc_x2_int(self)
            else:
                lut = self.constants['lut'].numpy()
                weights = self.constants['weights'].numpy()
                biases = self.constants['biases'].numpy()
                scale = self.constants['scale'].numpy() if 'scale' in self.constants else None
                shift = self.constants['shift'].numpy() if 'shift' in self.constants else None

                inp = self.inputs[0].irtensor.cpu().numpy().astype(np.int64)
                input_type = str(self.inputs[0].dtype)
                output_type = str(self.outputs[0].dtype)

                axis = self.params['axis']
                group = self.params['group']
                norm_scale_value = self.params['norm_scale_value']
                norm_shift_value = self.params['norm_shift_value']
                var_shift_value = self.params['var_shift_value']

                in_dim = inp.ndim
                axis = axis + in_dim if axis < 0 else axis
                if inp.ndim == 5:
                    n, d, h, w, c = inp.shape
                    if axis == 1:
                        inp = inp.reshape([n, d, h * w, c])
                    elif axis == 2:
                        inp = inp.reshape([n, d, h, w * c])
                    elif axis == 3:
                        inp = inp.reshape([n, d * h, w, c])
                        axis = 2
                    elif axis == 4:
                        inp = inp.reshape([n, d, h * w, c])
                        axis = 3
                in_dim = inp.ndim
                trans_index = list(range(in_dim))
                trans_index.insert(1, trans_index.pop(axis))  # -> nc'hw, axis is real c
                new_shape = list(inp.shape)
                new_shape.insert(1, new_shape.pop(axis))

                restore_index = list(range(in_dim))
                restore_index.insert(axis, restore_index[-1])
                restore_index.pop()

                # transpose
                inp = np.transpose(inp, trans_index)  # transpose -> nc'hw
                n, c, h, w = inp.shape
                inp = np.reshape(inp, [n, group, c // group, h, w])  # [n, c', h, w] -> [n, g, c'//g, h, w]

                # mvn
                if input_type.endswith('8'):
                    mean = np.round(np.mean(inp, axis=(2, 3, 4), keepdims=True))
                    variance = np.trunc(np.sum((inp - mean) ** 2, axis=(2, 3, 4), keepdims=True)
                                        ).astype('int64') >> var_shift_value
                    mean = mean.astype('int64')
                    rsqrt_1 = np.uint32(rsqrt(lut, np.uint32(variance)))
                    ngama = rsqrt_1.reshape(np.shape(variance)).astype(np.int64)
                else:
                    mean = np.round(np.mean(inp, axis=(2, 3, 4), keepdims=True))
                    variance = np.trunc(np.mean((inp - mean) ** 2, axis=(2, 3, 4), keepdims=True))
                    mean = mean.astype('int64')
                    rsqrt_1 = np.uint32(rsqrt(lut, np.uint32(variance)))
                    ngama = rsqrt_1.reshape(np.shape(variance)).astype(np.int64)
                    ngama = np.round(ngama * norm_scale_value * (0.5 ** (norm_shift_value)))

                if np.any(variance == 0):
                    WARN(f'[Executor] the variance include 0 in GroupNorm!!!')

                assert output_type in ['int8', 'uint8', 'int16', 'uint16']
                sqrt_q = np.clip(ngama, np.iinfo("uint16").min, np.iinfo("uint16").max).astype('uint16')

                if input_type.endswith('8'):
                    normalized_q = (inp - mean) * sqrt_q
                    normalized_q = normalized_q.astype(np.int64)
                    normalized_q = np.round(normalized_q * norm_scale_value * 0.5 ** (8 + norm_shift_value))
                else:
                    normalized_q = (inp - mean) >> 1
                    normalized_q = normalized_q * sqrt_q
                    normalized_q = normalized_q.astype(np.int64)
                    normalized_q = np.round(normalized_q * 0.5 ** 7)

                normalized_q = np.clip(normalized_q, np.iinfo(output_type).min,
                                       np.iinfo(output_type).max).astype('int64')

                # transpose
                mvn_out = normalized_q.reshape(new_shape)  # [n, g, c'//g, h, w] -> nc'hw
                mvn_out = np.transpose(mvn_out, [0, 2, 3, 1])  # nc'hw -> nhwc'

                # bn
                if 'scale' in self.constants:
                    scale_value, shift_value = _aiff_merge_shifts(scale, shift)
                else:
                    scale_value = self.params['scale_value']
                    shift_value = self.params['shift_value']
                is_32bit_bias = True if self.sgattrs is not None and 'bn_32bit_bias' in self.sgattrs else False
                # hardware only support int16 bias
                biases = bn_compression_bias(torch.from_numpy(biases), 16, is_32bit_bias).cpu().numpy()
                # scale_value, shift_value = scale, shift  # opt version, no merge_shifts and compress_bias

                outputs = np.int64(mvn_out) * np.int64(weights) + biases
                outputs = np.int64(outputs) * np.int64(scale_value)
                outputs = np.round(outputs * 0.5 ** shift_value)
                outputs -= self.outputs[0].quantization.offset

                outputs = clamp(outputs, output_type)

                # transpose
                outputs = np.transpose(outputs, restore_index)  # nhwc' -> nhwc(ori_shape)

    self.outputs[0].irtensor = outputs.reshape(self.inputs[0].irtensor.shape)
    return [o.betensor for o in self.outputs]


def groupnorm_16bit(self):
    axis = self.params['axis']
    group = self.params['group']
    norm_scale_value = self.params['norm_scale_value']
    norm_shift_value = self.params['norm_shift_value']
    var_shift_value = self.params['var_shift_value']
    inp = self.inputs[0]
    out = self.outputs[0]
    inp_betensor = inp.irtensor.long()
    dim_num = inp_betensor.dim()
    channel_axis = axis
    if channel_axis < 0:
        channel_axis += dim_num
    if inp_betensor.ndim == 5:
        n, d, h, w, c = inp_betensor.shape
        if channel_axis == 1:
            inp_betensor = inp_betensor.reshape([n, d, h * w, c])
        elif channel_axis == 2:
            inp_betensor = inp_betensor.reshape([n, d, h, w * c])
        elif channel_axis == 3:
            inp_betensor = inp_betensor.reshape([n, d * h, w, c])
            channel_axis = 2
        elif channel_axis == 4:
            inp_betensor = inp_betensor.reshape([n, d, h * w, c])
            channel_axis = 3
    dim_num = inp_betensor.dim()
    axis = list(range(1, dim_num))  # axis=[H, W, C//G], 0-dim is batch
    in_chunks = inp_betensor.split(inp_betensor.shape[channel_axis] // group, dim=channel_axis)
    axis_reshape = [inp_betensor.shape[channel_axis] if ax == channel_axis else 1 for ax in range(dim_num)]
    scale_shift_shape = axis_reshape

    mean_chunks = []
    ngamma_chunks = []
    for t in in_chunks:
        q_bits_activation = inp.dtype.bits
        if q_bits_activation <= 8:
            qmin, qmax = bits2range(inp.dtype.bits, is_signed=is_signed(inp.dtype))
            var_shift = var_shift_value
            t_mean = torch.mean(t.double(), dim=axis, keepdim=True).round().long()
            t_mean = torch.clip(t_mean, qmin, qmax)
            t_var = torch.sum((t.double() - t_mean) ** 2, dim=axis, keepdim=True).long() >> var_shift
            t_ngamma = calculate_inverse_sqrt(t_var)
            t_ngamma = (t_ngamma + 32768) >> 16
            mean_chunks.append(t_mean)
            ngamma_chunks.append(t_ngamma)
        else:
            t_mean = torch.mean(t.double(), dim=axis, keepdim=True).round().long()
            t_var = torch.mean((t.double() - t_mean) ** 2, dim=axis, keepdim=True).long()
            t_ngamma = calculate_inverse_sqrt(t_var)
            t_ngamma = (t_ngamma + 32768) >> 16
            t_ngamma = linear_requantize(t_ngamma, norm_scale_value, norm_shift_value, 'int16')
            mean_chunks.append(t_mean)
            ngamma_chunks.append(t_ngamma)
        mean = torch.cat(mean_chunks, dim=channel_axis)
        ngamma = torch.cat(ngamma_chunks, dim=channel_axis)

    if mean.shape[channel_axis] != inp_betensor.shape[channel_axis]:
        mean = torch.repeat_interleave(mean, inp_betensor.shape[channel_axis]//group, dim=channel_axis)
        ngamma = torch.repeat_interleave(ngamma, inp_betensor.shape[channel_axis]//group, dim=channel_axis)

    if q_bits_activation <= 8:
        normalized = (inp_betensor - mean) * ngamma
        multiplier = norm_scale_value
        shift = 8 + norm_shift_value
    else:
        normalized = ((inp_betensor - mean) >> 1) * ngamma
        multiplier = 1
        shift = 7
    normalized = linear_requantize(normalized, multiplier, shift, out.dtype)

    gamma = 1.0
    if 'weights' in self.constants:
        w = self.constants["weights"]
        # must be positive axis
        w.key_axis = len(w.shape) - 1
        gamma = w.betensor
        gamma = torch.reshape(gamma, axis_reshape)
    beta = 0.0
    if 'biases' in self.constants:
        b = self.constants["biases"]
        b.key_axis = len(b.shape)-1
        beta = b.betensor
        beta = torch.reshape(beta, axis_reshape)
    output = gamma * normalized + beta

    if "shift" not in self.constants:
        do_shift = self.params["shift_value"]
        do_scale = self.params["scale_value"]
    else:
        do_shift = self.constants["shift"].betensor
        do_scale = self.constants["scale"].betensor
        do_shift = torch.reshape(do_shift, scale_shift_shape)
        do_scale = torch.reshape(do_scale, scale_shift_shape)
    o_zp = self.outputs[0].quantization.offset
    output = linear_requantize(output, do_scale, do_shift, out.dtype, o_zp)
    return output


def rsqrt(lut, x):
    t = (np.int32(D_UTIL_inverse_sqrt(lut, x)) + 32768) >> 16
    return t


def groupnorm_float(self):
    inp = torch_to_numpy(self.inputs[0].irtensor,  self.inputs[0].dtype.np).astype("float32")
    in_dim = inp.ndim
    group = self.params['group']
    axis = self.params['axis']
    axis = axis + in_dim if axis < 0 else axis

    if inp.ndim == 5:
        n, d, h, w, c = inp.shape
        if axis == 1:
            inp = inp.reshape([n, d, h * w, c])
        elif axis == 2:
            inp = inp.reshape([n, d, h, w * c])
        elif axis == 3:
            inp = inp.reshape([n, d * h, w, c])
            axis = 2
        elif axis == 4:
            inp = inp.reshape([n, d, h * w, c])
            axis = 3
    in_dim = inp.ndim

    trans_index = list(range(in_dim))
    trans_index.insert(1, trans_index.pop(axis))  # -> nc'hw, axis is real c
    new_shape = list(inp.shape)
    new_shape.insert(1, new_shape.pop(axis))

    restore_index = list(range(in_dim))
    restore_index.insert(axis, restore_index[-1])
    restore_index.pop()

    # transpose
    inp = np.transpose(inp, trans_index)  # transpose -> nc'hw
    n, c, h, w = inp.shape
    inp = np.reshape(inp, [n, group, c // group, h, w])  # [n, c', h, w] -> [n, g, c'//g, h, w]

    # norm
    mean = np.mean(inp, axis=(2, 3, 4), keepdims=True)
    variance = np.mean((inp - mean) ** 2, axis=(2, 3, 4), keepdims=True)
    rsqrt_1 = 1 / np.sqrt(variance)
    normalized_q = (inp - mean) * rsqrt_1

    # transpose
    out = normalized_q.reshape(new_shape)  # [n, g, c'//g, h, w] -> nc'hw
    out = np.transpose(out, [0, 2, 3, 1])  # nc'hw -> nhwc'

    gamma = self.constants['weights'].numpy()
    beta = self.constants['biases'].numpy()

    if '_output_shapes' in self.attrs:
        ds_weight_shape = self.inputs[0].irtensor.shape[axis]
        gamma = gamma.flatten()[:ds_weight_shape]
        beta = beta.flatten()[:ds_weight_shape]

    out = out * gamma + beta

    # transpose
    out = np.transpose(out, restore_index)  # nhwc' -> nhwc(ori_shape)

    gt_data = out.astype(str(self.outputs[0].dtype))
    return gt_data


def groupnorm_tpc_x2_int(self):
    lut = self.constants['lut'].numpy()
    weights = self.constants['weights'].numpy()
    biases = self.constants['biases'].numpy()
    if "scale" in self.constants.keys():
        scale = self.constants["scale"].numpy()
        shift = self.constants["shift"].numpy()
    else:
        shift = self.params["shift_value"]
        scale = self.params["scale_value"]

    inp = self.inputs[0].irtensor.cpu().numpy().astype(np.int64)
    input_type = str(self.inputs[0].dtype)
    output_type = str(self.outputs[0].dtype)

    axis = self.params['axis']
    group = self.params['group']
    norm_scale_value = self.params['norm_scale_value']
    norm_shift_value = self.params['norm_shift_value']
    var_shift_value = self.params['var_shift_value']

    in_dim = inp.ndim
    axis = axis + in_dim if axis < 0 else axis
    if inp.ndim == 5:
        n, d, h, w, c = inp.shape
        if axis == 1:
            inp = inp.reshape([n, d, h * w, c])
        elif axis == 2:
            inp = inp.reshape([n, d, h, w * c])
        elif axis == 3:
            inp = inp.reshape([n, d * h, w, c])
            axis = 2
        elif axis == 4:
            inp = inp.reshape([n, d, h * w, c])
            axis = 3
    in_dim = inp.ndim
    trans_index = list(range(in_dim))
    trans_index.insert(1, trans_index.pop(axis))  # -> nc'hw, axis is real c
    new_shape = list(inp.shape)
    new_shape.insert(1, new_shape.pop(axis))

    restore_index = list(range(in_dim))
    restore_index.insert(axis, restore_index[-1])
    restore_index.pop()

    # transpose
    inp = np.transpose(inp, trans_index)  # transpose -> nc'hw
    n, c, h, w = inp.shape
    inp = np.reshape(inp, [n, group, c // group, h, w])  # [n, c', h, w] -> [n, g, c'//g, h, w]

    # mvn
    if input_type.endswith('8'):
        mean = np.round(np.mean(inp, axis=(2, 3, 4), keepdims=True))
        variance = np.trunc(np.sum((inp - mean) ** 2, axis=(2, 3, 4), keepdims=True)
                            ).astype('int64') >> var_shift_value
        mean = mean.astype('int64')
        rsqrt_1 = np.uint32(rsqrt(lut, np.uint32(variance)))
        ngama = rsqrt_1.reshape(np.shape(variance)).astype(np.int64)
    else:
        mean = np.round(np.mean(inp, axis=(2, 3, 4), keepdims=True))
        variance = np.trunc(np.mean((inp - mean) ** 2, axis=(2, 3, 4), keepdims=True))
        mean = mean.astype('int64')
        rsqrt_1 = np.uint32(rsqrt(lut, np.uint32(variance)))
        ngama = rsqrt_1.reshape(np.shape(variance)).astype(np.int64)
        ngama = np.round(ngama * norm_scale_value * (0.5 ** (norm_shift_value)))

    if np.any(variance == 0):
        WARN(f'[Executor] the variance include 0 in GroupNorm!!!')

    assert output_type in ['int8', 'uint8', 'int16', 'uint16']
    sqrt_q = np.clip(ngama, np.iinfo("uint16").min, np.iinfo("uint16").max).astype('uint16')

    if input_type.endswith('8'):
        normalized_q = (inp - mean) * sqrt_q
        normalized_q = normalized_q.astype(np.int64)
        normalized_q = np.round(normalized_q * norm_scale_value * 0.5 ** (8 + norm_shift_value))
    else:
        normalized_q = (inp - mean) >> 1
        normalized_q = normalized_q * sqrt_q
        normalized_q = normalized_q.astype(np.int64)
        normalized_q = np.round(normalized_q * 0.5 ** 7)

    normalized_q = np.clip(normalized_q, np.iinfo(output_type).min, np.iinfo(output_type).max).astype('int64')

    # transpose
    mvn_out = normalized_q.reshape(new_shape)  # [n, g, c'//g, h, w] -> nc'hw
    mvn_out = np.transpose(mvn_out, [0, 2, 3, 1])  # nc'hw -> nhwc'

    # bn
    gamma = self.constants["weights"].numpy().astype("int64")
    beta = self.constants["biases"].numpy()

    if '_output_shapes' in self.attrs:
        ds_weight_shape = self.attrs['_output_shapes'][0][axis]
        gamma = gamma.flatten()[:ds_weight_shape]
        beta = beta.flatten()[:ds_weight_shape]
        scale = scale.flatten()[:ds_weight_shape] if 'scale' in self.constants else self.params['scale_value']
        shift = shift.flatten()[:ds_weight_shape] if 'shift' in self.constants else self.params['shift_value']

    bias_type = str(self.constants["biases"].dtype)
    if input_type.endswith("16") and bias_type == "int32":
        beta_high31, beta_shift = beta, 0
    else:
        beta_high31, beta_shift = tpc_bias_compact(beta, 63, 0, 31)

    outputs_q = (gamma * mvn_out >> beta_shift) + beta_high31.reshape(
        beta.shape
    )
    outputs_q = outputs_q.astype(np.int64)
    if input_type.endswith("16"):
        tmph = outputs_q >> 16
        tmpl = outputs_q - (tmph << 16)
        tmph = (tmph * scale).astype(np.int64)
        tmpl = (tmpl * scale).astype(np.int64)
        # print(shift)
        shift = shift - beta_shift
        if isinstance(shift, np.ndarray):
            tmpl = np.where(shift > 0, tmpl >> shift, tmpl)
            tmpl = np.where(shift < 0, tmpl << (-shift), tmpl)
        else:
            if shift > 0:
                tmpl = tmpl >> shift
            else:
                tmpl = tmpl << (-shift)
        shift -= 16
        if isinstance(shift, np.ndarray):
            tmph = np.where(shift > 0, tmph >> shift, tmph)
            tmph = np.where(shift < 0, tmph << (-shift), tmph)
        else:
            if shift > 0:
                tmph = tmph >> shift
            else:
                tmph = tmph << (-shift)
        outputs_q = tmph + tmpl
    else:
        outputs_q = outputs_q * scale
        outputs_q = np.round(outputs_q * (0.5 ** (shift - beta_shift)))
    # print(outputs_q.reshape(-1)[0:16])
    outputs = outputs_q
    outputs -= self.outputs[0].quantization.offset
    gt_data = np.clip(
        outputs, np.iinfo(output_type).min, np.iinfo(output_type).max
    )

    # transpose
    outputs = np.transpose(gt_data, restore_index)  # nhwc' -> nhwc(ori_shape)
    return outputs


@register_gen_weight(OpType.GroupNorm)
def gen_random_weight(self):
    if 'weights' in self.constants:
        weights_shape = list(self.constants['weights'].shape)
        weights_type = str(self.constants['weights'].dtype)
        biases_shape = list(self.constants['biases'].shape)
        biases_type = str(self.constants['biases'].dtype)
        if 'float' in weights_type:
            self.constants['weights'].betensor = (np.random.randn(*weights_shape) * 10000).astype(weights_type)
            self.constants['biases'].betensor = np.float32(np.random.randn(*biases_shape) * 100000)
        else:
            self.constants['weights'].betensor = np.random.randint(-128, 127, size=weights_shape, dtype=weights_type)
            random_v = np.random.randint(0, 100, size=1)
            if random_v % 2 == 0:
                if str(self.inputs[0].dtype).endswith('16'):
                    bias_min = - 2 ** 31
                    bias_max = 2 ** 31 - 1
                else:
                    bias_min = np.iinfo(biases_type).min
                    bias_max = np.iinfo(biases_type).max
            else:
                bias_min = 0
                bias_max = 255
            self.constants['biases'].betensor = np.random.randint(bias_min, bias_max, size=biases_shape,
                                                                  dtype=biases_type)
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
    if 'scale' in self.constants:
        scale_shape = list(self.constants['scale'].shape)
        scale_type = str(self.constants['scale'].dtype)
        shift_shape = list(self.constants['shift'].shape)
        shift_type = str(self.constants['shift'].dtype)
        self.constants['scale'].betensor = np.random.randint(np.iinfo(scale_type).min, min(32767, np.iinfo(scale_type).max),
                                                             size=scale_shape, dtype=scale_type)
        self.constants['shift'].betensor = np.random.randint(0, 17,
                                                             size=shift_shape, dtype=shift_type)


@register_op_metric(OpType.GroupNorm)
@check_fp_data
def metric(self, outputs, index=None):
    # note: aiff's metric maybe called in layoutconvertor
    def aiff_call(r, u, c, t):
        if "float" in t:
            if np.all(u <= 1) or np.all(c >= 0.999):
                return True
            return np.all(r < 0.001) and np.all(u <= 2) and np.all(c >= 0.9)
        else:
            return np.all(u <= 1) or np.all(c >= 0.99)

    def fp_call(r, u, c, t):
        if np.all(u <= 1):
            return True
        elif np.all(c >= 0.999):
            return True
        return np.all(r < 0.001) and np.all(u <= 2) and np.all(c >= 0.9)

    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]

    layer_dtypes = [str(o.dtype) for o in self.outputs] if index is None else [str(self.outputs[index].dtype)]

    rel_diffs = relative_diff(outputs, groundtruths)
    ulp_diffs = ulp_diff(outputs, groundtruths)
    cos_dists = cos_distance(outputs, groundtruths)
    if self.attrs.get('is_aiff', False):
        return all(aiff_call(r, u, c, t) for r, u, c, t in zip(rel_diffs, ulp_diffs, cos_dists, layer_dtypes))
    if self.outputs[0].dtype.is_float:
        return all(fp_call(r, u, c, t) for r, u, c, t in zip(rel_diffs, ulp_diffs, cos_dists, layer_dtypes))

    return _OP_METRICS['_all'](self, outputs, index)
