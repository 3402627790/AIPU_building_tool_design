import torch
from ..engine import register_gt, register_gen_weight, _OP_METRICS, register_op_metric
from .gt_util import *
from .gt_util.math_utils import fp32_to_fp24
from AIPUBuilder.core import OpType, Dtype
from AIPUBuilder.logger import WARN
from .gt_util import aiff_itp_acc_channel
from .gt_util.dtype_utils import torch_to_numpy, numpy_to_torch
from .gt_util.dtype_utils import torch_to_numpy, numpy_to_torch


@register_gt(OpType.LayerNorm)
def layernorm(self, *args, **kwargs):
    is_aiff = kwargs.get('is_aiff', False)
    tpc_or_aiff = 'aiff' if is_aiff else 'tpc'
    print('run layernorm: %s version !' % tpc_or_aiff)
    self.attrs['is_aiff'] = is_aiff  # for layernorm's metric
    if is_aiff:
        if self.inputs[0].dtype.is_float:
            layernorm_float_aiff(self)
        else:
            layernorm_int_aiff(self)
    else:
        if self.inputs[0].dtype.is_float:
            gt_data = layernorm_float(self)
        else:
            inp = self.inputs[0].irtensor.cpu().numpy().astype(np.int64)
            axis = self.params["axis"]
            axis = [a + inp.ndim if a < 0 else a for a in axis]
            gamma = self.constants["weights"].numpy().astype("int64")
            beta = self.constants["biases"].numpy()
            # per-channel
            if "scale" in self.constants.keys():
                scale = self.constants["scale"].numpy()
                shift = self.constants["shift"].numpy()
            else:
                shift = self.params["shift_value"]
                scale = self.params["scale_value"]
            if '_output_shapes' in self.attrs:
                real_weights_shape = []
                real_weights_size = 1
                for a in axis:
                    real_weights_shape.append(self.attrs['_output_shapes'][0][a])
                    real_weights_size *= self.attrs['_output_shapes'][0][a]
                real_biases_shape = real_weights_shape
                real_biases_size = real_weights_size
                real_scale_shape = real_shift_shape = [self.attrs['_output_shapes'][0][-1]]
                gamma = gamma.flatten()[:real_weights_size].reshape(real_weights_shape)
                beta = beta.flatten()[:real_biases_size].reshape(real_biases_shape)
                scale = scale.flatten()[:real_scale_shape[0]].reshape(
                    real_scale_shape) if 'scale' in self.constants else self.params['scale_value']
                shift = shift.flatten()[:real_shift_shape[0]].reshape(
                    real_shift_shape) if 'shift' in self.constants else self.params['shift_value']
            inputs_shape = inp.shape

            output_type = str(self.outputs[0].dtype)
            input_type = str(self.inputs[0].dtype)
            # inputs_shape = list(self.inputs[0].betensor.shape)
            output_shape = list(self.outputs[0].betensor.shape)
            bias_type = str(self.constants["biases"].dtype)

            lut = self.constants["lut"].numpy()
            norm_shift_value = self.params["norm_shift_value"]
            norm_scale_value = self.params["norm_scale_value"]
            var_shift_value = self.params["var_shift_value"]

            if input_type.endswith("8"):
                mean = np.round(np.mean(inp, axis=tuple(axis), keepdims=True))
                sum1 = np.round(np.sum(inp, axis=tuple(axis), keepdims=True))
                variance = (
                    np.trunc(
                        np.sum((inp - mean) ** 2, axis=tuple(axis), keepdims=True)
                    ).astype("int64")
                    >> var_shift_value
                )
                mean = mean.astype("int64")
                rsqrt_int = np.uint32(rsqrt(lut, np.uint32(variance)))
                ngama = rsqrt_int.reshape(np.shape(variance)).astype(np.int64)
            else:
                mean = np.round(np.mean(inp, axis=tuple(axis), keepdims=True))
                variance = np.trunc(
                    np.mean((inp - mean) ** 2, axis=tuple(axis), keepdims=True)
                )
                mean = mean.astype("int64")
                rsqrt_int = np.uint32(rsqrt(lut, np.uint32(variance)))
                ngama = rsqrt_int.reshape(np.shape(variance)).astype(np.int64)
                ngama = np.round(ngama * norm_scale_value * (0.5 ** (norm_shift_value)))

            if np.any(variance == 0):
                WARN(f'[Executor] the variance include 0 in LayerNorm!!!')

            assert output_type in ["int8", "uint8", "int16", "uint16"]

            clip_type = "uint16"
            ngamma_q = np.clip(
                ngama, np.iinfo(clip_type).min, np.iinfo(clip_type).max
            ).astype(clip_type)

            if input_type.endswith("16"):
                normalized_q = (inp - mean) >> 1
                normalized_q = normalized_q * ngamma_q
                normalized_q = normalized_q.astype(np.int64)
                normalized_q = np.round(normalized_q * 0.5**7)
            else:
                normalized_q = (inp - mean) * ngamma_q
                normalized_q = normalized_q.astype(np.int64)
                normalized_q = np.round(normalized_q * 0.5 ** (8 + norm_shift_value))

            normalized_q = np.clip(
                normalized_q, np.iinfo(output_type).min, np.iinfo(output_type).max
            ).astype(output_type)

            if input_type.endswith("16") and bias_type == "int32":
                beta_high31, beta_shift = beta, 0
            else:
                beta_high31, beta_shift = tpc_bias_compact(beta, 63, 0, 31)
            outputs_q = (gamma * normalized_q >> beta_shift) + beta_high31.reshape(
                beta.shape
            )
            outputs_q = outputs_q.astype(np.int64)

            # outputs_q * scale >> (shift - beta_shift)
            outputs_q = outputs_q.reshape(inputs_shape)

            if input_type.endswith("16"):
                tmph = outputs_q >> 16
                tmpl = outputs_q - (tmph << 16)
                tmph = (tmph * scale).astype(np.int64)
                tmpl = (tmpl * scale).astype(np.int64)
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
            outputs = outputs_q
            outputs -= self.outputs[0].quantization.offset
            gt_data = np.clip(
                outputs, np.iinfo(output_type).min, np.iinfo(output_type).max
            )
        gt_data = numpy_to_torch(gt_data, self.outputs[0].dtype.np)
        self.outputs[0].irtensor = gt_data
        if len(self.outputs) > 1:
            self.outputs[1].irtensor = mean
            self.outputs[2].irtensor = ngamma_q
        return [o.betensor for o in self.outputs]


def layernorm_float(self):
    in_dtype = 'float32' if self.inputs[0].dtype.np == np.float16 else self.inputs[0].dtype.np  # add_layernorm fp fuse
    input_data = torch_to_numpy(self.inputs[0].betensor, in_dtype)
    axis = [a + input_data.ndim if a < 0 else a for a in self.params["axis"]]
    in_shape = input_data.shape
    gamma = self.constants["weights"].numpy().astype("float32")
    beta = self.constants["biases"].numpy().astype("float32")
    if '_output_shapes' in self.attrs:
        real_weights_shape = []
        real_weights_size = 1
        for a in axis:
            real_weights_shape.append(self.attrs['_output_shapes'][0][a])
            real_weights_size *= self.attrs['_output_shapes'][0][a]
        real_biases_shape = real_weights_shape
        real_biases_size = real_weights_size
        gamma = gamma.flatten()[:real_weights_size].reshape(real_weights_shape)
        beta = beta.flatten()[:real_biases_size].reshape(real_biases_shape)
    tmp_len = np.prod([in_shape[a] for a in self.params["axis"]])
    tmp1_len = np.prod(in_shape) // tmp_len

    if tmp_len <= 8 * 1024:
        data1 = input_data.reshape([tmp1_len, tmp_len])
        data_tmp = np.zeros([tmp1_len, 8], dtype=np.float32)
        data_tmp_square = np.zeros([tmp1_len, 8], dtype=np.float32)
        data_tmp, data_tmp_square = sum_square_fp16(
            data1, tmp_len, data_tmp, data_tmp_square
        )
    else:
        # calc sum and square_sum on multi-tec
        tec_num = 4
        max_elts_one_tec = 511 * 16
        core_total_buffer_elts = (
            max_elts_one_tec * tec_num
        )  # calc 511 * 16 * 4 elements at a time
        core_loop_cnts = tmp_len // core_total_buffer_elts
        core_tail_elts = tmp_len % core_total_buffer_elts

        data1 = input_data.reshape([tmp1_len, tmp_len])
        data_tmp = np.zeros([tmp1_len, tec_num, 8], dtype=np.float32)
        data_tmp_square = np.zeros([tmp1_len, tec_num, 8], dtype=np.float32)

        for i in range(core_loop_cnts):
            data_i = data1[
                :, core_total_buffer_elts * i: core_total_buffer_elts * (i + 1)
            ]
            data_i = data_i.reshape(tmp1_len, tec_num, max_elts_one_tec)
            data_tmp, data_tmp_square = sum_square_fp16(
                data_i, max_elts_one_tec, data_tmp, data_tmp_square, tec_num=4
            )

        if core_tail_elts > 0:
            core_tail_elts_per_pe = core_tail_elts // tec_num
            data_tail = data1[
                :,
                core_total_buffer_elts
                * core_loop_cnts: core_total_buffer_elts
                * core_loop_cnts
                + core_tail_elts_per_pe * tec_num,
            ]
            pad_num = core_tail_elts % tec_num
            if pad_num > 0:
                if core_tail_elts_per_pe > 0:
                    data_tail = data_tail.reshape(
                        tmp1_len, tec_num, core_tail_elts_per_pe
                    )
                    data_tail = np.pad(data_tail, ((0, 0), (0, 0), (0, pad_num)))
                    data_tail[..., 3, -pad_num:] = data1[..., -pad_num:]
                    core_tail_elts_per_pe += pad_num
                else:
                    core_tail_elts_per_pe += core_tail_elts % tec_num
                    data_tail = np.pad(data_tail, ((0, 0), (0, pad_num * 4)))
                    data_tail = data_tail.reshape(
                        tmp1_len, tec_num, core_tail_elts_per_pe
                    )
                    data_tail[..., 3, -pad_num:] = data1[..., -pad_num:]
            data_tail = data_tail.reshape(tmp1_len, tec_num, core_tail_elts_per_pe)
            data_tmp, data_tmp_square = sum_square_fp16(
                data_tail, core_tail_elts_per_pe, data_tmp, data_tmp_square, tec_num=4
            )

        data_tmp = data_tmp.sum(axis=-2)
        data_tmp_square = data_tmp_square.sum(axis=-2)

    tmp_sum0 = np.sum(input_data, axis=tuple(axis), keepdims=True)
    tmp_sum = np.sum(data_tmp, axis=-1, keepdims=True)  # _vrpadd
    tmp_sum = tmp_sum.reshape(tmp_sum0.shape).astype("float64")
    # mean = tmp_sum / tmp_len
    # var = np.var(input_data, axis=tuple(axis), keepdims=True)
    # variance = (np.sum(input_data ** 2, axis=tuple(axis), keepdims=True)
    #             * tmp_len - tmp_sum ** 2) / (tmp_len ** 2)  # E(X^2)-E(X)^2
    # rsqrt_1 = 1 / np.sqrt(variance)
    # normalized_q = (input_data - mean) * rsqrt_1
    # # normalized_q = normalized_q.astype('float32')

    # norm = x*N/sqrt(N*sum(x^2)-sum(x)^2) - sum(x)/sqrt(N*sum(x^2)-sum(x)^2)
    tmp_square_sum = (
        np.sum(data_tmp_square, axis=-1, keepdims=True)
        .reshape(tmp_sum0.shape)
        .astype("float64")
    )  # _vrpadd

    # ((a*b).to('fp32).to('fp64') - (c*d).to('fp32).to('fp64')).to('fp32')
    tmp_rsqrt = (tmp_len * tmp_square_sum).astype("float32")
    tmp_rsqrt1 = (tmp_sum**2).astype("float32")
    tmp_rsqrt = (tmp_rsqrt.astype("float64") - tmp_rsqrt1.astype("float64")).astype(
        "float32"
    )

    tmp_rsqrt = f32_rsqrt(tmp_rsqrt).astype('float64')
    # tmp_rsqrt = f32_rsqrt(tmp_rsqrt + self.params['epsilon']).astype('float64')

    tmp_a = (tmp_len * tmp_rsqrt).astype("float32")
    tmp_b = (-tmp_sum * tmp_rsqrt).astype("float32")
    from .gt_util.math_utils import fma

    normalized_q = fma(input_data, tmp_a, tmp_b)
    # normalized_q = input_data.astype('float64') * (tmp_len * tmp_rsqrt) - tmp_sum * tmp_rsqrt
    normalized_q = normalized_q.astype("float32")

    # out = normalized_q * gamma + beta
    out = fma(normalized_q, gamma, beta)  # use fma align to hardware
    return out.astype(str(self.outputs[0].dtype))


def rsqrt(lut, x):
    t = (np.int32(D_UTIL_inverse_sqrt(lut, x)) + 32768) >> 16
    return t


def layernorm_float_aiff(self):

    input_data = self.inputs[0].irtensor.to(torch.float32)

    c = input_data.shape[-1]
    input_data = fp32_to_fp24(input_data)
    axis = [a + input_data.ndim if a < 0 else a for a in self.params["axis"]]
    in_shape = input_data.shape
    reduce_num = torch.prod(torch.Tensor([in_shape[a] for a in axis]))

    input_shape = list(input_data.shape)
    if len(axis) > 1:
        input_data = torch.reshape(
            input_data, tuple(input_shape[: -len(axis)] + [int(reduce_num.item())])
        )
    gamma = self.constants["weights"].betensor.to(torch.float32)
    beta = self.constants["biases"].betensor.to(torch.float32)
    if '_output_shapes' in self.attrs:
        real_weights_shape = []
        real_weights_size = 1
        for a in axis:
            real_weights_shape.append(self.inputs[0].irtensor.shape[a])
            real_weights_size *= self.inputs[0].irtensor.shape[a]
        real_biases_shape = real_weights_shape
        real_biases_size = real_weights_size
        gamma = gamma.flatten()[:real_weights_size].reshape(real_weights_shape)
        beta = beta.flatten()[:real_biases_size].reshape(real_biases_shape)
    data_channel_sum = aiff_itp_acc_channel(input_data)
    # data_channel_sum = torch.reshape(data_channel_sum, input_shape[: -len(axis)] + [1] * len(axis))
    mean_fp32 = data_channel_sum / reduce_num
    mean_shape = mean_fp32.shape
    mean_fp24 = fp32_to_fp24(mean_fp32)

    power2 = fp32_to_fp24((fp32_to_fp24(input_data - mean_fp24)) ** 2)
    variance = aiff_itp_acc_channel(power2)
    # variance = fp32_to_fp24(variance)
    var_shape = variance.shape
    rsqrt_1_fp32 = f32_rsqrt(variance)
    rsqrt_1_fp24 = fp32_to_fp24(rsqrt_1_fp32)

    sqrt_n_fp32 = -1 * torch.sqrt(reduce_num)
    sqrt_n_fp24 = fp32_to_fp24(sqrt_n_fp32)
    beta_fp24 = beta
    subtract = fp32_to_fp24(mean_fp24 - input_data)
    output_without_scale = fp32_to_fp24((subtract * rsqrt_1_fp24).to(torch.float32))
    output_without_scale = fp32_to_fp24(output_without_scale * sqrt_n_fp24)
    output_without_scale = torch.reshape(output_without_scale, input_shape)

    output_without_scale = fp32_to_fp24(output_without_scale * gamma)
    output = output_without_scale.to(torch.double) + fp32_to_fp24(beta)
    self.outputs[0].irtensor = output.to(self.outputs[0].dtype.torch)
    return output


def layernorm_int_aiff(self):

    input_data = self.inputs[0].irtensor.cpu().numpy().astype("float32")
    input_type = str(self.inputs[0].dtype)
    in_shape = input_data.shape
    is_transpose = False
    axis = [a + input_data.ndim if a < 0 else a for a in self.params["axis"]]
    reduce_num = torch.prod(torch.Tensor([in_shape[a] for a in axis]))
    self.params["axis"] = axis
    reduce_mode = "div"
    if (input_type.endswith("8")) and (len(axis) == 1):
        reduce_mode = "acc"

    input_shape = list(input_data.shape)
    reduce_num = torch.prod(torch.Tensor([input_shape[a] for a in axis]))

    if len(axis) > 1:
        input_data = np.reshape(input_data, tuple(input_shape[: -len(axis)] + [int(reduce_num.item())]))
    output = groupnorm_aiff_int(self, input_data, in_shape, reduce_mode, is_transpose, reduce_num)
    return output


def groupnorm_aiff_int(self, input, input_shape, reduce_mode, transpose, acc_num):
    input_type = str(self.inputs[0].dtype)
    output_type = str(self.outputs[0].dtype)
    bias_type = str(self.constants["biases"].dtype)

    group = self.params["group"] if "group" in self.params else 1

    if "axis" not in self.params or self.params["axis"] is None:
        axis = 3
    else:
        axis = self.params["axis"]

    in_dim = input.ndim
    in_shape = input.shape
    var_scale_num = acc_num
    reduce_num = acc_num
    if (input_type.endswith("8")):
        var_scale_num = torch.tensor([1])

    if (input_type.endswith("8")) and (reduce_mode == "acc"):
        input_scale = reduce_num
        reduce_num = torch.tensor([1])
    else:
        input_scale = 1.0

    gamma = self.constants["weights"].betensor.to(torch.float32)
    beta = self.constants["biases"].numpy()

    if '_output_shapes' in self.attrs:
        tmp_axis = [axis] if isinstance(axis, int) else axis
        ds_weight_shape = np.prod([self.inputs[0].irtensor.shape[a] for a in tmp_axis])
        gamma = gamma.flatten()[:ds_weight_shape]
        beta = beta.flatten()[:ds_weight_shape]

    if input_type.endswith("16") and bias_type == "int32":
        beta_high31, beta_shift = beta, 0
    else:
        beta_high31, beta_shift = tpc_bias_compact(beta, 63, 0, 31)
    beta_high31 = torch.from_numpy(beta_high31).to(torch.float32).reshape(gamma.shape)

    sqrt_n_fp32 = -1 * torch.sqrt(var_scale_num)
    norm_scale = self.params["norm_scale_value"]
    norm_shift = self.params["norm_shift_value"]
    norm_shift = norm_shift - 7 + beta_shift
    norm_pre = norm_scale * 0.5**norm_shift

    if input_type.endswith("8"):
        var_shift = self.params["var_shift_value"]
        norm_pre *= 2 ** (var_shift / 2)

    sqrt_n_fp32 = sqrt_n_fp32 * norm_pre
    sqrt_n_fp24 = fp32_to_fp24(sqrt_n_fp32)
    if "scale" in self.constants.keys():
        scale = self.constants["scale"].numpy()
        shift = self.constants["shift"].numpy()
        scale = scale.astype("float32") * (0.5**shift)
        output_scale = 1
        output_shift = 0
        copy_times = int(acc_num/scale.shape[0])
        if copy_times != 1:
            scale = np.tile(scale, copy_times)
            scale = np.reshape(scale, gamma.shape)
        gamma *= scale
        beta_high31 *= scale

    else:
        output_shift = self.params["shift_value"]
        output_scale = self.params["scale_value"]

    # transpose
    if transpose:
        trans_index = list(range(in_dim))
        trans_index.insert(1, trans_index.pop(axis))  # -> nc'hw, axis is real c
        new_shape = list(input.shape)
        new_shape.insert(1, new_shape.pop(axis))

        restore_index = list(range(in_dim))
        restore_index.insert(axis, restore_index[-1])
        restore_index.pop()
        input = np.transpose(input, trans_index)
        input = np.reshape(input, [-1, int(acc_num.item())])

    input = torch.from_numpy(input).to(torch.float32)
    input = fp32_to_fp24(input)

    data_channel_sum = aiff_itp_acc_channel(input)
    mean_fp32 = data_channel_sum / reduce_num
    mean_fp24 = fp32_to_fp24(mean_fp32)

    power2 = fp32_to_fp24((fp32_to_fp24(input * input_scale - mean_fp24)) ** 2)
    variance = aiff_itp_acc_channel(power2)

    rsqrt_1_fp32 = f32_rsqrt(variance)
    rsqrt_1_fp24 = fp32_to_fp24(rsqrt_1_fp32)
    input_tmp = fp32_to_fp24(input * input_scale)
    subtract = fp32_to_fp24(mean_fp24 - input_tmp)
    output_without_scale = fp32_to_fp24((subtract * rsqrt_1_fp24).to(torch.float32))
    output_without_scale = fp32_to_fp24(output_without_scale * sqrt_n_fp24)
    if transpose:
        output_without_scale = torch.reshape(output_without_scale, new_shape)
        output_without_scale = torch.permute(output_without_scale, [0, 2, 3, 1])
    else:
        output_without_scale = torch.reshape(output_without_scale, input_shape)

    gamma_fp24 = fp32_to_fp24(gamma)
    output_without_scale = fp32_to_fp24(output_without_scale * gamma_fp24.to(output_without_scale.device))

    beta_fp32 = beta_high31
    output_without_scale = (output_without_scale.to(torch.double) + beta_fp32.to(output_without_scale.device))
    output_without_scale_int = np.round(output_without_scale).to(torch.int64)
    output = np.round(output_without_scale_int * output_scale * 0.5**output_shift)
    output -= self.outputs[0].quantization.offset
    output = np.clip(output, np.iinfo(output_type).min, np.iinfo(output_type).max)
    if output_type == "int8":
        self.outputs[0].irtensor = output.to(torch.int8)
    else:
        if output_type == "uint8":
            self.outputs[0].irtensor = output.to(torch.uint8)
        else:
            self.outputs[0].irtensor = output.to(torch.int16)

    return output


@register_gen_weight(OpType.LayerNorm)
def gen_random_weight(self):
    if 'weights' in self.constants:
        weights_shape = list(self.constants['weights'].shape)
        weights_type = str(self.constants['weights'].dtype)
        biases_shape = list(self.constants['biases'].shape)
        biases_type = str(self.constants['biases'].dtype)
        if 'float' in weights_type:
            self.constants['weights'].betensor = np.float16(np.random.randn(*weights_shape) * 1000)
            self.constants['biases'].betensor = np.float32(np.random.randn(*biases_shape) * 1000)
            # self.constants['weights'].betensor = np.ones(weights_shape, weights_type)
            # self.constants['biases'].betensor = np.zeros(biases_shape, biases_type)
        else:
            self.constants['weights'].betensor = np.random.randint(-128, 127, size=weights_shape, dtype=weights_type)
            random_v = np.random.randint(0, 100, size=1)
            if random_v % 2 == 0:
                if str(self.inputs[0].dtype).endswith('16') and biases_type == 'int64':
                    bias_min = - 2 ** 47
                    bias_max = 2 ** 47 - 1
                else:
                    bias_min = - 2 ** 16
                    bias_max = 2 ** 16 - 1
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
        self.constants['scale'].betensor = np.random.randint(np.iinfo(scale_type).min, np.iinfo(scale_type).max,
                                                             size=scale_shape, dtype=scale_type)
        self.constants['shift'].betensor = np.random.randint(0, 17,
                                                             size=shift_shape, dtype=shift_type)


@register_op_metric(OpType.LayerNorm)
@check_fp_data
def metric(self, outputs, index=None):
    # note: aiff's metric maybe called in layoutconvertor/reshape
    def aiff_call(r, u, c, t):
        if "float" in t:
            if np.all(u <= 1) or np.all(c >= 0.999):
                return True
            return np.all(r < 0.001) and np.all(u <= 2) and np.all(c >= 0.9)
        return np.all(u <= 1) or np.all(c >= 0.999)  # aiff int

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
