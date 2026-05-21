import torch
from .gt_util import *
from .gt_util.aiff import _aiff_compression_bias, _aiff_merge_shifts
from ..engine import register_gt, register_gen_weight, _OP_METRICS, register_op_metric
from AIPUBuilder.core import OpType
from AIPUBuilder.logger import WARN
from .gt_util.math_utils import fp32_to_fp24
from .layernorm import groupnorm_aiff_int
from .gt_util.dtype_utils import torch_to_numpy, numpy_to_torch


@register_gt(OpType.InstanceNorm)
def InstanceNorm(self, *args, **kwargs):
    is_aiff = kwargs.get('is_aiff', False)
    tpc_or_aiff = 'aiff' if is_aiff else 'tpc'
    print(f'run instancenorm: {tpc_or_aiff} version !')
    self.attrs['is_aiff'] = is_aiff  # for instancenorm's metric
    if self.inputs[0].dtype.is_float:
        outputs = instancenorm_float(self, is_aiff)
    else:
        if is_aiff:
            reduce_mode = "acc"
            is_transpose = True
            input = self.inputs[0].irtensor.cpu().numpy().astype('float32')
            if (str(self.inputs[0].dtype).endswith("16")):
                reduce_mode = "div"
            reduce_axis = [1, 2]
            reduce_num = (torch.prod(torch.Tensor([input.shape[a] for a in reduce_axis])))
            outputs = groupnorm_aiff_int(self, input, input.shape, reduce_mode, is_transpose, reduce_num)
        else:

            lut = self.constants['lut'].numpy()
            weights = self.constants['weights'].numpy()
            biases = self.constants['biases'].numpy()
            scale = self.constants['scale'].numpy() if 'scale' in self.constants.keys() else self.params['scale_value']
            shift = self.constants['shift'].numpy() if 'shift' in self.constants.keys() else self.params['shift_value']
            if '_output_shapes' in self.attrs:
                real_weights_shape = real_biases_shape = real_scale_shape = real_shift_shape = [
                    self.attrs['_output_shapes'][0][-1]]
                weights = weights.flatten()[:real_weights_shape[0]].reshape(real_weights_shape)
                biases = biases.flatten()[:real_biases_shape[0]].reshape(real_biases_shape)
                scale = scale.flatten()[:real_scale_shape[0]].reshape(
                    real_scale_shape) if 'scale' in self.constants else self.params['scale_value']
                shift = shift.flatten()[:real_shift_shape[0]].reshape(
                    real_shift_shape) if 'shift' in self.constants else self.params['shift_value']
            input = self.inputs[0].irtensor.cpu().numpy().astype(np.int64)
            input_type = str(self.inputs[0].dtype)
            output_type = str(self.outputs[0].dtype)
            bias_type = str(self.constants['biases'].dtype)

            norm_scale_value = self.params['norm_scale_value']
            norm_shift_value = self.params['norm_shift_value']
            var_shift_value = self.params['var_shift_value']

            shape = input.shape
            input_data1 = np.transpose(input, [0, 3, 1, 2])
            input = input_data1.copy()
            input_data1 = input_data1.reshape((shape[0], shape[3], shape[1] * shape[2]))

            size1 = shape[-1]
            gamma = weights.reshape(size1).astype(np.int32)
            beta = biases.reshape(size1).astype(np.int64)

            # Normalization on HW dim (input_data has been transposed into [0, 3, 1, 2])
            axis = (-2, -1)
            if input_type.endswith('8'):
                mean = np.round(np.mean(input, axis=tuple(axis), keepdims=True))
                variance = np.trunc(np.sum((input - mean) ** 2, axis=tuple(axis), keepdims=True)).astype(
                    'int64') >> var_shift_value
                # print(variance)
                # assert np.all(variance != 0)
                mean = mean.astype('int64')
                variance_np = np.reshape(variance, (-1,))
                rsqrt_1 = np.uint32(rsqrt(lut, np.uint32(variance_np)))
                ngama = rsqrt_1.astype(np.int64)
            else:
                mean = np.round(np.mean(input, axis=tuple(axis), keepdims=True))
                variance = np.trunc(np.mean((input - mean) ** 2, axis=tuple(axis), keepdims=True))
                mean = mean.astype('int64')
                variance_np = np.reshape(variance, (-1,))
                rsqrt_1 = np.uint32(rsqrt(lut, np.uint32(variance_np)))
                ngama = rsqrt_1.astype(np.int64)
                ngama = np.round(ngama * norm_scale_value * (0.5 ** (norm_shift_value)))

            if np.any(variance == 0):
                WARN(f'[Executor] the variance include 0 in InstanceNorm!!!')

            np_mean = mean.reshape(-1)
            clip_type = 'uint16'
            ngamma_q = np.clip(ngama, np.iinfo(clip_type).min, np.iinfo(clip_type).max).astype(clip_type)

            if input_type.endswith('16') and bias_type == 'int32':
                beta_high31, beta_shift = beta, 0
            elif input_type.endswith('16'):
                beta_high31, beta_shift = tpc_bias_compact(beta, 63, 0, 31)

            data_out = []
            # Calculate along N/C dim
            for i in range(0, shape[0]):
                data2 = input_data1[i]
                for k in range(0, shape[3]):
                    if 'scale' in self.constants.keys():
                        scale_value = scale[k]
                        shift_value = shift[k]
                    else:
                        scale_value = scale
                        shift_value = shift
                    data = data2[k].astype(np.int32)
                    beta_c = beta[k]
                    gamma_c = gamma[k]
                    if input_type.endswith('16'):
                        data_norm = (data - np_mean[i * shape[3] + k]) >> 1
                        data_norm = data_norm * ngamma_q[i * shape[3] + k]
                        data_norm = data_norm.astype(np.int64)
                        data_norm = np.round(data_norm * 0.5 ** 7)
                    else:
                        data_norm = (data - np_mean[i * shape[3] + k]).astype(np.int64)
                        data_norm = data_norm * ngamma_q[i * shape[3] + k]
                        data_norm = np.round(data_norm * 0.5 ** (8 + norm_shift_value))

                    data_norm = np.clip(data_norm, np.iinfo(output_type).min,
                                        np.iinfo(output_type).max).astype(np.int64)

                    if input_type.endswith('16'):
                        beta_high31_c = beta_high31[k]
                        data_norm = (gamma_c * data_norm >> beta_shift) + beta_high31_c
                        data_norm = data_norm.astype(np.int64)
                        tmph = data_norm >> 16
                        tmpl = data_norm - (tmph << 16)
                        tmph = (tmph * scale_value).astype(np.int64)
                        tmpl = (tmpl * scale_value).astype(np.int64)
                        shift1 = shift_value - beta_shift
                        if shift1 > 0:
                            tmpl = tmpl >> shift1
                        else:
                            tmpl = tmpl << (-shift1)
                        shift1 -= 16
                        if shift1 > 0:
                            tmph = tmph >> shift1
                        else:
                            tmph = tmph << (-shift1)
                        data_norm = tmph + tmpl
                    else:
                        data_norm = data_norm * gamma_c + beta_c
                        data_norm = np.round(data_norm * scale_value * 0.5 ** shift_value)
                    data_out.append(data_norm)

            out = np.asarray(data_out).reshape([shape[0], shape[3], shape[1], shape[2]])
            out = np.transpose(out, [0, 2, 3, 1]).astype(np.int64)
            out -= self.outputs[0].quantization.offset
            outputs = np.clip(out, np.iinfo(output_type).min, np.iinfo(output_type).max).astype(output_type)

    self.outputs[0].irtensor = outputs
    return [o.betensor for o in self.outputs]


def rsqrt(lut, x):
    t = (np.int32(D_UTIL_inverse_sqrt(lut, x)) + 32768) >> 16
    return t


def instancenorm_float(self, is_aiff=False):
    #input_data = self.inputs[0].betensor.cpu().numpy().astype('float32')
    input_data = torch_to_numpy(self.inputs[0].irtensor,  self.inputs[0].dtype.np).astype("float32")

    weights = self.constants['weights'].numpy()
    biases = self.constants['biases'].numpy()

    #shape = list(self.inputs[0].shape)
    shape = list(input_data.shape)
    input_data = np.transpose(input_data, [0, 3, 1, 2])
    input_data = input_data.reshape([shape[0], shape[3], shape[1], shape[2]])
    tmp1_len = shape[0] * shape[3]
    tmp_len = shape[1] * shape[2]
    data1 = input_data.reshape([tmp1_len, tmp_len])
    data_tmp = np.zeros([tmp1_len, 8], dtype=np.float32)
    data_tmpv = np.zeros([tmp1_len, 8], dtype=np.float32)
    data_tmp, data_tmpv = sum_fp16(data1, tmp_len, data_tmp, data_tmpv, calc_num=8)

    tmp_sum = np.sum(data_tmp, axis=-1).astype('float64')
    tmp_sum_v = np.sum(data_tmpv, axis=-1).astype('float64')
    variance = (tmp_sum_v * shape[1] * shape[2]).astype('float32')
    tmp_rsqrt1 = (tmp_sum ** 2).astype('float32')
    variance = (variance.astype('float64') - tmp_rsqrt1.astype('float64')).astype('float32')
    rsqrt_1 = f32_rsqrt(variance.astype("float32"))

    out = []
    for kk in range(0, shape[3]):
        gamma = weights[kk].astype("float32")
        beta = biases[kk].astype("float32")
        tmp_sum1 = tmp_sum.reshape(shape[0], shape[3], 1, 1)[:, kk, :, :]
        rsqrt1 = rsqrt_1.reshape(shape[0], shape[3], 1, 1)[:, kk, :, :]
        data1 = input_data[:, kk, :, :].astype('float64')
        out1 = (data1 * shape[1] * shape[2]).astype("float32")
        out1 = (out1.astype("float64") - tmp_sum1).astype("float32")
        out1 = (out1.astype("float64") * rsqrt1).astype("float32")
        out1 = (out1.astype("float64") * gamma).astype("float32")
        out1 = (out1.astype("float64") + beta).astype("float32")
        out.append(out1)
    out = np.asarray(out)
    out = out.reshape(shape[3], shape[0], shape[1], shape[2])  # cnhw
    out = np.transpose(out, [1, 0, 2, 3])  # cnhw -> nchw
    out = np.transpose(out, [0, 2, 3, 1])  # nchw -> nhwc
    #gt_data = out.astype(str(self.outputs[0].dtype)).reshape(self.outputs[0].shape)

    out = x3_aiff_fp32_2_fpx(out, is_aiff, fpx=self.outputs[0].dtype)
    gt_data = out.astype(self.outputs[0].dtype.np)
    return gt_data


@register_gen_weight(OpType.InstanceNorm)
def gen_random_weight(self):
    if 'weights' in self.constants:
        weights_shape = list(self.constants['weights'].shape)
        weights_type = str(self.constants['weights'].dtype)
        biases_shape = list(self.constants['biases'].shape)
        biases_type = str(self.constants['biases'].dtype)
        if 'float' in weights_type:
            self.constants['weights'].betensor = np.float16(np.random.randn(*weights_shape) * 10000)
            self.constants['biases'].betensor = np.float32(np.random.randn(*biases_shape) * 1000000)
            if weights_type.startswith('bfloat'):
                self.constants['weights'].betensor = self.constants['weights'].betensor.to(torch.bfloat16)
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


@register_op_metric(OpType.InstanceNorm)
@check_fp_data
def metric(self, outputs, index=None):
    # note: aiff's metric maybe called in layoutconvertor
    def aiff_call(r, u, c, t):
        if "float" in t:
            if np.all(u <= 1) or np.all(c >= 0.999):
                return True
            return np.all(r < 0.001) and np.all(u <= 2) and np.all(c >= 0.9)
        else:
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
