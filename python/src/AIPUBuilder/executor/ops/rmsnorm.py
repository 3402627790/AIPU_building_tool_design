import numpy as np
import torch
from ..engine import register_gt, register_gen_weight, _GEN_WEIGHTS, _OP_METRICS, register_op_metric
from .gt_util import *
from AIPUBuilder.core import OpType, Target
from .gt_util import aiff_itp_acc_channel


@register_gt(OpType.RMSNorm)
def rmsnorm(self, *args, **kwargs):
    sgattrs = kwargs.get("sgattrs", {})
    target = Target().name
    outputs = torch.zeros_like(self.outputs[0].betensor)
    if self.inputs[0].dtype.is_float:
        is_aiff = kwargs.get('is_aiff', False)
        tpc_or_aiff = 'aiff' if is_aiff else 'tpc'
        print('run rmsnorm fp: %s version !' % tpc_or_aiff)
        float_func = rmsnorm_float_aiff if is_aiff else rmsnorm_float

        input_data = self.inputs[0].irtensor.to(torch.float32)
        outputs = float_func(self, input_data)

        outputs = outputs.to(self.outputs[0].dtype.torch)
    elif sgattrs.get("dequantize_int_ir", False) == True:
        is_aiff = kwargs.get('is_aiff', False)
        tpc_or_aiff = 'aiff' if is_aiff else 'tpc'
        print('run rmsnorm dequantize_int_ir: %s version !' % tpc_or_aiff)

        input_data = self.inputs[0].irtensor.to(torch.float32)
        if is_aiff:
            outputs = rmsnorm_dequant_aiff(self, input_data)
            outputs = outputs.to(self.outputs[0].dtype.torch)
        else:
            in_scale = self.inputs[0].quantization.scale
            in_zp = self.inputs[0].quantization.offset
            input_data = (input_data + in_zp) / in_scale

            outputs = rmsnorm_float(self, input_data)

            out_scale = self.outputs[0].quantization.scale
            out_zp = self.outputs[0].quantization.offset
            outputs = outputs * out_scale - out_zp
            info = np.iinfo(self.outputs[0].dtype.np)
            outputs = np.clip(np.round(outputs), info.min, info.max)

            outputs = outputs.to(self.outputs[0].dtype.torch)
    else:
        norm_shift = self.params['norm_shift_value']
        norm_scale = self.params['norm_scale_value']
        input_data = self.inputs[0].irtensor.cpu().numpy().astype(np.int64)

        axis = [a + input_data[0].ndim if a < 0 else a for a in self.params['axis']]
        if target.upper() in ["X3_1304", "X3_1304MP4"] and len(self.params['axis']) == 1:
            axis = [3]
        variance = np.trunc(np.mean(input_data ** 2, axis=tuple(axis), keepdims=True))
        rsqrt_1 = np.uint32(rsqrt(np.uint32(variance), self.constants['lut'].numpy()))
        ngama = rsqrt_1.reshape(np.shape(variance)).astype(np.int64)
        ngama = np.round(ngama * norm_scale * (0.5 ** (norm_shift)))

        output_type = self.outputs[0].dtype.np

        clip_type = 'uint16'
        ngamma_q = np.clip(ngama, np.iinfo(clip_type).min, np.iinfo(clip_type).max).astype(clip_type)

        normalized_q = input_data * ngamma_q
        normalized_q = normalized_q.astype(np.int64)
        # TODO: check opt: normalized = torch.clip((normalized+128)>>8,qmin,qmax)
        normalized_q = np.round(normalized_q * 0.5 ** 8)

        normalized_q = np.clip(normalized_q, np.iinfo(output_type).min, np.iinfo(output_type).max).astype(output_type)
        outputs = normalized_q  # method=DEFAULT_ALL for dprnn

        gamma_shape = [input_data.shape[a] if a in axis else 1 for a in range(len(input_data.shape))]
        gamma = self.constants['weights'].numpy().astype('int64')
        gamma = gamma.reshape(gamma_shape)

        outputs_q = gamma * normalized_q
        outputs_q = outputs_q.astype(np.int64)

        # outputs_q * scale >> (shift - beta_shift)
        # per-channel
        if 'scale_shape' in self.params:
            scale = self.constants['scale'].numpy()
            shift = self.constants['shift'].numpy()
        else:
            shift = self.params['shift_value']
            scale = self.params['scale_value']
        # outputs_q = outputs_q.reshape(self.inputs[0].shape)
        if self.inputs[0].dtype.bits == 16:
            tmph = outputs_q >> 16
            tmpl = outputs_q - (tmph << 16)
            tmph = (tmph * scale).astype(np.int64)
            tmpl = (tmpl * scale).astype(np.int64)
            if isinstance(shift, np.ndarray):
                tmpl = np.where(shift > 0, tmpl >> shift,  tmpl)
                tmpl = np.where(shift < 0, tmpl << (-shift),  tmpl)
            else:
                if shift > 0:
                    tmpl = tmpl >> shift
                else:
                    # print(tmpl, shift, beta_shift)
                    # tmpl = (tmpl / (2 ** shift))
                    tmpl = tmpl << (-shift)
            shift -= 16
            if isinstance(shift, np.ndarray):
                tmph = np.where(shift > 0, tmph >> shift,  tmph)
                tmph = np.where(shift < 0, tmph << (-shift),  tmph)
            else:
                if shift > 0:
                    tmph = tmph >> shift
                else:
                    # print(tmpl, shift, beta_shift)
                    # tmpl = (tmpl / (2 ** shift))
                    tmph = tmph << (-shift)
            outputs_q = tmph + tmpl
        else:
            outputs_q = outputs_q * scale
            outputs_q = np.floor(outputs_q * (0.5 ** shift))
        outputs = outputs_q
        outputs -= self.outputs[0].quantization.offset
        outputs = np.clip(outputs, np.iinfo(output_type).min, np.iinfo(output_type).max).astype(output_type)
    self.outputs[0].irtensor = outputs
    return outputs


@register_gen_weight(OpType.RMSNorm)
def gen_random_weight(self):
    if 'weights' in self.constants:
        weights_shape = list(self.constants['weights'].shape)
        weights_type = str(self.constants['weights'].dtype)
        if 'float' in weights_type:
            self.constants['weights'].betensor = np.float16(np.random.randn(*weights_shape) * 10)
            # self.constants['weights'].betensor = np.ones(weights_shape, "float16")
            if weights_type.startswith('bfloat'):
                self.constants['weights'].betensor = self.constants['weights'].betensor.to(torch.bfloat16)
        else:
            self.constants['weights'].betensor = np.random.randint(-128, 127, size=weights_shape, dtype=weights_type)
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


def rmsnorm_float(self, input_data):
    axis = [a + input_data.ndim if a < 0 else a for a in self.params['axis']]

    variance = torch.mean((input_data) ** 2, axis=tuple(axis), keepdims=True)
    rsqrt_1 = 1 / torch.sqrt(variance + self.params['epsilon'])
    normalized_q = input_data * rsqrt_1
    gamma = self.constants['weights'].numpy()

    if '_output_shapes' in self.attrs:
        out_shape = self.attrs['_output_shapes'][0]
        out_shape = transfer_shape(out_shape, self.outputs[0].shape.layout.name, 'NHWC') \
            if self.outputs[0].shape.layout.name not in ['Flat', 'NHWC'] else out_shape
        ds_weights_shape = [out_shape[a] for a in self.params['axis']]
        ds_weights_size = np.prod(ds_weights_shape)
        gamma = gamma.flatten()[:ds_weights_size].reshape(ds_weights_shape)
    o_shape = list(normalized_q.shape)
    temp_perm_axis = []
    for index, s in enumerate(o_shape):
        if index not in axis:
            temp_perm_axis.append(index)
    temp_perm_axis += axis
    normalized_q = normalized_q.permute(temp_perm_axis)
    out = normalized_q * gamma
    out = out.permute(list(np.argsort(temp_perm_axis))) #invert the transpose
    return out


def rmsnorm_float_aiff(self, input_data):
    c = input_data.shape[-1]
    input_data = fp32_to_fp24(input_data)
    axis = [a + input_data.ndim if a < 0 else a for a in self.params['axis']]
    power2 = fp32_to_fp24(input_data ** 2)
    variance = aiff_itp_acc_channel(power2) / c
    rsqrt_1_fp32 = f32_rsqrt(variance + self.params['epsilon'])
    rsqrt_1_fp24 = fp32_to_fp24(rsqrt_1_fp32)
    normalized_q = fp32_to_fp24(input_data * rsqrt_1_fp24)
    gamma = self.constants['weights'].betensor.to(torch.float32)

    if '_output_shapes' in self.attrs:
        out_shape = self.attrs['_output_shapes'][0]
        out_shape = transfer_shape(out_shape, self.outputs[0].shape.layout.name, 'NHWC') \
            if self.outputs[0].shape.layout.name not in ['Flat', 'NHWC'] else out_shape
        ds_weights_shape = [out_shape[a] for a in self.params['axis']]
        ds_weights_size = np.prod(ds_weights_shape)
        gamma = gamma.flatten()[:ds_weights_size].reshape(ds_weights_shape)
    output = fp32_to_fp24(normalized_q * fp32_to_fp24(gamma))
    return output


def rmsnorm_dequant_aiff(self, input_data):
    c = input_data.shape[-1]

    in_scale = self.inputs[0].quantization.scale
    in_zp = self.inputs[0].quantization.offset
    dequant_input_data = (input_data + in_zp) * fp32_to_fp24(1.0 / in_scale)

    dequant_input_data = fp32_to_fp24(dequant_input_data)
    axis = [a + input_data.ndim if a < 0 else a for a in self.params['axis']]
    power2 = fp32_to_fp24(dequant_input_data ** 2 + self.params['epsilon'])
    variance = aiff_itp_acc_channel(power2)
    rsqrt_1_fp32 = f32_rsqrt(variance)
    rsqrt_1_fp24 = fp32_to_fp24(rsqrt_1_fp32, True)

    normalized_q = fp32_to_fp24((input_data + in_zp) * fp32_to_fp24(math.sqrt(c), True))
    gamma = self.constants['weights'].betensor.to(torch.float32)
    out_scale = self.outputs[0].quantization.scale
    out_zp = self.outputs[0].quantization.offset

    if '_output_shapes' in self.attrs:
        out_shape = self.attrs['_output_shapes'][0]
        out_shape = transfer_shape(out_shape, self.outputs[0].shape.layout.name, 'NHWC') \
            if self.outputs[0].shape.layout.name not in ['Flat', 'NHWC'] else out_shape
        ds_weights_shape = [out_shape[a] for a in self.params['axis']]
        ds_weights_size = np.prod(ds_weights_shape)
        gamma = gamma.flatten()[:ds_weights_size].reshape(ds_weights_shape)

    gamma = gamma * (out_scale / in_scale)
    gamma = fp32_to_fp24(gamma, True)
    normalized_q = fp32_to_fp24(normalized_q * gamma)
    output = fp32_to_fp24(normalized_q * rsqrt_1_fp24) - out_zp
    info = np.iinfo(self.outputs[0].dtype.np)
    output = np.clip(np.round(output), info.min, info.max)

    return output


def D_UTIL_norm_l(L_var1):
    var_out = np.zeros_like(L_var1, dtype='int32')
    for k in range(0, 30):
        indices = np.where(L_var1 < 0x40000000)
        L_var1[indices] = (L_var1[indices] << 1)
        var_out[indices] += 1
    var_out = 31 - var_out
    return var_out, L_var1


def D_UTIL_normalised_inverse_sqrt(frac, exp, lut):
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


def D_UTIL_inverse_sqrt(L_x, lut):
    exp, L_x = D_UTIL_norm_l(L_x)
    L_x, exp = D_UTIL_normalised_inverse_sqrt(L_x, exp, lut)
    L_y = L_x >> (exp)

    return L_y


def rsqrt(x, lut):
    t = (np.int32(D_UTIL_inverse_sqrt(x, lut)) + 32768) >> 16  # / 0x10000
    return t


@register_op_metric(OpType.RMSNorm)
@check_fp_data
def metric(self, outputs, index=None):
    def call(r, u, c, t):
        target = Target().name
        if "float" in t:
            if np.all(u <= 1):
                return True
            return np.all(r <= 1e-4) and np.all(u <= 4)
        if target.startswith(("X2", "X3")):
            print("X2 X3 qdq branch!!!!")
            return np.all(u <= 1)
        return np.all(u == 0)

    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]

    layer_dtypes = [str(o.dtype) for o in self.outputs] if index is None else [str(self.outputs[index].dtype)]

    rel_diffs = relative_diff(outputs, groundtruths)
    ulp_diffs = ulp_diff(outputs, groundtruths)
    cos_dists = cos_distance(outputs, groundtruths)
    return all(call(r, u, c, t) for r, u, c, t in zip(rel_diffs, ulp_diffs, cos_dists, layer_dtypes))
