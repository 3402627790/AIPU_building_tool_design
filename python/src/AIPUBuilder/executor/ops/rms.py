import numpy as np
from ..engine import register_gt, register_gen_weight, _OP_METRICS, register_op_metric
from AIPUBuilder.core import OpType
from .gt_util.metric import relative_diff, ulp_diff, cos_distance, check_fp_data
from .gt_util.dtype_utils import torch_to_numpy


@register_gt(OpType.RMSRsqrt)
def rms(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float:
        outputs = rmsrsqrt_float(self)
    else:
        norm_shift = self.params['norm_shift_value']
        norm_scale = self.params['norm_scale_value']
        # input_data = self.inputs[0].cpu().numpy().astype('int64')
        input_data = self.inputs[0].betensor.cpu().numpy().astype(np.int64)
        axis = [a + input_data[0].ndim if a < 0 else a for a in self.params['axis']]
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

        outputs = np.clip(normalized_q, np.iinfo(output_type).min, np.iinfo(output_type).max).astype(output_type)
        outputs = outputs.reshape(self.outputs[0].shape)
    self.outputs[0].betensor = outputs


@register_gen_weight(OpType.RMSRsqrt)
def gen_random_weight(self):
    if 'weights' in self.constants:
        weights_shape = list(self.constants['weights'].shape)
        weights_type = str(self.constants['weights'].dtype)
        if 'float' in weights_type:
            self.constants['weights'].betensor = np.float16(np.random.randn(*weights_shape) * 10000)
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


def rmsrsqrt_float(self):
    input_data = self.inputs[0].betensor.cpu().numpy().astype('float32')
    axis = [a + input_data[0].ndim if a < 0 else a for a in self.params['axis']]
    variance = np.mean(input_data ** 2, axis=tuple(axis), keepdims=True)
    rsqrt_1 = 1 / np.sqrt(variance)
    normalized_q = input_data * rsqrt_1
    out = normalized_q .astype(self.outputs[0].dtype.np).reshape(self.outputs[0].shape)
    return out


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


@register_op_metric(OpType.RMSRsqrt)
@check_fp_data
def metric(self, outputs, index=None):
    def call(r, u, c, t):
        if "float" in t:
            if np.all(u <= 1):
                return True
            return np.all(r <= 1e-4) and np.all(u <= 4)
        else:
            return np.all(r == 0)

    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]

    layer_dtypes = [str(o.dtype) for o in self.outputs] if index is None else [str(self.outputs[index].dtype)]

    rel_diffs = relative_diff(outputs, groundtruths)
    ulp_diffs = ulp_diff(outputs, groundtruths)
    cos_dists = cos_distance(outputs, groundtruths)
    if self.outputs[0].dtype.is_float:
        return all(call(r, u, c, t) for r, u, c, t in zip(rel_diffs, ulp_diffs, cos_dists, layer_dtypes))

    return _OP_METRICS['_all'](self, outputs, index)
