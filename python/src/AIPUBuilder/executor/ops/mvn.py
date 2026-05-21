import torch
from .gt_util import *
from ..engine import register_gt, register_gen_weight, _OP_METRICS, register_op_metric
from AIPUBuilder.core import OpType
from .gt_util.dtype_utils import torch_to_numpy, numpy_to_torch


@register_gt(OpType.MVN)
def MVN(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float:
        outputs = mvn_float(self)
        outputs = numpy_to_torch(outputs, self.outputs[0].dtype.np)
        self.outputs[0].betensor = outputs
    else:
        lut = self.constants['lut'].numpy()
        input_data = self.inputs[0].betensor.cpu().numpy().astype(np.int64)
        input_type = str(self.inputs[0].dtype)
        output_type = str(self.outputs[0].dtype)

        axis = self.params['axis']
        norm_shift = self.params['shift_value']
        norm_scale = self.params.get('scale_value', 1)
        norm_scale = 1 if norm_scale == None else norm_scale
        var_shift_value = self.params.get('var_shift_value', 0)
        if input_type.endswith('8'):
            mean = np.round(np.mean(input_data, axis=tuple(axis), keepdims=True))
            variance = np.trunc(np.sum((input_data - mean) ** 2, axis=tuple(axis), keepdims=True)).astype(
                'int64') >> var_shift_value
            mean = mean.astype('int64')
            rsqrt_1 = np.uint32(rsqrt(lut, np.uint32(variance)))
            ngama = rsqrt_1.reshape(np.shape(variance)).astype(np.int64)
        else:
            mean = np.round(np.mean(input_data, axis=tuple(axis), keepdims=True))
            variance = np.trunc(np.mean((input_data - mean) ** 2, axis=tuple(axis), keepdims=True))
            mean = mean.astype('int64')
            rsqrt_1 = np.uint32(rsqrt(lut, np.uint32(variance)))
            ngama = rsqrt_1.reshape(np.shape(variance)).astype(np.int64)
            ngama = np.round(ngama * norm_scale * (0.5 ** (norm_shift)))

        clip_type = 'uint16'
        sqrt_q = np.clip(ngama, np.iinfo(clip_type).min, np.iinfo(clip_type).max).astype(clip_type)

        if input_type.endswith('16'):
            normalized_q = (input_data - mean) >> 1
            normalized_q = normalized_q * sqrt_q
            normalized_q = normalized_q.astype(np.int64)
            normalized_q = np.round(normalized_q * 0.5 ** 7)
        else:
            normalized_q = (input_data - mean) * sqrt_q
            normalized_q = normalized_q.astype(np.int64)
            normalized_q = np.round(normalized_q * norm_scale * 0.5 ** (8 + norm_shift))

        normalized_q = np.clip(normalized_q, np.iinfo(output_type).min, np.iinfo(output_type).max).astype('int64')

        normalized_q -= self.outputs[0].quantization.offset
        normalized_q = np.clip(normalized_q, np.iinfo(output_type).min, np.iinfo(output_type).max).astype(output_type)

        self.outputs[0].betensor = normalized_q
    return [o.betensor for o in self.outputs]


def mvn_float(self):
    inp = torch_to_numpy(self.inputs[0].irtensor,  self.inputs[0].dtype.np).astype("float32")
    axis = self.params['axis']
    in_shape = list(self.inputs[0].betensor.shape)
    tmp_len = np.prod([in_shape[a] for a in axis])
    tmp1_len = np.prod(in_shape) // tmp_len

    trans_perm = axis_to_perm(axis, inp.ndim)
    restore_perm = invert_perm(trans_perm)
    new_shape = [inp.shape[trans_perm[i]] for i in range(inp.ndim)]

    input_data = inp.transpose(trans_perm)  # transpose, calc across the last dim
    # data1 = data1.reshape([tmp1_len, tmp_len])  # n-dim -> 2-dim

    if tmp_len <= 16 * 1024:
        data1 = input_data.reshape([tmp1_len, tmp_len])
        data_tmp = np.zeros([tmp1_len, 8], dtype=np.float32)
        data_tmp_square = np.zeros([tmp1_len, 8], dtype=np.float32)
        data_tmp, data_tmp_square = sum_square_fp16(data1, tmp_len, data_tmp, data_tmp_square)
    else:
        # calc sum and square_sum on multi-tec
        tec_num = 4
        max_elts_one_tec = 1023 * 16
        core_total_buffer_elts = (max_elts_one_tec * tec_num)  # calc 511 * 16 * 4 elements at a time
        core_loop_cnts = tmp_len // core_total_buffer_elts
        core_tail_elts = tmp_len % core_total_buffer_elts

        data1 = input_data.reshape([tmp1_len, tmp_len])
        data_tmp = np.zeros([tmp1_len, tec_num, 8], dtype=np.float32)
        data_tmp_square = np.zeros([tmp1_len, tec_num, 8], dtype=np.float32)

        for i in range(core_loop_cnts):
            data_i = data1[:, core_total_buffer_elts * i: core_total_buffer_elts * (i + 1)]
            data_i = data_i.reshape(tmp1_len, tec_num, max_elts_one_tec)
            data_tmp, data_tmp_square = sum_square_fp16(data_i, max_elts_one_tec, data_tmp, data_tmp_square, tec_num=4)

        if core_tail_elts > 0:
            core_tail_elts_per_pe = core_tail_elts // tec_num
            data_tail = data1[:,
                              core_total_buffer_elts * core_loop_cnts: core_total_buffer_elts * core_loop_cnts + core_tail_elts_per_pe * tec_num]
            pad_num = core_tail_elts % tec_num
            if pad_num > 0:
                if core_tail_elts_per_pe > 0:
                    data_tail = data_tail.reshape(tmp1_len, tec_num, core_tail_elts_per_pe)
                    data_tail = np.pad(data_tail, ((0, 0), (0, 0), (0, pad_num)))
                    data_tail[..., 3, -pad_num:] = data1[..., -pad_num:]
                    core_tail_elts_per_pe += pad_num
                else:
                    core_tail_elts_per_pe += core_tail_elts % tec_num
                    data_tail = np.pad(data_tail, ((0, 0), (0, pad_num * 4)))
                    data_tail = data_tail.reshape(tmp1_len, tec_num, core_tail_elts_per_pe)
                    data_tail[..., 3, -pad_num:] = data1[..., -pad_num:]
            data_tail = data_tail.reshape(tmp1_len, tec_num, core_tail_elts_per_pe)
            data_tmp, data_tmp_square = sum_square_fp16(data_tail, core_tail_elts_per_pe, data_tmp, data_tmp_square,
                                                        tec_num=4)

        data_tmp = data_tmp.sum(axis=-2)
        data_tmp_square = data_tmp_square.sum(axis=-2)

    tmp_sum0 = np.sum(inp, axis=tuple(axis), keepdims=True)
    tmp_sum = np.sum(data_tmp, axis=-1, keepdims=True)  # _vrpadd
    tmp_sum = tmp_sum.reshape(tmp_sum0.shape).astype("float64")

    tmp_square_sum = (
        np.sum(data_tmp_square, axis=-1, keepdims=True).reshape(tmp_sum0.shape).astype("float64"))  # _vrpadd

    tmp_rsqrt = (tmp_len * tmp_square_sum).astype("float32")
    tmp_rsqrt1 = (tmp_sum ** 2).astype("float32")
    tmp_rsqrt = (tmp_rsqrt.astype("float64") - tmp_rsqrt1.astype("float64")).astype("float32")
    tmp_rsqrt = f32_rsqrt(tmp_rsqrt).astype('float64')

    tmp_a = (tmp_len * tmp_rsqrt).astype("float32")
    tmp_b = (-tmp_sum * tmp_rsqrt).astype("float32")

    inp = input_data.reshape(new_shape).transpose(restore_perm)  # can be deleted
    from .gt_util.math_utils import fma
    normalized_q = fma(inp, tmp_a, tmp_b)
    gt_data = normalized_q

    return gt_data


def rsqrt(lut, x):
    t = (np.int32(D_UTIL_inverse_sqrt(lut, x)) + 32768) >> 16
    return t


@register_gen_weight(OpType.MVN)
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

# @register_op_metric(OpType.MVN)
# @check_fp_data
# def metric(self, outputs, index=None):
#     def call(r, u, c, t):
#         if "float" in t:
#             if np.all(u <= 1):
#                 return True
#             return np.all(r < 0.001)
#         else:
#             return np.all(r == 0)
#
#     if index is None:
#         groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
#     else:
#         groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]
#
#     layer_dtypes = [str(o.dtype) for o in self.outputs] if index is None else [str(self.outputs[index].dtype)]
#
#     rel_diffs = relative_diff(outputs, groundtruths)
#     ulp_diffs = ulp_diff(outputs, groundtruths)
#     cos_dists = cos_distance(outputs, groundtruths)
#     if self.outputs[0].dtype.is_float:
#         return all(call(r, u, c, t) for r, u, c, t in zip(rel_diffs, ulp_diffs, cos_dists, layer_dtypes))
#
#     return _OP_METRICS['_all'](self, outputs, index)
