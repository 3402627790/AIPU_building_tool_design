import torch
from .gt_util import *
from ..engine import register_gt, _OP_METRICS, register_op_metric
from AIPUBuilder.core import OpType
from .gt_util.dtype_utils import torch_to_numpy, numpy_to_torch


@register_gt(OpType.Moments)
def Moments(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float:
        mean, variance = moments_float(self)
    else:
        inp = self.inputs[0].betensor.cpu().numpy().astype(np.int64)
        input_type = str(self.inputs[0].dtype)
        output_type = str(self.outputs[0].dtype)

        axis = self.params['axis']
        var_shift_value = self.params['var_shift_value']
        var_scale_value = self.params['var_scale_value']
        input_scale_value = self.params['input_scale_value']
        input_shift_value = self.params['input_shift_value']

        if 'keepdims' in self.params.keys() and str(self.params['keepdims']).lower() == 'false':
            keepdims = False
        else:
            keepdims = True

        assert input_scale_value == 1 and input_shift_value == 0, 'only support inp scale,shift == 1,0'

        if input_type.endswith('8'):
            inp = np.round(inp.astype(np.int64) * input_scale_value * 0.5 ** input_shift_value)
            inp = clamp(inp, input_type)
            inp = inp.astype(np.float64)
            mean = np.round(np.mean(inp, axis=tuple(axis), keepdims=keepdims)).astype(np.int64)
            mean_int = np.round(np.mean(inp, axis=tuple(axis), keepdims=True)).astype(np.int64)
            variance = np.mean((inp - mean_int) ** 2, axis=tuple(axis), keepdims=keepdims).astype(np.int64)
            variance = np.round(variance * var_scale_value * (0.5 ** var_shift_value)).astype(np.int64)
            clip_type = 'uint8'
        else:
            inp = (inp.astype(np.int64) * input_scale_value) >> input_shift_value
            inp = clamp(inp, input_type)

            inp = inp.astype(np.float64)
            mean = np.round(np.mean(inp, axis=tuple(axis), keepdims=keepdims)).astype(np.int64)
            mean_int = np.round(np.mean(inp, axis=tuple(axis), keepdims=True)).astype(np.int64)
            variance = np.mean((inp - mean_int) ** 2, axis=tuple(axis), keepdims=keepdims).astype(np.int64)

            var_h = variance >> 16
            var_l = variance - (var_h << 16)
            var_h = (var_h * var_scale_value).astype(np.int64)
            var_l = (var_l * var_scale_value).astype(np.int64)
            var_h = (var_h * (0.5 ** (var_shift_value - 16))).astype(np.int64)
            var_l = (var_l * (0.5 ** var_shift_value)).astype(np.int64)
            variance = var_h + var_l
            clip_type = 'uint16'

        mean = np.clip(mean, np.iinfo(output_type).min, np.iinfo(output_type).max).astype(output_type)

        # var clip to uint8/uint16 in lib
        variance = np.clip(variance, np.iinfo(clip_type).min, np.iinfo(clip_type).max).astype(clip_type)

    self.outputs[0].betensor = mean
    self.outputs[1].betensor = variance
    return [o.betensor for o in self.outputs]


def moments_float(self):
    inp = torch_to_numpy(self.inputs[0].betensor,  self.inputs[0].dtype.np).astype("float32")

    axis = self.params['axis']
    in_shape = list(self.inputs[0].betensor.shape)
    tmp_len = np.prod([in_shape[a] for a in axis])
    tmp1_len = np.prod(in_shape) // tmp_len

    trans_index = list(range(inp.ndim))
    for a in axis:
        trans_index.remove(a)
    trans_index += axis
    data1 = inp.transpose(trans_index).reshape([tmp1_len, tmp_len])
    data_tmp = np.zeros([tmp1_len, 8], dtype=np.float32)
    # align to lib, calc 8 data at a time
    data_tmp = sum_square_fp16(data1, tmp_len, data_tmp)

    if 'keepdims' in self.params.keys() and str(self.params['keepdims']).lower() == 'false':
        keepdims = False
    else:
        keepdims = True
    tmp_sum0 = np.sum(inp, axis=tuple(axis), keepdims=keepdims)
    tmp_sum = np.sum(data_tmp, axis=-1, keepdims=keepdims)
    tmp_sum = tmp_sum.reshape(tmp_sum0.shape)
    mean = tmp_sum / tmp_len
    variance = (np.sum(inp ** 2, axis=tuple(axis), keepdims=keepdims) * tmp_len - tmp_sum ** 2) / (tmp_len ** 2)
    # variance = np.var(inp, axis=tuple(self.axis), keepdims=keepdims)
    # mean = mean.astype('float16')
    # variance = variance.astype('float16')
    mean = numpy_to_torch(mean, self.outputs[0].dtype.np)

    variance = numpy_to_torch(variance, self.outputs[0].dtype.np)

    return mean, variance


@register_op_metric(OpType.Moments)
@check_fp_data
def metric(self, outputs, index=None):
    def call(r, u, c, t):
        if "float" in t:
            if np.all(u <= 2):
                return True
            return np.all(r < 0.001)
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
