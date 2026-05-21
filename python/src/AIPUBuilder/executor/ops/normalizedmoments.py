import torch
from .gt_util import *
from ..engine import register_gt
from AIPUBuilder.core import OpType
from .gt_util.quant_utils import _tpc_lut


@register_gt(OpType.NormalizedMoments)
def NormalizedMoments(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float:
        mean_out, var_out = normalizedMoments_float(self)
    else:
        # Calculate sample mean and sample variance
        input_data = [d.betensor.cpu().numpy().astype(np.int64) for d in self.inputs]
        mean_t, var_t, shift_t = input_data
        lut = self.constants['lut'].numpy().astype('int32')
        mean_scale, mean_scale0, mean_scale1 = self.params['mean_scale_value']
        var_scale, var_scale0, var_scale1 = self.params['var_scale_value']
        mean_shift = self.params['mean_shift_value']
        var_shift = self.params['var_shift_value']

        # mean_out = mean/count + shift
        scaled_mean = mean_t * mean_scale0
        scaled_shift = shift_t * mean_scale1
        shifted_mean = scaled_mean + scaled_shift
        mean_out = np.round(shifted_mean * mean_scale * 0.5 ** mean_shift)
        mean_out = clamp(mean_out, self.outputs[0].dtype)

        # var_out = var/count - (mean/count)^2
        mean_flatten = mean_t.flatten()
        in_type = str(self.inputs[0].dtype)
        qbits = int(in_type[in_type.index('int') + 3:])
        mean_flatten = _tpc_lut(mean_flatten, lut, qbits, is_signed(self.inputs[0].dtype)).cpu().numpy()
        square_mean = mean_flatten.reshape(list(self.inputs[0].shape))
        scaled_var = var_t * var_scale0
        square_mean = square_mean * var_scale1
        var_out = scaled_var - square_mean
        var_out = np.round(var_out * var_scale * 0.5 ** var_shift)
        var_out = clamp(var_out, self.outputs[1].dtype)

    self.outputs[0].betensor = mean_out
    self.outputs[1].betensor = var_out
    return [o.betensor for o in self.outputs]


def normalizedMoments_float(self):
    mean_t = self.inputs[0]
    var_t = self.inputs[1]
    shift_t = self.inputs[2]
    counts = int(self.params['counts'])
    shifted_mean = mean_t.betensor.float() / counts
    mean_out = shifted_mean + shift_t.betensor.float()
    mean_square = shifted_mean * shifted_mean
    var_out = var_t.betensor.float() / counts - mean_square
    return mean_out.half(), var_out.half()
