from ..engine import register_gt, register_gen_weight, register_gen_input, register_op_metric
from .gt_util.quant_utils import clamp, _tpc_lut
from AIPUBuilder.core import OpType, DataLayout
from .gt_util.dtype_utils import torch_to_numpy, numpy_to_torch
from .gt_util.metric import relative_diff, ulp_diff, cos_distance, check_fp_data
import torch
import numpy as np


@register_gt(OpType.Pow)
def pow(self, *args, **kwargs):
    temp_layout = None
    if self.inputs[0].dtype.is_int:
        input0, input1 = self.inputs[0].irtensor, self.inputs[1].irtensor
        if 'exponent' in self.params.keys():
            lut_in_bits = 8 if '8' in str(self.inputs[0].dtype) else 16
            in_is_signed = True if str(self.inputs[0].dtype).startswith('int') else False
            out = _tpc_lut(input0, self.constants['lut_log'].betensor, lut_in_bits, in_is_signed)
            out_type = str(self.outputs[0].dtype)
            lut_out_bits = int(out_type[out_type.index('int')+3:])
            output = clamp(out, self.outputs[0].dtype)
        else:
            inp0 = input0.cpu().numpy()
            inp1 = input1.cpu().numpy()
            input_zp = np.int64(inp0.copy())  # negative for -1, positive and 0 for 1
            input_zp = input_zp + self.inputs[0].quantization.offset
            sign = np.sign(input_zp)
            input1 = np.int32(inp1.copy())
            inp0_type = str(self.inputs[0].dtype)
            lut_exp_type = str(self.constants['lut_exp'].dtype)
            in_is_signed = True if inp0_type.startswith('int') else False
            lut_index_bits = 8 if inp0_type.endswith('8') else 16
            ln_out = _tpc_lut(self.inputs[0].betensor, self.constants['lut_log'].betensor,
                              lut_index_bits, in_is_signed).cpu().numpy()
            pow_out = np.multiply(input1, ln_out)
            exp_in = np.round(pow_out * self.params['scale_value'] * 0.5 ** self.params['shift_value'])
            exp_in = np.clip(exp_in, -32768, 32767).astype('int16')
            lut_out_bits = int(lut_exp_type[lut_exp_type.index('int')+3:])
            out_is_signed = True if lut_exp_type.startswith('int') else False
            exp_out = _tpc_lut(torch.from_numpy(exp_in), self.constants['lut_exp'].betensor, 16, True).cpu().numpy()
            sign = sign.astype('float64')
            output = np.power(sign, np.abs(input1)) * exp_out
            output_type = str(self.outputs[0].dtype)
            output = np.clip(output, np.iinfo(output_type).min, np.iinfo(output_type).max)
            output = output.astype(np.int64)
    else:
        output = pow_float(self)
    self.outputs[0].irtensor = output
    return self.outputs[0].betensor


def pow_float(self):
    from ml_dtypes import bfloat16
    output = np.power(self.inputs[0].betensor.double().cpu().numpy().astype(np.float32),
                      self.inputs[1].betensor.double().cpu().numpy().astype(np.float32))
    output = output.astype(self.outputs[0].dtype.np)

    return output


@register_gen_weight(OpType.Pow)
def gen_random_weight(self):
    in_dtype = str(self.inputs[0].dtype)
    if 'int' in in_dtype:
        lut_log_type = str(self.constants['lut_log'].dtype)
        if 'exponent' in self.params:
            lut_log = np.random.randint(np.iinfo(lut_log_type).min, np.iinfo(lut_log_type).max,
                                        size=self.constants['lut_log'].shape, dtype=lut_log_type)
            lut_log.sort()
            self.constants['lut_log'].betensor = lut_log
        else:
            lut_log = np.random.randint(np.iinfo(lut_log_type).min, np.iinfo(lut_log_type).max,
                                        size=self.constants['lut_log'].shape, dtype='int16')
            lut_log.sort()
            if in_dtype.endswith('8'):
                lut_exp = np.random.randint(0, np.iinfo(in_dtype).max, size=self.constants['lut_exp'].shape,
                                            dtype=in_dtype)
            else:
                lut_exp = np.random.randint(0, np.iinfo(in_dtype).max, size=self.constants['lut_exp'].shape,
                                            dtype=in_dtype)
            lut_exp.sort()
            self.constants['lut_log'].betensor = lut_log
            self.constants['lut_exp'].betensor = lut_exp


@register_gen_input(OpType.Pow)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)

    if self.inputs[0].dtype.is_float:
        input0 = (np.random.random(size=in_shape0) - 0.5) * 65504 * 2
        input1 = (np.random.random(size=in_shape0) - 0.5) * 20
        input_data = [np.float16(input0), np.float16(np.round(input1))]
        if in_type0 == 'bfloat16':
            from ml_dtypes import bfloat16
            input_data = [input0.astype(bfloat16), input1.astype(bfloat16)]
    else:
        input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)
        input1 = np.random.randint(np.iinfo(in_type1).min, np.iinfo(in_type1).max, size=in_shape1, dtype=in_type1)
        input_data = [input0, input1]

    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    return input_data

@register_op_metric(OpType.Pow)
@check_fp_data
def metric(self, outputs, index=None):
    def call(r, u, c, t):
        if "float" in t:
            if np.all(u <= 1):
                return True
            else:
                return False
        else:
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
