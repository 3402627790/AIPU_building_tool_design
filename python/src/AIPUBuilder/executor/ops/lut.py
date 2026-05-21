from ..engine import register_gt, register_gen_input, register_gen_weight, _GEN_WEIGHTS, _OP_METRICS, register_op_metric
from AIPUBuilder.core import Target
from .gt_util.quant_utils import clamp, _tpc_lut, linear_dequantize
from .gt_util.math_utils import fexp_tpc
from .gt_util.aiff import _aiff_lut, _aiff_lookup_float_index_lut, _x3_aiff_exp_approximation, x3_aiff_fp32_2_fpx
from .gt_util.metric import relative_diff, ulp_diff, cos_distance, check_fp_data
from .gt_util.dtype_utils import torch_to_numpy
from AIPUBuilder.core import OpType
import torch
import numpy as np


@register_gt(OpType.Rsqrt)
@register_gt(OpType.Sqrt)
@register_gt(OpType.Square)
@register_gt(OpType.Exp)
@register_gt(OpType.Log)
@register_gt(OpType.Reciprocal)
@register_gt(OpType.Round)
@register_gt(OpType.Trunc)
@register_gt(OpType.BNLL)
@register_gt(OpType.Ceil)
@register_gt(OpType.Floor)
@register_gt(OpType.Erf)
def lut(self, *args, **kwargs):
    is_aiff = True
    if 'is_aiff' in kwargs.keys() and not kwargs['is_aiff']:
        is_aiff = False
    if self.inputs[0].dtype.is_float:
        is_perf_mode = self.params.get('is_perf_mode', False)
        lut = self.constants["lut"].betensor if 'lut' in self.constants else None
        if is_perf_mode and not lut is None and is_aiff:
            inp = self.inputs[0].irtensor
            inp = inp.to(torch.float32)
            if self.type.name in ["Log", 'Erf']:
                index_offset_value = self.params.get("index_offset_value", 0)
                index_scale_value = self.params.get("index_scale_value", 1)
                mirror_mode = self.params.get("lut_mode", "") == "MIRROR"
                value_offset_value = self.params.get("value_offset_value", 0)
                outp = _aiff_lookup_float_index_lut(
                    inp,
                    lut,
                    index_scale_value,
                    index_offset_value,
                    mirror_mode,
                    value_offset_value,
                )
            elif self.type.name == "Exp":
                f_vdata = inp * 1.44268798828125
                outp = _x3_aiff_exp_approximation(f_vdata, lut)
            output = x3_aiff_fp32_2_fpx(outp, fpx=self.outputs[0].dtype)
            self.outputs[0].irtensor = output
            return self.outputs[0].irtensor
        else:
            return lut_float(self, is_aiff=is_aiff)
    else:
        if not is_aiff or np.prod((self.constants['lut'].shape)) != 256:
            print('run tpc lut...')
            lut_in_bits = 8 if '8' in str(self.inputs[0].dtype) else 16
            in_is_signed = True if str(self.inputs[0].dtype).startswith('int') else False
            out = _tpc_lut(self.inputs[0].betensor, self.constants['lut'].betensor, lut_in_bits, in_is_signed)
            out = clamp(out, self.outputs[0].dtype)
        else:
            print('run aiff lut...')
            out = _aiff_lut(self.inputs[0].irtensor.cpu().numpy().astype(self.inputs[0].dtype.np),
                            self.constants['lut'].betensor.cpu().numpy().astype(self.constants['lut'].dtype.np),
                            target=Target().name).astype(self.outputs[0].dtype.np)
        self.outputs[0].irtensor = out
    return self.outputs[0].irtensor


def lut_float(self, is_aiff=False):
    inp_tensor = self.inputs[0].irtensor
    if is_aiff:
        inp_tensor = inp_tensor.to(torch.float32)
    output = None
    if self.type.name == 'Exp':
        output = fexp_tpc(inp_tensor)
    elif self.type.name == 'Sqrt':
        output = torch.sqrt(inp_tensor)
    elif self.type.name == 'Square':
        output = torch.square(inp_tensor)
    elif self.type.name == 'Ceil':
        output = torch.ceil(inp_tensor)
    elif self.type.name == 'Floor':
        output = torch.floor(inp_tensor)
    elif self.type.name == 'Round':
        output = torch.round(inp_tensor)
    elif self.type.name == 'Trunc':
        output = torch.trunc(inp_tensor)
    elif self.type.name == 'Log':
        output = torch.log(inp_tensor)
    elif self.type.name == 'Reciprocal':
        output = torch.reciprocal(inp_tensor)
    elif self.type.name == 'Rsqrt':
        output = torch.rsqrt(inp_tensor)
    elif self.type.name == 'Erf':
        output = torch.erf(inp_tensor)
    else:
        raise NotImplementedError
    output = x3_aiff_fp32_2_fpx(output, is_aiff, fpx=self.outputs[0].dtype)
    self.outputs[0].irtensor = output
    return self.outputs[0].irtensor


@register_gen_input(OpType.Log)
@register_gen_input(OpType.Erf)
def gen_input_data(self):
    if self.inputs[0].dtype.is_float:
        in_shape = list(self.inputs[0].betensor.shape)
        is_perf_mode = self.params.get('is_perf_mode', False)
        lut_mode = self.params.get('lut_mode', None)
        lut = self.constants["lut"].betensor if 'lut' in self.constants else None
        if is_perf_mode and not lut is None:
            index_scale = self.params['index_scale_value']
            index_offset = self.params['index_offset_value']
            if lut_mode:
                clip_min = (-511.0+index_offset)/index_scale
                clip_max = (511.0+index_offset)/index_scale
                input_0 = np.random.uniform(clip_min, clip_max, size=in_shape)
                input0 = np.float16(input_0)
                index_0 = np.where(input0 < clip_min)
                index_1 = np.where(input0 > clip_max)
                input0[index_0] = np.float16((clip_min+clip_max)/2)
                input0[index_1] = np.float16((clip_min+clip_max)/2)
            else:
                clip_min = (0.0+index_offset)/index_scale
                clip_max = (511.0+index_offset)/index_scale
                input_0 = np.random.uniform(clip_min, clip_max, size=in_shape)
                input0 = np.float16(input_0)
                index_0 = np.where(input0 < clip_min)
                index_1 = np.where(input0 > clip_max)
                input0[index_0] = np.float16((clip_min+clip_max)/2)
                input0[index_1] = np.float16((clip_min+clip_max)/2)
        else:
            input0 = np.float16(np.random.uniform(-1000, 1000, size=in_shape))
            if self.type.name == 'Erf':
                input0 = np.float16(np.random.uniform(-10, 10, size=in_shape))
        if str(self.inputs[0].dtype).startswith('bfloat'):
            from ml_dtypes import bfloat16
            input0 = input0.astype(bfloat16)
    else:
        input0 = np.random.randint(np.iinfo(self.inputs[0].dtype.np).min, np.iinfo(self.inputs[0].dtype.np).max,
                                   size=self.inputs[0].shape, dtype=self.inputs[0].dtype.np)
    self.inputs[0].betensor = np.int64(input0) if not self.inputs[0].dtype.is_float else input0
    input_data = [input0]

    return input_data


@register_gen_input(OpType.Exp)
def gen_input_data(self):
    if self.inputs[0].dtype.is_float:
        input0 = np.float16(np.random.uniform(-1000, 1000, size=self.inputs[0].shape))
        if str(self.inputs[0].dtype).startswith('bfloat'):
            from ml_dtypes import bfloat16
            input0 = input0.astype(bfloat16)
    else:
        input0 = np.random.randint(np.iinfo(self.inputs[0].dtype.np).min, np.iinfo(self.inputs[0].dtype.np).max,
                                   size=self.inputs[0].shape, dtype=self.inputs[0].dtype.np)
    self.inputs[0].betensor = np.int64(input0) if not self.inputs[0].dtype.is_float else input0
    input_data = [input0]

    return input_data


@register_gen_input(OpType.Rsqrt)
def gen_input_data(self):
    if self.inputs[0].dtype.is_float:
        input0 = np.float16(np.random.uniform(0, 1000, size=self.inputs[0].shape))
        if str(self.inputs[0].dtype).startswith('bfloat'):
            from ml_dtypes import bfloat16
            input0 = input0.astype(bfloat16)
    else:
        input0 = np.random.randint(np.iinfo(self.inputs[0].dtype.np).min, np.iinfo(self.inputs[0].dtype.np).max,
                                   size=self.inputs[0].shape, dtype=self.inputs[0].dtype.np)
    self.inputs[0].betensor = np.int64(input0) if not self.inputs[0].dtype.is_float else input0
    input_data = [input0]

    return input_data


@register_gen_weight(OpType.Log)
@register_gen_weight(OpType.Erf)
@register_gen_weight(OpType.Exp)
def gen_random_weight(self):
    _GEN_WEIGHTS['_all'](self)
    if 'lut' in self.constants.keys() and self.inputs[0].dtype.is_float:
        if self.params.get('lut_mode', None) == 'Exp':
            self.constants['lut'].betensor = np.random.uniform(1, 100, size=self.constants['lut'].shape).astype(
                self.constants['lut'].dtype.np)
        elif self.params.get('is_perf_mode', False):
            index_scale = self.params['index_scale_value']
            index_offset = self.params['index_offset_value']
            lut_items_in_bits = 9
            lut = linear_dequantize(torch.arange(0, 2 ** lut_items_in_bits), index_scale, index_offset)
            self.constants['lut'].betensor = lut
        else:
            self.constants['lut'].betensor = np.random.uniform(-100, 100, size=self.constants['lut'].shape).astype(
                self.constants['lut'].dtype.np)


@register_op_metric(OpType.Rsqrt)
@register_op_metric(OpType.Sqrt)
@register_op_metric(OpType.Square)
@register_op_metric(OpType.Exp)
@register_op_metric(OpType.Log)
@register_op_metric(OpType.Reciprocal)
@register_op_metric(OpType.Round)
@register_op_metric(OpType.Trunc)
@register_op_metric(OpType.BNLL)
@register_op_metric(OpType.Ceil)
@register_op_metric(OpType.Floor)
@register_op_metric(OpType.Erf)
@register_op_metric(OpType.Sine)
@register_op_metric(OpType.Cosine)
@register_op_metric(OpType.Sinh)
@register_op_metric(OpType.Cosh)
@register_op_metric(OpType.Asin)
@register_op_metric(OpType.Acos)
@register_op_metric(OpType.Asinh)
@register_op_metric(OpType.Acosh)
@register_op_metric(OpType.Atan)
@register_op_metric(OpType.Atanh)
@register_op_metric(OpType.Tan)
@check_fp_data
def metric(self, outputs, index=None):
    ulp_dict_tpc = {'Erf': 5}
    ulp_dict_aiff = {'Cosh': 5, 'Asin': 4, 'Erf': 2, 'Log': 2, 'Asinh': 2, 'Atan': 2}

    def call(r, u, c, t, o, gt):
        if "float" in t:
            op_type = str(self.type).split('OpType.')[-1]
            if not self.params.get('is_perf_mode', False):  # tpc
                max_ulp = ulp_dict_tpc.get(op_type, 1)
            else:
                max_ulp = ulp_dict_aiff.get(op_type, 1)
                if op_type == 'Exp':
                    return c > 0.999
            if np.all(u <= max_ulp):
                return True
            if self.params.get('is_perf_mode', False):
                return c > 0.999 and c < 1.001
            return np.all(r < 0.001) or np.all(np.abs(o - gt)[np.where(r > 0.001)] < 1e-4)
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
        return all(call(r, u, c, t, o, gt) for r, u, c, t, o, gt in
                   zip(rel_diffs, ulp_diffs, cos_dists, layer_dtypes, outputs, groundtruths))

    return _OP_METRICS['_all'](self, outputs, index)
