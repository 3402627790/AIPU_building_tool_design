from ..engine import register_gt, _OP_METRICS, register_op_metric, register_gen_weight, _GEN_WEIGHTS, register_gen_input, _GEN_INPUTS
from AIPUBuilder.core import OpType, Dtype
import torch
from .gt_util.aiff import _aiff_compression_bias_combin, _aiff_merge_shifts
from .gt_util.aiff import _aiif_ahead_shift_biases, _aiif_ahead_shift_biases_for_merge
from .gt_util.aiff import _aiff_inner_activation, _aiff_final_output, x3_aiff_fp32_2_fpx
from AIPUBuilder.core import Target
from .gt_util.aiff import clip_to_bit, _aiff_wt_dequant
from .gt_util.quant_utils import clamp, fp_activation
from .gt_util.metric import relative_diff, ulp_diff, cos_distance, check_fp_data
from .gt_util.dtype_utils import torch_to_numpy
from .gt_util.math_utils import fp32_to_fp24
from ml_dtypes import float4_e2m1fn, float8_e4m3fn, float8_e5m2
import numpy as np
from .gt_util.gen_weight import INTEGER_RANGE, TYPE_TO_DTYPE


@register_gt(OpType.FullyConnected)
def fullyconected(self, *args, **kwargs):
    self.sgattrs = kwargs['sgattrs'] if 'sgattrs' in kwargs else None
    if self.sgattrs is None:
        self.sgattrs = dict()
    fuse = kwargs['fuse'] if 'fuse' in kwargs.keys() else False

    if self.inputs[0].dtype.is_float:
        return fp_fullyconnected(self, fuse=fuse)

    inp = self.inputs[0].betensor
    weights = self.constants['weights'].betensor
    bias_type = self.constants['biases'].betensor_type
    biases = _aiff_compression_bias_combin(self.constants['biases'].betensor.type(bias_type))

    if 'scale_value' in self.params.keys():
        scale = self.params['scale_value']
        shift = self.params['shift_value']
    else:
        scale = self.constants["scale"].betensor
        shift = self.constants["shift"].betensor
        scale, shift = _aiff_merge_shifts(scale, shift)

    if 'rounding_when_16bit_merge' in self.sgattrs and self.sgattrs['rounding_when_16bit_merge']:
        self.add_bias_after_merge = self.sgattrs['add_bias_after_merge'] if 'add_bias_after_merge' in self.sgattrs else False
        merge_mtp0_c = self.sgattrs['merge_mtp0_c']
        # merge_mtp1_c = self.sgattrs['merge_mtp1_c']
        # merge_mtp1_c = inp.shape[1] - merge_mtp0_c
        last_merge_mtp0_c = self.sgattrs['last_merge_mtp0_c']
        # last_merge_mtp1_c = self.sgattrs['last_merge_mtp1_c']
        # last_merge_mtp1_c = inp.shape[1] - last_merge_mtp0_c
        last_out_c = self.sgattrs['last_out_c']
        # np.split is increased num not independent num

        def merge_fc(inp, w, mtp0_c):
            inps = torch.split(inp, [mtp0_c, inp.shape[1]-mtp0_c], dim=1)
            wses = torch.split(w, [mtp0_c, inp.shape[1]-mtp0_c], dim=1)
            o0 = torch.matmul(inps[0].double(), wses[0].T.double())
            o1 = torch.matmul(inps[1].double(), wses[1].T.double())
            return o0, o1

        if last_out_c < weights.shape[0]:
            weightses = torch.split(weights, [weights.shape[0] - last_out_c, last_out_c], dim=0)
            output00, output01 = merge_fc(inp, weightses[0], merge_mtp0_c)
            output10, output11 = merge_fc(inp, weightses[1], last_merge_mtp0_c)
            output0 = torch.cat([output00, output10], axis=1)
            output1 = torch.cat([output01, output11], axis=1)
        else:
            output0, output1 = merge_fc(inp, weights, merge_mtp0_c)
        # print("%s will use merge method" % self.name)
        outputs, remain_shift = _aiif_ahead_shift_biases_for_merge(
            self, output0, output1, origin_shift=shift, biases=biases)
    else:
        outputs = torch.matmul(inp.double(), weights.T.double())
        outputs, remain_shift = _aiif_ahead_shift_biases(self, outputs, origin_shift=shift, biases=biases)
    outputs, final_scale, final_shift = _aiff_inner_activation(self, outputs, scale, remain_shift)
    # final doquant and -zp
    outputs = _aiff_final_output(self, outputs, final_scale, final_shift)
    if not fuse:
        outputs = clamp(outputs, self.outputs[0].dtype)

    self.outputs[0].betensor = outputs
    return outputs


def fp_fullyconnected(self, fuse=False):
    input_tensor = self.inputs[0].betensor.float()

    weights = torch.tensor(self.constants['weights'].np.astype(np.float32)).to(input_tensor.device)  # [oc, ic]
    weights = weights.permute([1, 0])  # [ic, oc]
    # wt dequant
    weights = _aiff_wt_dequant(weights, self.constants, weights.shape[1], weights.shape[0], kh=1, kw=1)
    weights = weights.permute([1, 0])  # [oc, ic]
    # weights = weights.to(self.inputs[0].dtype.torch).float()

    biases = self.constants['biases'].betensor.float()

    output = torch.nn.functional.linear(input_tensor, weights)
    if 'scale1' in self.constants:
        output = output * self.constants['scale1'].betensor.flatten().float()  # [oc]

    output = output + biases
    output = fp_activation(output, self)

    scale = 1.0
    if 'scale' in self.constants:
        scale = self.constants['scale'].betensor.float()  # [oc]
    elif 'scale_value' in self.params.keys():
        scale = self.params['scale_value']
    output = output * scale

    from copy import deepcopy
    bak_out_dtype = deepcopy(self.outputs[0].dtype)
    if fuse:
        self.outputs[0].dtype = Dtype.FP32
    else:
        output = x3_aiff_fp32_2_fpx(output, fpx=self.outputs[0].dtype)
    self.outputs[0].betensor = output
    if fuse:
        self.outputs[0].dtype = bak_out_dtype
    return output


@register_op_metric(OpType.FullyConnected)
@check_fp_data
def metric(self, outputs, index=None):
    def call(r, u, c, t):
        if "float" in t:
            if np.all(u <= 1) or np.all(r <= 1e-4):
                return True
            else:
                return np.all(c > 0.999)
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


def create_scipy_sparse_matrix_advanced(shape, sparse_rate, distribution='uniform', **kwargs):
    """
    params:
    shape: matrix shape
    sparse_rate:
    distribution: option of ['uniform', 'normal', 'integer', 'exponential']
    **kwargs:
      - uniform: low=0, high=1
      - normal: mean=0, std=1
      - integer: low=0, high=10
      - exponential: scale=1.0

    return:
    the sparse matrix
    """
    from scipy import sparse

    total_elements = np.prod(shape)
    num_non_zero = int(total_elements * (1 - sparse_rate))

    rows = np.random.randint(0, shape[0], num_non_zero)
    cols = np.random.randint(0, shape[1], num_non_zero)

    if distribution == 'uniform':
        low = kwargs.get('low', 0)
        high = kwargs.get('high', 1)
        data = np.random.uniform(low, high, num_non_zero)
    elif distribution == 'normal':
        mean = kwargs.get('mean', 0)
        std = kwargs.get('std', 1)
        data = np.random.normal(mean, std, num_non_zero)
    elif distribution == 'integer':
        low = kwargs.get('low', 0)
        high = kwargs.get('high', 10)
        data = np.random.randint(low, high, num_non_zero)
    elif distribution == 'exponential':
        scale = kwargs.get('scale', 1.0)
        data = np.random.exponential(scale, num_non_zero)
    else:
        raise ValueError(f"not support type: {distribution}")

    sparse_matrix = sparse.csr_matrix((data, (rows, cols)), shape=shape)

    return sparse_matrix


@register_gen_input(OpType.FullyConnected)
def gen_input_data(self):
    if self.inputs[0].dtype == Dtype.FP8E4M3FN:
        input_data = np.random.uniform(-8, 7, size=self.inputs[0].shape).astype(self.inputs[0].dtype.np)
        self.inputs[0].betensor = input_data
        input_datas = [input_data]
        return input_datas
    elif self.inputs[0].dtype == Dtype.FP8E5M2:  # float8_e5m2 data range is greater than float8_e4m3
        input_data = np.random.uniform(-100, 100, size=self.inputs[0].shape).astype(self.inputs[0].dtype.np)
        self.inputs[0].betensor = input_data
        input_datas = [input_data]
        return input_datas
    else:
        return _GEN_INPUTS[OpType.Input](self)


@register_gen_weight(OpType.FullyConnected)
def gen_random_weight(self):
    _GEN_WEIGHTS["_all"](self)
    if self.constants["weights"].dtype == Dtype.ALIGNED_INT4 and \
            self.inputs[0].dtype in [Dtype.FP8E4M3FN, Dtype.FP8E5M2]:
        for w in self.constants:
            params_name = w
            param_shape = list(getattr(self.constants[w], 'shape'))
            param_type = str(getattr(self.constants[w], 'dtype'))
            if params_name == "weights":
                d = create_scipy_sparse_matrix_advanced(
                    param_shape, 0.9, distribution="integer", low=-8, high=7).toarray().astype("int8")
            elif params_name == "biases":
                d = np.random.randint(-8, 7, size=param_shape).astype(param_type)
            elif params_name == "zp0":
                if param_type in ['aligned_int4', 'aligned_int12']:
                    d = np.random.randint(INTEGER_RANGE[param_type][0],
                                          INTEGER_RANGE[param_type][1],
                                          dtype=TYPE_TO_DTYPE[param_type],
                                          size=param_shape)
                else:
                    d = np.random.randint(-2, 2, dtype=param_type, size=param_shape)
                    # d = d * 0
            elif params_name == "scale0":
                d = np.random.uniform(0.9, 0.95, size=param_shape).astype(self.constants[w].dtype.np)
            elif params_name == "scale1":
                d = np.random.uniform(0.9, 0.95, size=param_shape).astype(param_type)
            self.constants[w].betensor = d
