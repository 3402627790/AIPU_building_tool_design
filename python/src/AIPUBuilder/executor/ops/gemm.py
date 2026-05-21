import torch
from AIPUBuilder.core import OpType
from .gt_util import *
from .gt_util.quant_utils import _tpc_lut
from ..engine import register_gt, register_gen_input, _OP_METRICS, register_op_metric
from .matmul import matmul
from .eltwise import eltwise


@register_gt(OpType.Gemm)
def Gemm(self, *args, **kwargs):
    alpha = self.params['alpha']
    beta = self.params['beta']
    trans_a = self.params['trans_a']
    trans_b = self.params['trans_b']
    input0 = torch_to_numpy(self.inputs[0].betensor, self.inputs[0].dtype.np)
    input1 = torch_to_numpy(self.inputs[1].betensor, self.inputs[1].dtype.np)

    def transpose(x, is_trans):
        return np.transpose(x) if is_trans else x

    if 'float16' in str(self.outputs[0].dtype):
        input0 = transpose(input0, trans_a).astype('float32')
        input1 = transpose(input1, trans_b).astype('float32')
        print(trans_a, trans_b, input0.shape, input1.shape)
        matmul_output = np.matmul(input0, input1)
        matmul_output = alpha * matmul_output
        if len(self.inputs) == 3:
            input2 = torch_to_numpy(self.inputs[2].betensor, self.inputs[2].dtype.np)
            input2 = beta * np.float32(input2)
            output = matmul_output + input2
            output = np.float16(output)
        else:
            output = np.float16(matmul_output)
    else:
        matmul_node = self.clone()
        matmul_node.params['scale_value'] = 1
        matmul_node.params['shift_value'] = 0
        matmul_node.outputs[0].dtype = np.int16
        matmul_output = matmul(matmul_node)

        if len(self.inputs) == 3:
            eltwise_node = self.clone()
            eltwise_node.params['method'] = 'ADD'
            eltwise_node.params['with_activation'] = 'NONE'
            eltwise_node.inputs[0].betensor = matmul_output
            eltwise_node.inputs[1].betensor = self.inputs[2].betensor
            output = eltwise(eltwise_node)
        else:
            output = matmul_output

    self.outputs[0].betensor = output
    return [o.betensor for o in self.outputs]


@register_gen_input(OpType.Gemm)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)

    if self.inputs[0].dtype.is_float:
        input0 = (np.random.random(size=in_shape0) - 0.5) * 100  # only support small float data now.
        input1 = np.random.random(size=in_shape1)
        if len(self.inputs) == 3:
            in_shape2 = list(self.inputs[2].shape)
            input2 = np.random.random(size=in_shape2)
            input_data = [input0.astype(in_type0), input1.astype(in_type1), input2.astype(in_type0)]
            self.inputs[2].betensor = input_data[2]
        else:
            input_data = [input0.astype(in_type0), input1.astype(in_type1)]
    else:
        input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)
        input1 = np.random.randint(np.iinfo(in_type1).min, np.iinfo(in_type1).max, size=in_shape1, dtype=in_type1)
        if len(self.inputs) == 3:
            in_shape2 = list(self.inputs[2].shape)
            in_type2 = str(self.inputs[2].dtype)
            input2 = np.random.randint(np.iinfo(in_type2).min, np.iinfo(in_type2).max, size=in_shape2, dtype=in_type2)
            input_data = [input0, input1, input2]
            self.inputs[2].betensor = input_data[2]
        else:
            input_data = [input0, input1]

    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    return input_data


@register_op_metric(OpType.Gemm)
@check_fp_data
def metric(self, outputs, index=None):
    def call(r, u, c, t):
        if "float" in t:
            if np.all(u <= 1):
                return True
            return np.all(r < 0.02)
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
