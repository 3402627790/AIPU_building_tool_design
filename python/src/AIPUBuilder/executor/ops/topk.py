import torch
import time
import numpy as np
from AIPUBuilder.core import OpType
from ..engine import register_gt, register_gen_input, _OP_METRICS, register_op_metric
from .gt_util.dtype_utils import torch_to_numpy



@register_gt(OpType.TopK)
def topk(self, *args, **kwargs):
    k = int(self.inputs[1].betensor.item()) if len(self.inputs) == 2 else self.params.get("k")
    axis = self.params['axis']
    largest = self.params['largest']
    issorted = self.params['sorted']
    select_index = self.params['select_index']
    axis = axis if axis >= 0 else self.outputs[0].betensor.ndim+axis

    out1 = torch.zeros_like(self.outputs[0].betensor, dtype=torch.float64)
    out2 = torch.zeros_like(self.outputs[1].betensor, dtype=torch.int64)
    inp_betensors = self.inputs[0].betensor

    if select_index == 'first':
        transp_inp = torch.transpose(inp_betensors, axis, -1)
        out_order = torch.sort(transp_inp, dim=-1, descending=largest, stable=True)
        transp_out = out_order[0][..., 0:k], out_order[1][..., 0:k]
        recover_out0 = torch.transpose(transp_out[0], axis, -1)
        recover_out1 = torch.transpose(transp_out[1], axis, -1)

        [out1, out2] = recover_out0, recover_out1
    else:  # last mode
        transp_inp = torch.transpose(inp_betensors, axis, -1)
        out_order = torch.sort(transp_inp, dim=-1, descending=largest, stable=True)
        out_order = reorder_index(out_order)
        transp_out = out_order[0][..., 0:k], out_order[1][..., 0:k]

        recover_out0 = torch.transpose(transp_out[0], axis, -1)
        recover_out1 = torch.transpose(transp_out[1], axis, -1)

        [out1, out2] = recover_out0, recover_out1
    if len(self.inputs) == 2:
        in_shape = self.inputs[0].betensor.shape
        in_size = self.inputs[0].betensor.numel()
        out_size = out1.numel()
        out1 = torch.concat((out1.flatten(), torch.zeros((in_size - out_size), dtype=torch.float64))).reshape(list(self.outputs[0].shape))
        out2 = torch.concat((out2.flatten(), torch.zeros((in_size - out_size), dtype=torch.int64))).reshape(list(self.outputs[1].shape))
    self.outputs[0].betensor = out1.to(self.outputs[0].betensor_type) if self.outputs[0].dtype.is_float else out1
    self.outputs[1].betensor = out2.to(self.outputs[1].betensor_type) if self.outputs[1].dtype.is_float else out2
    return [out1, out2]


def reorder_index(origin_order):
    values = origin_order[0]
    index = origin_order[1]
    axis_dim = values.shape[-1]
    all_dim = values.numel()
    length = all_dim//axis_dim
    new_shape = (length, axis_dim)
    new_values = values.reshape(new_shape)
    new_index = index.reshape(new_shape)
    idx = 0
    while idx < length:
        tmp_values = new_values[idx, :]
        tmp_index = new_index[idx, :]
        l = 0
        while l < tmp_values.shape[0]:
            equal_value = torch.where(tmp_values == tmp_values[l])[0]  # tmp_values == tmp_values[l]
            equal_len = len(equal_value)
            if equal_len > 1:
                equal_index = tmp_index[l:l+equal_len]
                equal_index = torch.flip(equal_index, dims=[0])
                tmp_index[l:l+equal_len] = equal_index
                l += equal_len
            else:
                l += 1
        new_index[idx, :] = tmp_index
        idx += 1
    new_values = torch.reshape(new_values, values.shape)
    new_index = torch.reshape(new_index, values.shape)
    return [new_values, new_index]


@register_gen_input(OpType.TopK)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_type0 = str(self.inputs[0].dtype)
    if self.inputs[0].dtype.is_float:
        input0 = (1000 * np.random.random(in_shape0)).astype(in_type0)
    else:
        input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)

    if len(self.inputs) == 2:  # topk_v2
        k = self.params['k']
        input1 = np.asarray([k], dtype=str(self.outputs[1].dtype))
        input_data = [input0, input1]
        self.inputs[0].betensor = input_data[0]
        self.inputs[1].betensor = input_data[1]
    else:
        input_data = [input0]
        self.inputs[0].betensor = input_data[0]
    return input_data


@register_op_metric(OpType.TopK)
def metric(self, outputs, index=None):
    if len(self.inputs) == 2:
        k = int(self.inputs[1].betensor.item())
        in_shape = list(self.inputs[0].betensor.shape)
        axis = self.params['axis']
        out_shape = in_shape.copy()
        out_shape[axis] = k
        out_size = np.prod(out_shape)

        if index is None:
            groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
        else:
            groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]
        gts = groundtruths

        valid_gts = [gt.flatten()[: out_size].reshape(out_shape) for gt in gts]
        valid_outputs = [output.flatten()[: out_size].reshape(out_shape) for output in outputs]
        diff_gts = [gt - output for gt, output in zip(valid_gts, valid_outputs)]
        diff_gt = np.mean(np.asarray(diff_gts))
        if np.any(diff_gt != 0):
            print('diff gt: ', diff_gt)
            return False
        return True

    return _OP_METRICS['_all'](self, outputs, index)
