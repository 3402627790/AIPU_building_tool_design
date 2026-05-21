import torch
import numpy as np
import random
from ..engine import register_gt, register_gen_input, _OP_METRICS, register_op_metric
from .gt_util.metric import relative_diff, ulp_diff, cos_distance, check_fp_data
from .gt_util.dtype_utils import torch_to_numpy
from AIPUBuilder.core import OpType, Dtype


@register_gt(OpType.NonZero)
def NonZero(self, *args, **kwargs):
    torch_output = torch.nonzero(self.inputs[0].irtensor)
    num, dim = list(torch_output.shape)
    self.outputs[1].irtensor = torch.tensor([num], dtype=torch.int64, device=torch_output.device)
    torch_output = torch_output.permute(1, 0)
    if '_output_shapes' in self.attrs:
        self.outputs[0].irtensor = torch_output
        return self.outputs[0].betensor, self.outputs[1].betensor
    in_size = self.inputs[0].betensor.numel()
    fill_elements = torch.zeros((in_size - num) * dim, dtype=torch.int64, device=self.outputs[0].betensor.device)
    self.outputs[0].irtensor = torch.concat((torch_output.flatten(), fill_elements)).reshape(dim, in_size)
    return self.outputs[0].betensor, self.outputs[1].betensor


@register_gen_input(OpType.NonZero)
def gen_input_data(self):
    in_shape = self.inputs[0].shape
    in_size = np.prod(in_shape)
    if self.inputs[0].dtype.is_float:
        input0 = (np.random.uniform(-1000, 1000, size=self.inputs[0].shape)).astype(self.inputs[0].dtype.np)
    else:
        input0 = np.random.randint(np.iinfo(self.inputs[0].dtype.np).min, np.iinfo(self.inputs[0].dtype.np).max,
                                   size=self.inputs[0].shape, dtype=self.inputs[0].dtype.np)
    min_z_num = np.random.choice(list(range(in_size)))
    input0 = input0.flatten()
    input0[: min_z_num] = 0
    random.shuffle(input0)
    input0 = input0.reshape(self.inputs[0].shape)
    self.inputs[0].betensor = np.int64(input0) if not self.inputs[0].dtype.is_float else input0
    input_data = [input0]

    return input_data


@register_op_metric(OpType.NonZero)
@check_fp_data
def metric(self, outputs, index=None):
    # out = [[in_dim, in_size], [nz_num]]
    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]
    gts = groundtruths

    nz_num = gts[1][0]
    in_dim = len(self.inputs[0].shape)
    if index is None:
        diff_gt1 = np.mean(np.abs(gts[1] - outputs[1]))
        valid_gt0 = np.abs(gts[0].flatten()[: in_dim * nz_num].reshape(in_dim, nz_num))
        valid_out0 = np.abs(outputs[0].flatten()[: in_dim * nz_num].reshape(in_dim, nz_num))
        diff_gt0 = np.mean(np.abs(valid_gt0 - valid_out0)) if not valid_gt0.size == 0 else 0
        diff_gt = np.asarray([diff_gt0, diff_gt1])
    elif index == 0:
        valid_gt0 = np.abs(gts[0].flatten()[: in_dim * nz_num].reshape(in_dim, nz_num))
        valid_out0 = np.abs(outputs[0].flatten()[: in_dim * nz_num].reshape(in_dim, nz_num))
        diff_gt = np.mean(np.abs(valid_gt0 - valid_out0)) if not valid_gt0.size == 0 else 0
    else:
        diff_gt = np.mean(np.abs(gts[1] - outputs[1]))
    if np.any(diff_gt != 0):
        print('diff gt: ', diff_gt)
        return False
    return True
