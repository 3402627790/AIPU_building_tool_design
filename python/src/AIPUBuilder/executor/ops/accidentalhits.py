import torch
import numpy as np
from ..engine import register_gt, register_gen_input, register_op_metric
from AIPUBuilder.core import OpType, Dtype
from .gt_util.quant_utils import clamp
from .gt_util.metric import check_fp_data
from .gt_util.dtype_utils import torch_to_numpy


@register_gt(OpType.AccidentalHits)
def accidental_hits(self, *args, **kwargs):
    # parser has handles this max(32768, len(batch_num*num_true))
    max_out_len = self.outputs[0].shape[0]

    # only support integer of true_classes and sampled_candidate
    true_classes = self.inputs[0].betensor.to(torch.int64)
    sampled_candidate = self.inputs[1].betensor.to(torch.int64)
    batch_size = true_classes.shape[0]
    num_true = true_classes.shape[1]
    dev = true_classes.device
    out_len = 0
    out1 = torch.zeros(max_out_len, device=dev)
    out2 = torch.zeros(max_out_len, device=dev)

    outt1, outt2 = [], []
    yy = sampled_candidate.unsqueeze(0).repeat(true_classes.shape[1], 1)
    for d in range(true_classes.shape[0]):
        xx = true_classes[d].unsqueeze(-1).repeat(1, yy.shape[1])
        idx = torch.nonzero(torch.eq(xx, yy))
        if idx.numel() > 0:
            outt2.append(idx[:, 1])
            outt1.append(torch.ones_like(idx[:, 1]) * d)
    # print(time.time()-aa,outt1,outt2)  # 0.004672050476074219s

    if outt1:
        outt1 = torch.hstack(outt1)
        outt2 = torch.hstack(outt2)
        out_len = min(len(outt1), max_out_len)
        out1[:out_len] = outt1[:out_len]
        out2[:out_len] = outt2[:out_len]

    self.outputs[0].betensor = out1
    self.outputs[1].betensor = out2
    self.outputs[2].betensor = torch.tensor([out_len])

    return [o.betensor for o in self.outputs]


@register_gen_input(OpType.AccidentalHits)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)

    if self.inputs[0].dtype.is_float:
        input0 = np.random.uniform(-1000, 1000, size=in_shape0).astype(in_type0)
    else:
        input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)
    input0_copy = input0.copy().flatten()
    np.random.shuffle(input0_copy)
    input1_size = np.prod(in_shape1)
    max_same_num = min(input0.size, input1_size)
    same_num = np.random.randint(0, max_same_num)
    if self.inputs[1].dtype.is_float:
        input1 = np.random.uniform(-1000, 1000, size=in_shape1).astype(in_type1)
    else:
        input1 = np.random.randint(np.iinfo(in_type1).min, np.iinfo(in_type1).max, size=in_shape1, dtype=in_type1)
    input1 = input1.flatten()
    input1[: same_num] = input0_copy[: same_num]
    input1_unique = np.unique(input1)
    more_num = np.size(input1) - np.size(input1_unique)
    while (more_num > 0):
        if self.inputs[1].dtype.is_float:
            more_data = np.random.uniform(-1000, 1000, size=more_num).astype(in_type1)
        else:
            more_data = np.random.randint(np.iinfo(in_type1).min, np.iinfo(in_type1).max, size=more_num, dtype=in_type1)
        input1_unique = np.concatenate([input1_unique, more_data])
        input1_unique = np.unique(input1_unique)
        more_num = np.size(input1) - np.size(input1_unique)

    input1_unique = input1_unique.reshape(in_shape1)
    np.random.shuffle(input1_unique)

    input_data = [input0, input1_unique]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    return input_data


@register_op_metric(OpType.AccidentalHits)
@check_fp_data
def metric(self, outputs, index=None):
    # out = indices, ids, length
    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]

    length = int(groundtruths[-1][0])
    if length == 0:  # no hit
        return True
    gts = groundtruths
    if index is None:
        diff_gt0 = np.mean(np.abs(gts[0][: length] - outputs[0][: length]))
        diff_gt1 = np.mean(np.abs(gts[1][: length] - outputs[1][: length]))
        diff_gt2 = np.mean(np.abs(gts[2] - outputs[2]))
        diff_gt = np.asarray([diff_gt0, diff_gt1, diff_gt2])
        if (diff_gt != 0).any():
            print("avg error per-output:", diff_gt)
            return False
    else:
        if index in [0, 1]:
            diff_gt = np.mean(np.abs(gts[index][: length] - outputs[: length]))
        elif index == 2:
            diff_gt = np.mean(np.abs(gts[index] - outputs))
        if diff_gt != 0:
            return False
    return True
