import torch
from .gt_util import *
from ..engine import register_gt, register_gen_input
from AIPUBuilder.core import OpType


@register_gt(OpType.CollapseRepeated)
def collapserepeated(self, *args, **kwargs):
    labels = self.inputs[0].betensor
    seq_length = self.inputs[1].betensor.long()
    batch = labels.shape[0]
    maxlen = labels.shape[1]
    out_newlen = self.outputs[0].betensor.shape[1]
    label_ones = torch.ones_like(labels[:, :1]).bool()
    label_xor = (labels[:, 1:] != labels[:, :-1])
    label_mask = torch.cat((label_ones, label_xor), dim=1)

    seq_mask = torch.zeros((batch, maxlen), device=labels.device).bool()
    for b in range(batch):
        if seq_length[b] > 0:
            seq_mask[b, :seq_length[b]] = True
    label_mask = torch.bitwise_and(seq_mask, label_mask)

    new_seq_len = torch.sum(label_mask.int(), dim=1, keepdim=False)
    new_maxlen = torch.max(new_seq_len).int().item()
    idx_mask = torch.zeros((batch, new_maxlen), device=labels.device).bool()
    for b in range(batch):
        idx_mask[b, :new_seq_len[b]] = True

    padding_value = -self.inputs[0].quantization.offset
    padding_diff = (0, out_newlen - new_maxlen)

    flat_labels = torch.flatten(labels)
    flat_label_mask = torch.flatten(label_mask)
    flat_idx_mask = torch.flatten(idx_mask)
    idx = torch.arange(idx_mask.numel(), device=labels.device).long()
    output = (torch.zeros_like(flat_idx_mask, dtype=labels.dtype, device=labels.device) + torch.tensor(padding_value, dtype=labels.dtype))\
        .scatter_(0, idx[flat_idx_mask], flat_labels[flat_label_mask])
    output = torch.reshape(output, (batch, new_maxlen))
    self.outputs[0].betensor = torch.nn.functional.pad(output, padding_diff, "constant", padding_value)
    self.outputs[1].betensor = new_seq_len

    return (self.outputs[0].betensor, self.outputs[1].betensor)


@register_gen_input(OpType.CollapseRepeated)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)

    input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)
    length = in_shape0[-1]
    input1 = np.random.randint(1, length + 1, size=in_shape1, dtype=in_type1)
    input_data = [input0, input1]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    return input_data
