from ..engine import register_gt, register_gen_input
from .gt_util.quant_utils import clamp
from AIPUBuilder.core import OpType
import torch
import numpy as np


@register_gt(OpType.CTCGreedyDecoder)
def ctc_greedy_decoder(self, *args, **kwargs):
    inps = self.inputs[0].betensor
    seq_lens = self.inputs[1].betensor
    if 'merge_repeated' in self.params:
        is_merge_repeated = self.params['merge_repeated']
    else:
        is_merge_repeated = True
    batch = inps.shape[0]
    max_value = np.iinfo(str(self.outputs[0].dtype)).max
    tout = torch.full([batch] + [*self.outputs[0].shape][1:], max_value)
    for b in range(batch):
        inp = inps[b]
        seq_len = seq_lens[b].int().item()
        act_inp = inp[:seq_len, :]
        num_classes = inp.shape[1]
        max_idx = torch.argmax(act_inp, dim=-1)
        if is_merge_repeated:
            out = merge_repeated(max_idx)
        else:
            out = max_idx
        out = remove_blank(out, blanks=[num_classes - 1])
        out = out.reshape([*out.shape, 1, 1])
        tout[b, :out.shape[0], :, :] = out
    self.outputs[0].betensor = tout
    return tout


def remove_blank(indexes, blanks=[0]):
    # remove blank
    no_blank_index = indexes
    for blank in blanks:
        no_blank_index = no_blank_index[torch.where(no_blank_index != blank)]
    return no_blank_index


def merge_repeated(indexes):
    '''
        index = [1,2,3,3,3,4,4,7,8,8]
        index_pad = [1,2,3,3,3,4,4,7,8,8,-1]
        pad_index = [-1,1,2,3,3,3,4,4,7,8,8]
        diff = index_pad - pad_index
        diff = [2,1,1,0,0,1,0,1,0,-9]
        diff = diff[:-1]
        according to the nonzero element idx of diff, gather from index, then get merged_repeated result
    '''
    dev = indexes.device
    raw_shape = indexes.shape
    pad_shape = [*raw_shape[:-1], 1]
    pad_index = torch.full(pad_shape, -1, device=dev, dtype=torch.int32)

    indexes_pad = torch.cat([indexes, pad_index], dim=-1)
    pad_indexes = torch.cat([pad_index, indexes], dim=-1)

    diff = indexes_pad - pad_indexes
    diff_no_pad = diff[..., :-1]
    merge_index = indexes[torch.where(diff_no_pad != 0)]
    return merge_index


@register_gen_input(OpType.CTCGreedyDecoder)
def gen_input_data(self):
    import numpy as np
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)

    # real_time_steps stored in 2nd input, real_time_steps <= time_steps in ir
    time_steps = in_shape0[1]
    input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)
    input1 = np.random.randint(1, time_steps + 1, size=in_shape1, dtype=in_type1)

    input_data = [input0, input1]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    return input_data
