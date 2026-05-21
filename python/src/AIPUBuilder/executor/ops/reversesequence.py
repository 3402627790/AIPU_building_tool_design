from ..engine import register_gt, register_gen_input
from AIPUBuilder.core import OpType
import torch


def local_reserve_sequence(inp, lengths, batch_axis, time_axis):

    out = torch.zeros_like(inp, device=inp.device)
    for b in range(lengths.shape[0]):
        seq_len = lengths[b].int()
        temp = torch.index_select(inp, batch_axis, torch.tensor(b, device=inp.device))
        flip_part = torch.index_select(temp, time_axis, torch.arange(seq_len, device=inp.device)).flip(dims=[time_axis])
        reserve_part = torch.index_select(temp, time_axis, torch.arange(
            temp.shape[time_axis] - seq_len, device=inp.device)+seq_len)
        one_out = torch.cat([flip_part, reserve_part], dim=time_axis)
        out = torch.cat([out, one_out], dim=batch_axis)
    out = torch.split(out, inp.shape[batch_axis], dim=batch_axis)[-1]
    return out


@register_gt(OpType.ReverseSequence)
def reversesequence(self, *args, **kwargs):
    inpt = self.inputs[0].irtensor
    batch_axis = self.params['batch_axis']
    time_axis = self.params['time_axis']
    batch_size = inpt.shape[batch_axis]
    time_step = inpt.shape[time_axis]
    seq_lens = [time_step] * batch_size
    if len(self.inputs) == 1:
        seq_lent = torch.tensor([time_step] * batch_size)
    else:
        seq_lent = self.inputs[1].irtensor

    outt = local_reserve_sequence(inpt, seq_lent, batch_axis, time_axis)
    self.outputs[0].irtensor = outt
    return outt


@register_gen_input(OpType.ReverseSequence)
def gen_input_data(self):
    import numpy as np
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape) if len(self.inputs) == 2 else None
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype) if len(self.inputs) == 2 else None

    batch, time_steps = in_shape0[:2]
    input_shape = in_shape0
    input_shape = input_shape + [1] if len(input_shape) == 2 else input_shape
    batch = input_shape[self.params['batch_axis']] if 'batch_axis' in self.params else batch
    time_steps = input_shape[self.params['time_axis']] if 'time_axis' in self.params else time_steps

    if self.inputs[0].dtype.is_float:
        input0 = (np.random.uniform(0, 10000, size=in_shape0)).astype(in_type0)
    else:
        input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)

    if len(self.inputs) == 2:  # 2 input version
        input1 = np.random.randint(0, time_steps, size=in_shape1, dtype=in_type1)
        input_data = [input0, input1]
        self.inputs[1].betensor = input_data[1]
    else:
        input_data = [input0]

    self.inputs[0].betensor = input_data[0]
    return input_data
