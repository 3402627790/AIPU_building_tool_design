import torch
from .gt_util import *
from ..engine import register_gt, register_gen_input
from AIPUBuilder.core import OpType


@register_gt(OpType.ZeroFraction)
def ZeroFraction(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float:
        output = zerofraction_fp(self)
    else:
        input_tensor = self.inputs[0].betensor.cpu().numpy().astype('int64')
        output_scale = self.params['output_scale']
        output_type = str(self.outputs[0].dtype)

        batch = self.inputs[0].shape[0]
        output = np.zeros((batch))
        for b in range(batch):
            input = input_tensor[b]
            zero_count = np.size(input) - np.count_nonzero(input)
            input_size = np.size(input)
            zero_ratio = (zero_count * output_scale + input_size // 2) // input_size
            output[b] = zero_ratio
        output = clamp(output, self.outputs[0].dtype)

    self.outputs[0].betensor = output
    return [o.betensor for o in self.outputs]


def zerofraction_fp(self):
    input_tensor = torch_to_numpy(self.inputs[0].betensor, self.inputs[0].dtype.np)
    batch = self.inputs[0].shape[0]
    output = np.zeros((batch), dtype='float32')
    for b in range(batch):
        input = input_tensor[b]
        zero_count = np.size(input) - np.count_nonzero(input)
        input_size = np.size(input)
        zero_ratio = zero_count / input_size
        output[b] = zero_ratio
    return output


@register_gen_input(OpType.ZeroFraction)
def gen_input_data(self):
    in_shape = list(self.inputs[0].shape)
    in_type = str(self.inputs[0].dtype)
    batch = in_shape[0]
    input = np.zeros(shape=in_shape, dtype=in_type)
    for b in range(batch):
        input_size = np.prod(in_shape[1:])
        zero_num = np.random.randint(0, input_size)
        max_zero_num = 2 ** 15 - 1 if '16' in in_type else 2 ** 23 - 1  # make sure zero_num * out_scale <= 2 ** 31 - 1
        zero_num = min(zero_num, max_zero_num)
        if self.inputs[0].dtype.is_float:
            input0 = np.random.uniform(-10000, 10000, size=in_shape[1:]).flatten().astype(in_type)
        else:
            input0 = np.random.randint(np.iinfo(in_type).min,
                                       np.iinfo(in_type).max,
                                       size=in_shape[1:],
                                       dtype=in_type).flatten()
        input0[input0 == 0] = 2
        input0[: zero_num] = 0
        np.random.shuffle(input0)
        input0 = input0.reshape(in_shape[1:])
        input[b] = input0
    self.inputs[0].betensor = input
    return [input]
