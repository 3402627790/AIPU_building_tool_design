import torch
from .gt_util import *
from ..engine import register_gt, register_gen_input
from AIPUBuilder.core import OpType


@register_gt(OpType.Sort)
def Sort(self, *args, **kwargs):
    input = self.inputs[0].betensor.cpu().numpy().astype(str(self.inputs[0].dtype))
    input_type = str(self.inputs[0].dtype)
    if input_type.endswith('16'):
        score_output, keep_output = sort_16bit(self, input, dtype=input_type)
    else:
        score_output, keep_output = sort_8bit(self, input, dtype=input_type)

    self.outputs[0].betensor = score_output.reshape(self.outputs[0].shape)
    self.outputs[1].betensor = keep_output.reshape(self.outputs[1].shape)
    return [o.betensor for o in self.outputs]


def sort_8bit(self, input, dtype='uint8'):
    scores = input.flatten()
    o_size = len(scores)

    # asm: align to 32, c: align to 128
    def cal_zero_num():
        n = 0
        dada_len = scores.size
        if dada_len > 64:
            while (1):
                temp = 64 << n
                if dada_len <= temp:
                    temp2 = temp
                    break
                else:
                    n = n + 1
            return temp2 - dada_len
        else:
            return 128 - dada_len

    zero_num = cal_zero_num()
    pad_value = np.iinfo(dtype).min
    scores = np.concatenate([scores, np.ones([zero_num]) * pad_value])
    order = list(range(len(scores)))
    scores, order = bitonic_sort(len(scores), scores, order)
    scores, order = merge_sort(len(scores), scores, order)
    scores = list(scores)
    order = list(order)
    oo = []
    ss = []
    for o in order:
        if o < o_size:
            oo.append(o)
            ss.append(scores[order.index(o)])

    scores = np.asarray(ss)
    order = np.asarray(oo)
    return scores, order


def sort_16bit(self, input, dtype='uint16'):
    scores = input.flatten()
    o_size = len(scores)

    n = 0
    while 1:
        temp = 64 << n
        if o_size <= temp:
            num_length = temp
            break
        else:
            n = n + 1

    zero_num = num_length - o_size
    if dtype == 'bfloat16':
        bf16_max_value = 3.38953e38
        pad_value = -bf16_max_value
        scores = np.concatenate([scores, np.ones([zero_num]) * pad_value]).astype(bfloat16)
    else:
        pad_value = np.finfo(dtype).min if dtype.startswith('float') else np.iinfo(dtype).min
        scores = np.concatenate([scores, np.ones([zero_num]) * pad_value])
        scores = scores.astype('float16') if dtype.startswith('float') else scores
    order = np.asarray(list(range(num_length)), dtype=np.uint16)

    scores, order = bitonic_sort_16bit(len(scores), scores, order)

    score_output = np.zeros(o_size, dtype=scores.dtype)
    keep_output = np.zeros(o_size, dtype='uint16')

    ii = 0
    for i in range(num_length):
        if order[i] < o_size:
            score_output[ii] = scores[i]
            keep_output[ii] = order[i]
            ii = ii + 1
    return score_output, keep_output


@register_gen_input(OpType.Sort)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_type0 = str(self.inputs[0].dtype)

    seed = np.random.randint(1, 100)
    if seed % 2 == 0:
        input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)
    else:  # all values are the same
        input0 = np.ones(in_shape0, dtype=in_type0) * seed

    input_data = [input0]
    self.inputs[0].betensor = input_data[0]
    return input_data
