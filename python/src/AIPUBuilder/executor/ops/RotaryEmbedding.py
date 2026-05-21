import numpy as np
from ..engine import register_gt, register_gen_weight, register_gen_input, _GEN_WEIGHTS
from AIPUBuilder.core import OpType
import torch
import numpy as np


@register_gt(OpType.RotaryEmbedding)
def rotaryembedding(self, *args, **kwargs):
    input_data = self.inputs[0].betensor.to(torch.float32)
    cos_cache = self.inputs[1].betensor.to(torch.float32)
    sin_cache = self.inputs[2].betensor.to(torch.float32)
    if len(self.inputs) == 4:
        position_ids = self.inputs[3].betensor
    else:
        position_ids = None

    if len(input_data.shape) == 4:
        input_data = torch.transpose(input_data, 1, 2)
    batch_size = input_data.shape[0]
    sequence_length = input_data.shape[1]
    rotary_embedding_dim = self.params["rotary_embedding_dim"]
    interleaved = self.params["interleaved"]
    original_input_shape = input_data.shape
    if len(input_data.shape) == 3:
        hidden_size = input_data.shape[2]
        num_heads = self.params['num_heads']
        head_size = int(hidden_size / num_heads)
        new_shape = [batch_size, sequence_length, num_heads, head_size]
        input_data = torch.reshape(input_data, new_shape)
    head_size = input_data.shape[3]

    if rotary_embedding_dim == 0:
        rotary_embedding_dim = head_size
    rotary_embedding_dim_half = int(rotary_embedding_dim / 2)
    x_rotate = input_data[:, :, :, :rotary_embedding_dim]
    x_not_rotate = input_data[:, :, :, rotary_embedding_dim:]

    if position_ids is not None:
        cos = cos_cache[position_ids]
        sin = sin_cache[position_ids]
    else:
        cos = cos_cache[:, :sequence_length, :]
        sin = sin_cache[:, :sequence_length, :]

    cos = cos[:, :, :rotary_embedding_dim_half]
    sin = sin[:, :, :rotary_embedding_dim_half]
    cos = torch.unsqueeze(cos, dim=2)
    sin = torch.unsqueeze(sin, dim=2)

    if interleaved:
        x1 = x_rotate[:, :, :, 0::2]
        x2 = x_rotate[:, :, :, 1::2]
    else:
        x1, x2 = torch.split(x_rotate, rotary_embedding_dim_half, dim=-1)

    real = cos * x1 - sin * x2
    imag = sin * x1 + cos * x2

    if interleaved:
        real = torch.unsqueeze(real, dim=-1)
        imag = torch.unsqueeze(imag, dim=-1)
        x_rotate_concat = torch.concatenate((real, imag), dim=-1)
        x_rotate = torch.reshape(x_rotate_concat, x_rotate.shape)
    else:
        x_rotate = torch.concatenate((real, imag), dim=-1)
    output = torch.concatenate((x_rotate, x_not_rotate), dim=-1)
    if len(original_input_shape) == 3:
        output = torch.reshape(output, original_input_shape)
    else:
        output = torch.transpose(output, 1, 2)
    self.outputs[0].betensor = output.to(torch.float16)
    return output


@register_gen_input(OpType.RotaryEmbedding)
def gen_random_input(self):
    position_ids_flag = True if len(self.inputs) == 4 else False
    in_shape0 = list(self.inputs[0].shape)
    in_type0 = self.inputs[0].dtype.torch
    batch_size = in_shape0[0]
    input_data = []
    if len(in_shape0) == 4:
        sequence_length = in_shape0[2]
    elif len(in_shape0) == 3:
        sequence_length = in_shape0[1]
    input0 = (2 * torch.rand(size=in_shape0, dtype=in_type0) - 1) * 1000
    input_data.append(input0)

    cache_shape = self.inputs[1].shape
    rotary_embedding_dim = cache_shape[-1] * 2
    if position_ids_flag:
        max_seq_len = cache_shape[0]
    else:
        max_seq_len = cache_shape[1]
    cos_cache, sin_cache = cal_sin_cos_cache(position_ids_flag, batch_size, rotary_embedding_dim, max_seq_len, in_type0)

    input_data.extend([cos_cache, sin_cache])

    if position_ids_flag:
        in_shape1 = list(self.inputs[-1].shape)
        in_type1 = str(self.inputs[-1].dtype)
        input1 = np.array([
            np.random.permutation(sequence_length)  # generate random sequence of [0, 1, ..., sequence_length-1]
            for _ in range(batch_size)
        ])
        input1 = input1.astype(in_type1)
        input_data.append(input1)
    for i, j in enumerate(input_data):
        self.inputs[i].betensor = torch.tensor(input_data[i])
    return input_data


def cal_sin_cos_cache(position_ids_flag, batch_size, rotary_embedding_dim, max_seq_len, d_type):
    dim = rotary_embedding_dim // 2
    # []
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, dtype=d_type) / dim))

    # Create position indexes [max_seq_len]
    position = torch.arange(max_seq_len, dtype=d_type)

    freqs = torch.outer(position, inv_freq)

    cos_cache = torch.cos(freqs)  # [max_seq_len, rotary_embedding_dim // 2]
    sin_cache = torch.sin(freqs)  # [max_seq_len, rotary_embedding_dim // 2]

    if not position_ids_flag:
        position_ids = torch.arange(max_seq_len).unsqueeze(0).expand(batch_size, -1)
        cos_cache = cos_cache[position_ids]
        sin_cache = sin_cache[position_ids]  # [batch_size,max_seq_len, rotary_embedding_dim // 2]

    cos_cache = cos_cache.to(d_type)
    sin_cache = sin_cache.to(d_type)

    return cos_cache, sin_cache
