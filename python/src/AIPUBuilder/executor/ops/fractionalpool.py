from AIPUBuilder.core import OpType

from .gt_util import *
from ..engine import register_gt


@register_gt(OpType.FractionalPool)
def FractionalPool(self, *args, **kwargs):
    method = self.params['method'].upper()
    seed = int(self.params['seed'])
    pseudo = self.params['pseudo']
    overlap = self.params['overlap']

    inp = self.inputs[0].betensor
    batch = inp.shape[0]
    in_height, in_width, in_channel = list(self.inputs[0].shape)[1:4]
    out_height, out_width = list(self.outputs[0].shape)[1:3]

    out = torch.zeros([batch, out_height, out_width, in_channel], device=inp.device)

    height_cum_seq = GeneratePoolingSequence(in_height, out_height, pseudo, seed).to(inp.device)
    width_cum_seq = GeneratePoolingSequence(in_width, out_width, pseudo, seed).to(inp.device)

    StartH = height_cum_seq[:out_height]
    EndH = height_cum_seq[1: out_height+1] + (1 if overlap else 0)
    StartW = width_cum_seq[:out_width]
    EndW = width_cum_seq[1: out_width+1] + (1 if overlap else 0)
    EndH = torch.clamp(EndH, 0, in_height)
    EndW = torch.clamp(EndW, 0, in_width)

    for h_idx in range(out_height):
        for w_idx in range(out_width):
            inp_windows = inp[:, StartH[h_idx]:EndH[h_idx], StartW[w_idx]:EndW[w_idx], :]
            inp_windows = inp_windows.float() if self.inputs[0].dtype.is_float else inp_windows
            if method == "MAX":
                out[:, h_idx, w_idx, :] = torch.amax(inp_windows, dim=[1, 2])
            else:
                kernel_y = EndH[h_idx] - StartH[h_idx]
                kernel_x = EndW[w_idx] - StartW[w_idx]
                psum = torch.sum(inp_windows, dim=[1, 2])
                area = (kernel_x * kernel_y)
                if self.inputs[0].dtype.is_float:
                    out[:, h_idx, w_idx, :] = psum / area
                else:
                    do_scale, _, do_shift, _ = get_scale_approximation_params(
                        1. / area, mult_bits=8, force_shift_positive=False)
                    out[:, h_idx, w_idx, :] = linear_requantize(
                        psum, do_scale, do_shift, str(self.outputs[0].dtype)).int()

    self.outputs[0].betensor = out
    self.outputs[1].betensor = height_cum_seq
    self.outputs[2].betensor = width_cum_seq
    return [o.betensor for o in self.outputs]


def generate_pseudo_random_seq(input_length, output_length, seed):
    cum_seq = torch.zeros([output_length + 1])
    diff = torch.zeros([output_length])

    alpha = float(input_length) / output_length
    k = input_length // output_length
    u_max1 = (k + 2) / alpha - 1
    u_max2 = (input_length + 1 - k) / alpha - (output_length - 1)
    max_u = min(u_max1, u_max2)

    if seed != 0:
        mt = mersenne_twister_engine(mt19937_params)
        mt.set_seed(seed)
        rand_value = uniform_real_distribution(0, 1, mt).getrand()
    else:
        rand_value = torch.rand([1]).item()
    u = rand_value * max_u

    cum_seq[0] = 1
    cum_seq[output_length] = input_length + 1
    update_value = [alpha * (i + u) for i in range(1, output_length)]
    cum_seq[1:output_length] = torch.ceil(torch.tensor(update_value))

    for i in range(output_length):
        diff[i] = cum_seq[i + 1] - cum_seq[i]

    return diff


def generate_random_seq(input_length, output_length, seed):
    k = input_length // output_length
    num_random_spot = input_length % output_length
    diff = torch.ones([output_length]) * k
    diff[:num_random_spot] += 1
    if seed == 0:
        diff = shuffle(diff, None, 1, diff.shape[0])
    else:
        mt = mersenne_twister_engine(mt19937_params)
        mt.set_seed(seed)
        diff = shuffle(diff, mt, 0, diff.shape[0])
    return diff


def GeneratePoolingSequence(in_length, out_length, pseudo, seed):
    if pseudo:
        diff = generate_pseudo_random_seq(in_length, out_length, seed)
    else:
        diff = generate_random_seq(in_length, out_length, seed)
    cum_seq = torch.zeros([out_length + 1])
    for index in range(1, cum_seq.shape[0]):
        cum_seq[index] = cum_seq[index - 1] + diff[index - 1]
    return cum_seq.int()
