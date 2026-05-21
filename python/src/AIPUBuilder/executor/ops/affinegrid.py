import torch
import numpy as np
from AIPUBuilder.core import OpType, Dtype
from AIPUBuilder.logger import WARN
from ..engine import register_gt, register_gen_input, register_gen_weight
from .gt_util.quant_utils import clamp, linear_requantize
from .gt_util.dtype_utils import bits2range


@register_gt(OpType.AffineGrid)
def AffineGrid(self, *args, **kwargs):
    theta = self.inputs[0].betensor
    align_corners = self.params['align_corners']  # todo, not used?
    size = list(self.outputs[0].shape)
    is_5dims = True if len(size) == 5 else False

    if is_5dims:
        out = affine_grid_generator_3d(self, theta, size)
    else:
        out = affine_grid_generator_2d(self, theta, size)

    self.outputs[0].betensor = out
    return [o.betensor for o in self.outputs]


def generate_base_grid_2d(node, H, W):
    row = node.constants['w_lut'].betensor.double()
    col = node.constants['h_lut'].betensor.double()
    row_vec = row.unsqueeze(0)
    col_vec = col.unsqueeze(1)
    row_vec = row_vec.repeat(H, 1)
    col_vec = col_vec.repeat(1, W)
    base_grid = torch.stack((row_vec, col_vec), dim=2).reshape((H * W, 2)).permute(1, 0)
    return base_grid


def generate_base_grid_3d(node, D, H, W):
    row = node.constants['w_lut'].betensor.double()
    col = node.constants['h_lut'].betensor.double()
    slice = node.constants['d_lut'].betensor.double()
    row_vec = row.unsqueeze(0).unsqueeze(1)
    col_vec = col.unsqueeze(0).unsqueeze(2)
    slice_vec = slice.unsqueeze(1).unsqueeze(2)
    row_vec = row_vec.repeat(D, H, 1)
    col_vec = col_vec.repeat(D, 1, W)
    slice_vec = slice_vec.repeat(1, H, W)
    base_grid = torch.stack((row_vec, col_vec, slice_vec), dim=3).reshape((D * H * W, 3)).permute(1, 0)
    return base_grid


def affine_grid_generator_2d(node, theta, size):
    N, H, W, C, = size[0], size[1], size[2], size[3]
    output = torch.zeros([N, H, W, 2], device=theta.device)
    shift = node.params.get('shift_value', 0)
    scale = node.params.get('scale_value', 1)
    multiplier_bits = node.params.get('multiplier_bits', 0)
    qmin, qmax = bits2range(32, True)

    base_grid = generate_base_grid_2d(node, H, W)
    for b in range(N):
        theta_flatten = theta[b].flatten()
        theta_R = torch.tensor([
            [theta_flatten[0], theta_flatten[1]],
            [theta_flatten[3], theta_flatten[4]]
        ], device=theta.device).double()
        theta_T = torch.tensor([
            [theta_flatten[2]],
            [theta_flatten[5]]
        ], device=theta.device).double()
        batch_out = theta_R @ base_grid + theta_T * (2**(multiplier_bits))
        if batch_out.max() > 2 ** 31 - 1 or batch_out.min() < -2 ** 31:
            WARN('[Executor] affinegrid overflow !')
        batch_out = torch.clamp(batch_out, qmin, qmax)
        batch_out = linear_requantize(batch_out, scale, shift, str(
            node.outputs[0].dtype), node.outputs[0].quantization.offset)
        batch_out = batch_out.reshape(2, H, W).permute(1, 2, 0)
        output[b, ...] = batch_out
    return output


def affine_grid_generator_3d(node, theta, size):
    N, D, H, W, C, = size[0], size[1], size[2], size[3], size[4]
    output = torch.zeros([N, D, H, W, 3], device=theta.device)
    shift = node.params.get('shift_value', 0)
    scale = node.params.get('scale_value', 1)
    multiplier_bits = node.params.get('multiplier_bits', 0)
    qmin, qmax = bits2range(32, True)

    base_grid = generate_base_grid_3d(node, D, H, W)
    for b in range(N):
        theta_flatten = theta[b].flatten()
        theta_R = torch.tensor([
            [theta_flatten[0], theta_flatten[1], theta_flatten[2]],
            [theta_flatten[4], theta_flatten[5], theta_flatten[6]],
            [theta_flatten[8], theta_flatten[9], theta_flatten[10]]
        ], device=theta.device).double()
        theta_T = torch.tensor([
            [theta_flatten[3]],
            [theta_flatten[7]],
            [theta_flatten[11]]
        ], device=theta.device).double()
        batch_out = theta_R @ base_grid + theta_T * (2**(multiplier_bits))
        if batch_out.max() > 2 ** 31 - 1 or batch_out.min() < -2 ** 31:
            WARN('[Executor] affinegrid overflow !')
        batch_out = torch.clamp(batch_out, qmin, qmax)
        # avoid 32bit overflow
        batch_out = linear_requantize(batch_out, scale, shift, str(
            node.outputs[0].dtype), node.outputs[0].quantization.offset)
        batch_out = batch_out.reshape(3, D, H, W).permute(1, 2, 3, 0)
        output[b, ...] = batch_out
    return output


@register_gen_input(OpType.AffineGrid)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)

    input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)
    input1 = np.asarray(self.outputs[0].shape, dtype=in_type1)
    input_data = [input0, input1]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    return input_data


@register_gen_weight(OpType.AffineGrid)
def gen_random_weight(self):
    self.constants['h_lut'].betensor = np.random.randint(-1000, 1000, self.constants['h_lut'].shape,
                                                         str(self.constants['h_lut'].dtype))
    self.constants['w_lut'].betensor = np.random.randint(-1000, 1000, self.constants['w_lut'].shape,
                                                         str(self.constants['w_lut'].dtype))
    if 'd_lut' in self.constants:
        self.constants['d_lut'].betensor = np.random.randint(-100, 100, self.constants['d_lut'].shape,
                                                             str(self.constants['d_lut'].dtype))
