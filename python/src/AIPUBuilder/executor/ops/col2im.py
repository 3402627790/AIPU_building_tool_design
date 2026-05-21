import torch
import numpy as np
from ..engine import register_gt, register_gen_input
from AIPUBuilder.core import OpType, Dtype
from .gt_util.quant_utils import clamp, linear_requantize


@register_gt(OpType.Col2Im)
def Col2Im(self, *args, **kwargs):
    inp = self.inputs[0].betensor.float()
    image_dims = self.inputs[1].betensor.int()
    kernel_dims = self.inputs[2].betensor.int()

    rank = image_dims.numel()
    if rank == 2:
        out = col2im2D(self, inp.clone(), image_dims, kernel_dims)
    else:
        out = col2imND(self, inp.clone(), image_dims, kernel_dims)

    self.outputs[0].betensor = out
    return [o.betensor for o in self.outputs]


def unfoldNd(node, input, kernel_size, kernel_size_numel):
    batch_size, in_channels = input.shape[0], input.shape[1]
    stride = (node.params['stride_y'], node.params['stride_x'])
    dilation = (node.params['dilation_y'], node.params['dilation_x'])
    padding = (node.params['pad_left'], node.params['pad_right'],
               node.params['pad_top'], node.params['pad_bottom'])
    repeat = [in_channels, 1] + [1 for _ in kernel_size]
    weight = torch.eye(kernel_size_numel, device=input.device, dtype=input.dtype).reshape(
        (kernel_size_numel, 1, *kernel_size)).repeat(*repeat)
    input = torch.nn.functional.pad(input, padding, mode='constant', value=-32767)
    unfold = torch.nn.functional.conv2d(
        input,
        weight,
        bias=None,
        stride=stride,
        padding=0,
        dilation=dilation,
        groups=in_channels,
    )
    return unfold.reshape(batch_size, in_channels * kernel_size_numel, -1)


def col2im2D(node, inp, image_dims, kernel_dims):
    batch_size = inp.shape[0]
    in_channels_kernel_size_numel = inp.shape[1]
    kernel_size_numel = torch.prod(kernel_dims)
    output_size_numel = torch.prod(image_dims)
    in_channels = in_channels_kernel_size_numel // kernel_size_numel

    idx = torch.arange(output_size_numel, dtype=torch.float32, device=inp.device).reshape(1, 1, *image_dims)
    idx = unfoldNd(node, idx, kernel_dims, kernel_size_numel)

    inp = inp.reshape(batch_size, in_channels, -1)
    idx = idx.reshape(1, 1, -1).long().expand(batch_size, in_channels, -1)

    mask = idx == -32767
    inp[mask] = 0
    idx[mask] = 0

    out = torch.zeros(
        batch_size, in_channels, output_size_numel, dtype=inp.dtype, device=inp.device)

    out.scatter_add_(2, idx, inp)

    out = out.reshape(batch_size, in_channels, *image_dims)
    if 'scale_value' in node.params and node.params['scale_value'] != None:
        do_scale = node.params['scale_value']
        do_shift = node.params['shift_value']
        out = linear_requantize(out, do_scale, do_shift, str(
            node.outputs[0].dtype), node.outputs[0].quantization.offset)
    return out


def col2imND(node, inp, image_dims, kernel_dims):
    from AIPUBuilder.Optimizer.utils import construct_torch_tensor as torch_tensor
    padding = [node.params['pad_z_begin'], node.params['pad_y_begin'], node.params['pad_x_begin'],
               node.params['pad_z_end'], node.params['pad_y_end'], node.params['pad_x_end']]
    dilations = [node.params['dilation_z'], node.params['dilation_y'], node.params['dilation_x']]
    strides = [node.params['stride_z'], node.params['stride_y'], node.params['stride_x']]
    rank = image_dims.numel()
    image_shape_size = torch.prod(image_dims)
    kernel_shape_size = torch.prod(kernel_dims)
    adjusted_kernel_shape = torch_tensor(dilations, device=inp.device) * (kernel_dims - 1) + 1

    N = inp.shape[0]
    C = int(inp.shape[1] / kernel_shape_size)
    col_stride = C * image_shape_size  # 3*14*15*16
    col_data_stride = inp.shape[1] * inp.shape[2]
    batched_image_shape_dims = torch.cat(
        (torch_tensor([N, C], device=inp.device), torch.zeros([rank], device=inp.device)))
    batched_image_shape_dims[2:] = image_dims
    adjusted_image_shape_dims = image_dims - adjusted_kernel_shape + 1
    batched_image_shape_dims = batched_image_shape_dims.long().tolist()

    inp_flatten = inp.flatten()
    image_data = torch.zeros(batched_image_shape_dims, device=inp.device).flatten()
    for b in range(N):
        data_im = inp_flatten[b * col_data_stride:]
        im_shape = image_dims
        output_shape = adjusted_image_shape_dims
        channels_col = kernel_shape_size * C
        kernel_shape = adjusted_kernel_shape
        data_col = image_data[b * col_stride:]
        kernel_size = torch.prod(kernel_shape)
        d_offset = [0] * rank
        d_iter = [0] * rank
        for c_col in range(channels_col):
            offset = c_col
            for di in range(rank - 1, -1, -1):
                if di < (rank - 1):
                    offset = offset // kernel_shape[di + 1]
                d_offset[di] = int(offset) % kernel_shape[di]
            while True:
                index_col = c_col
                index_im = c_col // kernel_size
                is_padding = False
                for d_i in range(rank):
                    d = d_iter[d_i]
                    d_im = d * strides[d_i] - padding[d_i] + d_offset[d_i] * dilations[d_i]
                    is_padding = is_padding or (not (d_im >= 0 and d_im < im_shape[d_i]))
                    index_col *= output_shape[d_i]
                    index_col += d
                    index_im *= im_shape[d_i]
                    index_im += d_im
                if not is_padding:
                    if index_col < data_im.shape[0] and index_im < data_col.shape[0]:
                        data_col[index_im] += data_im[index_col]

                def NextPosition(shape, d_iter):
                    has_next_output = False
                    for d_x in range(rank - 1, -1, -1):
                        d_max = shape[d_x]
                        if d_iter[d_x] == (d_max - 1):
                            d_iter[d_x] = 0
                        else:
                            d_iter[d_x] += 1
                            has_next_output = True
                            break
                    return has_next_output, d_iter

                has_next_output, d_iter = NextPosition(output_shape, d_iter)
                if not has_next_output:
                    break

    output = torch.reshape(image_data, batched_image_shape_dims)
    if 'scale_value' in node.params and node.params['scale_value'] != None:
        do_scale = node.params['scale_value']
        do_shift = node.params['shift_value']
        output = linear_requantize(output, do_scale, do_shift, str(
            node.outputs[0].dtype), node.outputs[0].quantization.offset)
    return output


@register_gen_input(OpType.Col2Im)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_shape2 = list(self.inputs[2].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)
    in_type2 = str(self.inputs[2].dtype)

    image_size = list(self.outputs[0].shape)[-2:]
    kernel_y = self.params.get('kernel_y', 1)
    kernel_x = self.params.get('kernel_x', 1)
    block_size = [kernel_y, kernel_x]

    if self.inputs[0].dtype.is_float:
        input0 = (np.random.uniform(0, 10000, size=in_shape0)).astype(in_type0)
    else:
        input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)
    input1 = np.asarray(image_size, dtype=in_type1)
    input2 = np.asarray(block_size, dtype=in_type2)

    input_data = [input0, input1, input2]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    self.inputs[2].betensor = input_data[2]
    return input_data
