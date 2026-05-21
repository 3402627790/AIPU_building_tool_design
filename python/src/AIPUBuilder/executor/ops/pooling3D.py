from ..engine import register_gt, _OP_METRICS, register_op_metric
from .gt_util import *
from AIPUBuilder.core import OpType
import torch
import numpy as np
import math
from .gt_util.aiff import _aiff_compression_bias, _aiff_merge_shifts, clip_to_bit
from .gt_util.layout_transfer import layout_convert
from .gt_util.quant_utils import clamp


@register_gt(OpType.Pooling3D)
def pooling3D(self, *args, **kwargs):
    self.attrs['is_aiff'] = kwargs.get('is_aiff', False)  # for pooling3d's metric
    if self.inputs[0].dtype.is_float:
        output = pooling3d_fp(self)
    else:
        inp = self.inputs[0].irtensor.long()

        kernel = [self.params['kernel_z'], self.params['kernel_y'], self.params['kernel_x']]
        stride = [self.params['stride_z'], self.params['stride_y'], self.params['stride_x']]
        dilation = [self.params['dilation_z'], self.params['dilation_y'], self.params['dilation_x']]

        method = self.params['method']
        ceil_mode = self.params['ceil_mode']
        input_type = self.inputs[0].dtype.np
        inp = clamp(inp, self.inputs[0].dtype)

        # is_aiff is true defaultly in gt
        is_aiff = kwargs['is_aiff'] if 'is_aiff' in kwargs.keys() else True

        lut = gen_div_lut(self)

        if method == 'MAX':
            padding = [self.params['pad_x_begin'], self.params['pad_x_end'],
                       self.params['pad_y_begin'], self.params['pad_y_end'],
                       self.params['pad_z_begin'], self.params['pad_z_end']]
            inp = layout_convert(inp, 'NDHWC', 'NCDHW')
            pad_val = np.iinfo(input_type).min
            inp = torch.nn.functional.pad(inp, padding, value=pad_val)
            output = torch.max_pool3d(inp, kernel_size=kernel, stride=stride,
                                      padding=0, ceil_mode=ceil_mode)
            output = layout_convert(output, 'NCDHW', 'NDHWC').cpu().numpy()

        elif method == 'MIN':
            padding = [self.params['pad_x_begin'], self.params['pad_x_end'],
                       self.params['pad_y_begin'], self.params['pad_y_end'],
                       self.params['pad_z_begin'], self.params['pad_z_end']]
            inp = layout_convert(inp, 'NDHWC', 'NCDHW')
            pad_val = np.iinfo(input_type).max
            inp = torch.nn.functional.pad(inp, padding, value=pad_val)
            output = -torch.max_pool3d(-inp, kernel_size=kernel, stride=stride,
                                       padding=0, ceil_mode=ceil_mode)
            output = layout_convert(output, 'NCDHW', 'NDHWC').cpu().numpy()

        elif method == 'AVG':
            padding = [self.params['pad_z_begin'], self.params['pad_z_end'],
                       self.params['pad_y_begin'], self.params['pad_y_end'],
                       self.params['pad_x_begin'], self.params['pad_x_end']]
            if "pad_z_begin" in self.attrs:
                padding[0] = self.attrs['pad_z_begin']
                padding[1] = self.attrs['ori_pad_z_end']
                begin = padding[0]
                end = inp.shape[1] - self.attrs["pad_z_end"]
                inp = inp[:, begin:end, :, :, :]
            _, i_d, i_h, i_w, _ = inp.shape
            ceil_mode_pad_tail = 0
            ceil_mode_pad_bottom = 0
            ceil_mode_pad_right = 0
            if ceil_mode:
                o_d = math.ceil((i_d - kernel[0] + padding[0] + padding[1]) / stride[0]) + 1
                o_h = math.ceil((i_h - kernel[1] + padding[2] + padding[3]) / stride[1]) + 1
                o_w = math.ceil((i_w - kernel[2] + padding[4] + padding[5]) / stride[2]) + 1
                ceil_mode_pad_tail = (o_d - 1) * stride[0] + kernel[0] - (padding[0] + padding[1]) - i_d
                ceil_mode_pad_bottom = (o_h - 1) * stride[1] + kernel[1] - (padding[2] + padding[3]) - i_h
                ceil_mode_pad_right = (o_w - 1) * stride[2] + kernel[2] - (padding[4] + padding[5]) - i_w
            # tpc version
            if not is_aiff and np.prod(kernel) <= 17 * 17 * 17:
                print('tpc avgpool3d!')
                output = avg_pool_3d(inp.cpu().numpy(), kernel,
                                     stride, dilation, padding,
                                     [ceil_mode_pad_tail, ceil_mode_pad_bottom, ceil_mode_pad_right],
                                     lut, layer=self)
            # aiff version
            elif np.prod(kernel) <= 17 * 17 * 17:
                print('aiff avgpool3d!')
                output = avg_pool_3d_aiff(inp.cpu().numpy(), kernel,
                                          stride, dilation, padding,
                                          [ceil_mode_pad_tail, ceil_mode_pad_bottom, ceil_mode_pad_right],
                                          lut, layer=self)
            else:
                raise Exception("avg pooling only support kernel_z * kernel_y * kernel_x <= 17 * 17 * 17")
        else:
            raise Exception("No supported pooling type:", method)

        output = clip_to_bit(output, 32, True)
        output = np.round(output).astype(np.int32)
        output = clamp(output, self.outputs[0].dtype)
    self.outputs[0].irtensor = output
    return output


def avg_pool_3d(inputs, kernel, stride, dilation, pad_value, ceil_mode_pad, DivLUT, layer=None):
    # hardware does not support division, so use lut method to realize avg_pool_3d function

    inputs += layer.inputs[0].quantization.offset if layer.params['count_include_pad'] else 0  # align to opt

    in_batch, in_depth, in_height, in_width, in_channel = np.shape(inputs)
    kernel_z, kernel_y, kernel_x = kernel
    stride_z, stride_y, stride_x = stride
    dilation_z, dilation_y, dilation_x = dilation

    pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right = pad_value
    ceil_mode_pad_tail, ceil_mode_pad_bottom, ceil_mode_pad_right = ceil_mode_pad

    if layer.params.get('ceil_mode', default=False):
        out_depth = math.ceil(
            (in_depth + pad_head + pad_tail - (1 + (kernel_z - 1) * dilation_z)) / stride_z) + 1
        out_height = math.ceil(
            (in_height + pad_top + pad_bottom - (1 + (kernel_y - 1) * dilation_y)) / stride_y) + 1
        out_width = math.ceil(
            (in_width + pad_left + pad_right - (1 + (kernel_x - 1) * dilation_x)) / stride_x) + 1
    else:
        out_depth = math.floor(
            (in_depth + pad_head + pad_tail - (1 + (kernel_z - 1) * dilation_z)) / stride_z) + 1
        out_height = math.floor(
            (in_height + pad_top + pad_bottom - (1 + (kernel_y - 1) * dilation_y)) / stride_y) + 1
        out_width = math.floor(
            (in_width + pad_left + pad_right - (1 + (kernel_x - 1) * dilation_x)) / stride_x) + 1

    output = np.zeros((in_batch, out_depth, out_height, out_width, in_channel))
    output_dump = np.zeros((in_batch, out_depth, out_height, out_width, in_channel))

    for pd in range(out_depth):
        d_start = pd * stride_z - pad_head
        d_end = min(d_start + kernel_z, in_depth)
        d_start = max(d_start, 0)
        # height
        for ph in range(out_height):
            h_start = ph * stride_y - pad_top
            h_end = min(h_start + kernel_y, in_height)
            h_start = max(h_start, 0)
            # width
            for pw in range(out_width):
                w_start = pw * stride_x - pad_left
                w_end = min(w_start + kernel_x, in_width)
                w_start = max(w_start, 0)

                output_dump[:, pd, ph, pw, :] = np.sum(
                    inputs[:, d_start: d_end, h_start: h_end, w_start: w_end, :], axis=(1, 2, 3))

                if layer.params['count_include_pad']:
                    real_kernel_y = kernel_y - ceil_mode_pad_bottom if (
                        layer.params['ceil_mode'] and ph == out_height - 1) else kernel_y
                    real_kernel_x = kernel_x - ceil_mode_pad_right if (
                        layer.params['ceil_mode'] and pw == out_width - 1) else kernel_x
                    box_size_xy = real_kernel_y * real_kernel_x
                else:
                    box_size_xy = (h_end - h_start) * (w_end - w_start)

                if layer.params['count_include_pad']:
                    real_kernel_z = kernel_z - ceil_mode_pad_tail if (
                        layer.params['ceil_mode'] and pd == out_depth - 1) else kernel_z
                    box_size_d = real_kernel_z
                else:
                    box_size_d = (d_end - d_start)

                if layer.params['count_include_pad']:
                    box_without_pad = box_size_xy * box_size_d
                    box_with_pad = (d_end - d_start) * (h_end - h_start) * (w_end - w_start)
                    total_pads = box_without_pad - box_with_pad
                else:
                    total_pads = 0

                if layer.inputs[0].dtype.is_float:
                    output_dump[:, pd, ph, pw, :] = np.sum(
                        inputs.astype('float32')[:, d_start: d_end, h_start: h_end, w_start: w_end, :], axis=(1, 2, 3))
                    output[:, pd, ph, pw, :] = output_dump[:, pd, ph, pw, :] / (box_size_xy * box_size_d)
                else:
                    # output_dump[:, pd, ph, pw, :] -= total_pads * layer.inputs[0].quantization.offset  # ori gt/lib

                    # for kh*kw
                    scale = DivLUT[2 * box_size_xy]
                    shift = DivLUT[2 * box_size_xy + 1]
                    output[:, pd, ph, pw, :] = np.round(output_dump[:, pd, ph, pw, :] *
                                                        scale * (0.5 ** shift)).astype(np.int32)

                    # for kd
                    scale = DivLUT[2 * box_size_d]
                    shift = DivLUT[2 * box_size_d + 1]
                    output[:, pd, ph, pw, :] = np.round(output[:, pd, ph, pw, :] *
                                                        scale * (0.5 ** shift)).astype(np.int32)

    output -= layer.outputs[0].quantization.offset if layer.params['count_include_pad'] else 0  # align to opt

    return output


def avg_pool_3d_aiff(inputs, kernel, stride, dilation, pad_value, ceil_mode_pad, DivLUT, layer=None):
    '''
    avg_pool_3d aiff version, calc hw -> scale/shift(unit on hw) -> calc d -> scale/shift(d)
    :param inputs:
    :param kernel:
    :param stride:
    :param pad_value:
    :param DivLUT:
    :return:
    '''

    in_batch, in_depth, in_height, in_width, in_channel = np.shape(inputs)
    kernel_z, kernel_y, kernel_x = kernel
    stride_z, stride_y, stride_x = stride
    dilation_z, dilation_y, dilation_x = dilation

    pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right = pad_value
    ceil_mode_pad_tail, ceil_mode_pad_bottom, ceil_mode_pad_right = ceil_mode_pad

    if layer.params.get('ceil_mode', default=False):
        out_depth = math.ceil(
            (in_depth + pad_head + pad_tail - (1 + (kernel_z - 1) * dilation_z)) / stride_z) + 1
        out_height = math.ceil(
            (in_height + pad_top + pad_bottom - (1 + (kernel_y - 1) * dilation_y)) / stride_y) + 1
        out_width = math.ceil(
            (in_width + pad_left + pad_right - (1 + (kernel_x - 1) * dilation_x)) / stride_x) + 1
    else:
        out_depth = math.floor(
            (in_depth + pad_head + pad_tail - (1 + (kernel_z - 1) * dilation_z)) / stride_z) + 1
        out_height = math.floor(
            (in_height + pad_top + pad_bottom - (1 + (kernel_y - 1) * dilation_y)) / stride_y) + 1
        out_width = math.floor(
            (in_width + pad_left + pad_right - (1 + (kernel_x - 1) * dilation_x)) / stride_x) + 1

    output = np.zeros((in_batch, out_depth, out_height, out_width, in_channel))
    output_dump = np.zeros((in_batch, out_depth, out_height, out_width, in_channel))
    output_hw = np.zeros((in_batch, in_depth, out_height, out_width, in_channel))
    output_dump_hw = np.zeros((in_batch, in_depth, out_height, out_width, in_channel))

    scale_list = []
    shift_list = []
    box_sizes = []

    # height
    for ph in range(out_height):
        h_start = ph * stride_y - pad_top
        h_end = min(h_start + kernel_y, in_height)
        h_start = max(h_start, 0)
        # width
        for pw in range(out_width):
            w_start = pw * stride_x - pad_left
            w_end = min(w_start + kernel_x, in_width)
            w_start = max(w_start, 0)

            # for kh*kw
            if layer.params['count_include_pad']:
                real_kernel_y = kernel_y - ceil_mode_pad_bottom if (
                    layer.params['ceil_mode'] and ph == out_height - 1) else kernel_y
                real_kernel_x = kernel_x - ceil_mode_pad_right if (
                    layer.params['ceil_mode'] and pw == out_width - 1) else kernel_x
                box_size = real_kernel_y * real_kernel_x
            else:
                box_size = (h_end - h_start) * (w_end - w_start)
            scale = DivLUT[2 * box_size]
            shift = DivLUT[2 * box_size + 1]
            scale_list.append(scale)
            shift_list.append(shift)
            box_sizes.append(box_size)
    new_scale, new_shift = _aiff_merge_shifts(torch.Tensor(scale_list).int(), torch.Tensor(shift_list).int())
    new_scale = new_scale.cpu().numpy()
    new_shift = new_shift.cpu().numpy()

    for pd in range(out_depth):
        d_start = pd * stride_z - pad_head
        d_end = min(d_start + kernel_z, in_depth)
        d_start = max(d_start, 0)
        # height
        for ph in range(out_height):
            h_start = ph * stride_y - pad_top
            h_end = min(h_start + kernel_y, in_height)
            h_start = max(h_start, 0)
            # width
            for pw in range(out_width):
                w_start = pw * stride_x - pad_left
                w_end = min(w_start + kernel_x, in_width)
                w_start = max(w_start, 0)

                scale_index = ph * out_width + pw
                scale = new_scale[scale_index]
                shift = new_shift
                box_size = box_sizes[scale_index]
                sumed_num = (w_end - w_start) * (h_end - h_start)
                pad_num = box_size - sumed_num
                pad_v = 0 - layer.inputs[0].quantization.offset
                # sum h/w at first
                output_dump_hw[:, d_start: d_end, ph, pw, :] = np.sum(
                    inputs[:, d_start: d_end, h_start: h_end, w_start: w_end, :], axis=(2, 3)) \
                    + (pad_v * pad_num)  # h/w

                # for kh*kw, clip
                output_hw[:, d_start: d_end, ph, pw, :] = clamp(np.round(
                    output_dump_hw[:, d_start: d_end, ph, pw, :] * scale * (0.5 ** shift)),
                    dtype=layer.outputs[0].dtype)

                # for kd
                if layer.params['count_include_pad']:
                    real_kernel_z = kernel_z - ceil_mode_pad_tail if (
                        layer.params['ceil_mode'] and pd == out_depth - 1) else kernel_z
                    box_size = real_kernel_z
                else:
                    box_size = (d_end - d_start)
                # then sum d
                sumed_num = (d_end - d_start)
                pad_num = box_size - sumed_num
                output_dump[:, pd, ph, pw, :] = np.sum(output_hw[:, d_start: d_end, ph, pw, :], axis=(1)) \
                    + (pad_v * pad_num)

                scale = DivLUT[2 * box_size]
                shift = DivLUT[2 * box_size + 1]
                output[:, pd, ph, pw, :] = np.round(
                    output_dump[:, pd, ph, pw, :] * scale * (0.5 ** shift)).astype(
                    np.int32)
    return output


def gen_div_lut(layer):
    # use internal lut rather than external lut from 202107
    div_lut = np.array([1, 0, 1, 0], dtype=np.uint8)
    kernel_z = layer.params['kernel_z']
    kernel_y = layer.params['kernel_y']
    kernel_x = layer.params['kernel_x']
    kernel_size = max(kernel_z, kernel_y * kernel_x) + 1
    for i in range(2, kernel_size):
        scale = 1 / i
        do_scale = 0
        do_shift = 0
        for j in reversed(range(0, 9)):
            do_scale = np.round(2 ** (np.log2(scale) - np.round(np.log2(scale)) + j)).astype(np.uint32)
            do_shift = abs(np.round(np.log2(scale)) - j).astype(np.uint32)
            if (do_scale < 256 and do_scale % 2 != 0):
                break
        div_lut = np.concatenate((div_lut, [do_scale, do_shift])).astype(np.uint8)

    return div_lut


def pooling3d_fp(self):
    inp = self.inputs[0].irtensor.float()
    count_include_pad = self.params.get('count_include_pad', default=True)
    kernel = [self.params['kernel_z'], self.params['kernel_y'], self.params['kernel_x']]
    stride = [self.params['stride_z'], self.params['stride_y'], self.params['stride_x']]
    dilation = [self.params['dilation_z'], self.params['dilation_y'], self.params['dilation_x']]
    padding = [self.params['pad_x_begin'], self.params['pad_x_end'],
               self.params['pad_y_begin'], self.params['pad_y_end'],
               self.params['pad_z_begin'], self.params['pad_z_end']]
    ceil_mode = self.params.get('ceil_mode',  default=False)
    method = self.params['method']
    input_type = str(self.inputs[0].dtype)

    bf16_max_value = 3.38953e38

    if method == 'MAX':
        inp = layout_convert(inp, 'NDHWC', 'NCDHW')
        pad_val = -bf16_max_value if input_type == 'bfloat16' else np.finfo(input_type).min
        inp = torch.nn.functional.pad(inp, padding, value=pad_val)
        output = torch.max_pool3d(inp, kernel_size=kernel, stride=stride,
                                  padding=0, ceil_mode=ceil_mode)
        output = layout_convert(output, 'NCDHW', 'NDHWC')

    elif method == 'MIN':
        inp = layout_convert(inp, 'NDHWC', 'NCDHW')
        pad_val = bf16_max_value if input_type == 'bfloat16' else np.finfo(input_type).max
        inp = torch.nn.functional.pad(inp, padding, value=pad_val)
        output = -torch.max_pool3d(-inp, kernel_size=kernel, stride=stride,
                                   padding=0, ceil_mode=ceil_mode)
        output = layout_convert(output, 'NCDHW', 'NDHWC')

    elif method == 'AVG':
        padding = [self.params['pad_z_begin'], self.params['pad_z_end'],
                   self.params['pad_y_begin'], self.params['pad_y_end'],
                   self.params['pad_x_begin'], self.params['pad_x_end']]
        if "pad_z_begin" in self.attrs:
            padding[0] = self.attrs['pad_z_begin']
            padding[1] = self.attrs['pad_z_end']
        _, i_d, i_h, i_w, _ = inp.shape
        ceil_mode_pad_tail = 0
        ceil_mode_pad_bottom = 0
        ceil_mode_pad_right = 0
        if ceil_mode:
            o_d = math.ceil((i_d - kernel[0] + padding[0] + padding[1]) / stride[0]) + 1
            o_h = math.ceil((i_h - kernel[1] + padding[2] + padding[3]) / stride[1]) + 1
            o_w = math.ceil((i_w - kernel[2] + padding[4] + padding[5]) / stride[2]) + 1
            ceil_mode_pad_tail = (o_d - 1) * stride[0] + kernel[0] - (padding[0] + padding[1]) - i_d
            ceil_mode_pad_bottom = (o_h - 1) * stride[1] + kernel[1] - (padding[2] + padding[3]) - i_h
            ceil_mode_pad_right = (o_w - 1) * stride[2] + kernel[2] - (padding[4] + padding[5]) - i_w

        output = avg_pool_3d(inp.cpu().numpy(), kernel,
                             stride, dilation, padding,
                             [ceil_mode_pad_tail, ceil_mode_pad_bottom, ceil_mode_pad_right],
                             DivLUT=None, layer=self)
    return output


@register_op_metric(OpType.Pooling3D)
@check_fp_data
def metric(self, outputs, index=None):
    # note: aiff's metric maybe called in layoutconvertor/reshape
    def aiff_call(r, u, c, t):
        if "float" in t:
            return np.all(u <= 1) or np.all(c >= 0.9999)
        return np.all(u <= 0)

    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]

    layer_dtypes = [str(o.dtype) for o in self.outputs] if index is None else [str(self.outputs[index].dtype)]

    if self.attrs.get('is_aiff', False):
        rel_diffs = relative_diff(outputs, groundtruths)
        ulp_diffs = ulp_diff(outputs, groundtruths)
        cos_dists = cos_distance(outputs, groundtruths)
        return all(aiff_call(r, u, c, t) for r, u, c, t in zip(rel_diffs, ulp_diffs, cos_dists, layer_dtypes))
    return _OP_METRICS['_all'](self, outputs, index)
