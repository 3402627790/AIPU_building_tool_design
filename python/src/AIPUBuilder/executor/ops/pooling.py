import math
import torch
import numpy as np
from ..engine import register_gt, register_gen_weight, _OP_METRICS, register_op_metric
from AIPUBuilder.core import OpType
from AIPUBuilder.logger import DEBUG

from .gt_util.aiff import _aiff_compression_bias, _aiff_merge_shifts
from .gt_util.layout_transfer import layout_convert, transfer_shape
from .gt_util.aiff import _aiif_ahead_shift_biases, get_ops_options
from .gt_util.aiff import _aiff_inner_activation, _aiff_final_output
from .gt_util.aiff import clip_to_bit
from .gt_util import clamp
from .gt_util.metric import relative_diff, ulp_diff, cos_distance, check_fp_data
from .gt_util.dtype_utils import torch_to_numpy


@register_gt(OpType.Pooling)
@register_gt(OpType.GlobalPool)
def pooling(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float:
        return fp_pool(self)
    is_aiff = False
    if 'is_aiff' in kwargs.keys():
        is_aiff = kwargs['is_aiff']
    is_16b_scale = True if is_aiff and get_ops_options("AIPU_AVG_POOL_16BIT_SCALE", '1') == '1' else False
    o_type = str(self.outputs[0].dtype)

    inp = self.inputs[0].irtensor.long()
    # avgpool1d
    if inp.dim() == 3:
        inp = inp[:, None, :, :]

    count_include_pad = self.params.get('count_include_pad', default=True)
    kernel_size = [self.params.get('kernel_y', default=1), self.params.get('kernel_x', default=1)]
    stride = [self.params.get("stride_y",  default=1), self.params.get("stride_x",  default=1)]
    dilation = [self.params.get("dilation_y",  default=1), self.params.get("dilation_x",  default=1)]
    padding = [self.params.get('pad_top',  default=0),
               self.params.get('pad_bottom',  default=0),
               self.params.get('pad_left', 0),
               self.params.get('pad_right', 0)]
    kernel_size[0] = kernel_size[0] if isinstance(kernel_size[0], int) else 1
    stride[0] = stride[0] if isinstance(stride[0], int) else 1
    dilation[0] = dilation[0] if isinstance(dilation[0], int) else 1
    padding[0] = padding[0] if isinstance(padding[0], int) else 0
    padding[1] = padding[1] if isinstance(padding[1], int) else 0
    if self.type == OpType.GlobalPool:
        kernel_size = [inp.shape[1], inp.shape[2]]

    ceil_mode = self.params.get('ceil_mode',  default=False)
    inputs_type = str(self.inputs[0].dtype)
    if self.params['method'] == 'MAX':
        inp = layout_convert(inp, 'NHWC', 'NCHW')
        pad_val = np.iinfo(inputs_type).min
        padding_torch = padding[-2:] + padding[:-2]
        inp = torch.nn.functional.pad(inp, padding_torch, value=pad_val)
        y = torch.nn.functional.max_pool2d(inp.float(),
                                           kernel_size=kernel_size,
                                           stride=stride, padding=0,
                                           dilation=dilation,
                                           ceil_mode=ceil_mode)

        # conner case under ceil_mode=True CTL-6524/CB-6234
        out_shape = self.attrs['_output_shapes'][0] if '_output_shapes' in self.attrs else self.outputs[0].shape
        out_shape = transfer_shape(out_shape, self.outputs[0].shape.layout.name, 'NHWC') \
            if self.outputs[0].shape.layout.name not in ['Flat', 'NHWC'] else out_shape
        out_shape = [out_shape[0]] + [1] + list(out_shape)[1:] if len(out_shape) == 3 else out_shape
        _, out_h, out_w, out_c = out_shape
        current_outh, current_outw = y.shape[2: 4]
        if current_outh < out_h or current_outw < out_w:
            DEBUG('Pooling: conner case under ceil_mode=True.')
            auto_pad_h = out_h - current_outh
            auto_pad_w = out_w - current_outw
            padding = (0, auto_pad_w, 0, auto_pad_h)  # left, right, top, bottom
            y = torch.nn.functional.pad(y, padding, value=pad_val)

        y = clip_to_bit(y, 32, True, to_type=str(self.outputs[0].dtype))
        y = layout_convert(y, 'NCHW', 'NHWC')
        y = clip_to_bit(y, 32, True, o_type)

        if self.outputs[0].betensor.dim() == 3:
            y = torch.squeeze(y, dim=1)
        self.outputs[0].irtensor = y
        return y

    elif self.params['method'] == 'MIN':
        inp = layout_convert(inp, 'NHWC', 'NCHW')
        pad_val = np.iinfo(inputs_type).max
        padding_torch = padding[-2:] + padding[:-2]
        inp = torch.nn.functional.pad(inp, padding_torch, value=pad_val)
        y = -torch.nn.functional.max_pool2d(-inp.float(),
                                            kernel_size=kernel_size,
                                            stride=stride, padding=0,
                                            dilation=dilation,
                                            ceil_mode=ceil_mode)

        # conner case under ceil_mode=True CTL-6524/CB-6234
        out_shape = self.attrs['_output_shapes'][0] if '_output_shapes' in self.attrs else self.outputs[0].shape
        out_shape = transfer_shape(out_shape, self.outputs[0].shape.layout.name, 'NHWC') \
            if self.outputs[0].shape.layout.name not in ['Flat', 'NHWC'] else out_shape
        out_shape = [out_shape[0]] + [1] + list(out_shape)[1:] if len(out_shape) == 3 else out_shape
        _, out_h, out_w, out_c = out_shape
        current_outh, current_outw = y.shape[2: 4]
        if current_outh < out_h or current_outw < out_w:
            DEBUG('Pooling: conner case under ceil_mode=True.')
            auto_pad_h = out_h - current_outh
            auto_pad_w = out_w - current_outw
            padding = (0, auto_pad_w, 0, auto_pad_h)  # left, right, top, bottom
            y = torch.nn.functional.pad(y, padding, value=pad_val)

        y = clip_to_bit(y, 32, True, to_type=str(self.outputs[0].dtype))
        y = layout_convert(y, 'NCHW', 'NHWC')
        y = clip_to_bit(y, 32, True, o_type)
        self.outputs[0].irtensor = y
        return y

    elif self.params['method'] == 'L1':
        ceil_mode_pad_bottom = 0
        ceil_mode_pad_right = 0
        if ceil_mode:
            o_h = math.ceil((inp.shape[1] - kernel_size[0] + self.params['pad_top'] +
                            self.params['pad_bottom']) / stride[0]) + 1
            o_w = math.ceil((inp.shape[2] - kernel_size[1] + self.params['pad_left'] +
                            self.params['pad_right']) / stride[1]) + 1
            ceil_mode_pad_bottom = (o_h - 1) * stride[0] + kernel_size[0] - (padding[0] + padding[1]) - inp.shape[1]
            ceil_mode_pad_right = (o_w - 1) * stride[1] + kernel_size[1] - (padding[2] + padding[3]) - inp.shape[2]
        pad_val = -self.inputs[0].quantization.offset
        inp = torch.nn.functional.pad(inp, (0, 0, self.params['pad_left'],
                                            self.params['pad_right']+ceil_mode_pad_right,
                                            self.params['pad_top'], self.params['pad_bottom']+ceil_mode_pad_bottom, 0, 0), value=pad_val)
        inp = inp + self.inputs[0].quantization.offset

        kernel_x = kernel_size[1]
        kernel_y = kernel_size[0]
        stride_x = stride[1]
        stride_y = stride[0]
        out_height = math.floor((inp.shape[1] - (dilation[0]*(kernel_y-1)+1)) / stride_y) + 1
        out_width = math.floor((inp.shape[2] - (dilation[1]*(kernel_x-1)+1)) / stride_x) + 1
        scale = self.params['scale_value']
        shift = self.params['shift_value']
        output = torch.zeros((inp.shape[0], inp.shape[3], out_height, out_width))
        input_trans = inp.permute(0, 3, 1, 2)
        input_trans = torch.absolute(input_trans)

        for hi in range(out_height):
            h_start = hi * stride_y
            h_end = min(h_start + kernel_y + (kernel_y-1)*(dilation[0]-1), np.shape(inp)[1])
            for wi in range(out_width):
                w_start = wi * stride_x
                w_end = min(w_start + kernel_x + (kernel_x-1)*(dilation[1]-1), np.shape(inp)[2])
                sum_kernel = torch.sum(input_trans[:, :, h_start:h_end:dilation[0],
                                                   w_start:w_end:dilation[1]], dim=(2, 3))
                sum_kernel2 = torch.round(sum_kernel.type(torch.double) * scale / (2 ** shift)).type(torch.int32)
                output[:, :, hi, wi] = sum_kernel2
        y = output.permute(0, 2, 3, 1)

        y -= self.outputs[0].quantization.offset
        y = clamp(y, self.outputs[0].dtype)
        self.outputs[0].irtensor = y
        return y

    elif self.params['method'] == 'L2':
        ceil_mode_pad_bottom = 0
        ceil_mode_pad_right = 0
        if ceil_mode:
            o_h = math.ceil((inp.shape[1] - kernel_size[0] + self.params['pad_top'] +
                            self.params['pad_bottom']) / stride[0]) + 1
            o_w = math.ceil((inp.shape[2] - kernel_size[1] + self.params['pad_left'] +
                            self.params['pad_right']) / stride[1]) + 1
            ceil_mode_pad_bottom = (o_h - 1) * stride[0] + kernel_size[0] - (padding[0] + padding[1]) - inp.shape[1]
            ceil_mode_pad_right = (o_w - 1) * stride[1] + kernel_size[1] - (padding[2] + padding[3]) - inp.shape[2]
        pad_val = -self.inputs[0].quantization.offset
        inp = torch.nn.functional.pad(inp, (0, 0, self.params['pad_left'],
                                            self.params['pad_right']+ceil_mode_pad_right,
                                            self.params['pad_top'], self.params['pad_bottom']+ceil_mode_pad_bottom, 0, 0), value=pad_val)
        inp = inp + self.inputs[0].quantization.offset

        lut = self.constants['sqrt_lut'].betensor.long()
        if lut.numel() == 256:
            lut = torch.nn.functional.pad(lut, (0, 1), mode='constant', value=lut[255])
        kernel_x = kernel_size[1]
        kernel_y = kernel_size[0]
        stride_x = stride[1]
        stride_y = stride[0]
        out_height = math.floor((inp.shape[1] - (dilation[0]*(kernel_y-1)+1)) / stride_y) + 1
        out_width = math.floor((inp.shape[2] - (dilation[1]*(kernel_x-1)+1)) / stride_x) + 1
        scale = self.params['scale_value']
        shift = self.params['shift_value']
        output = torch.zeros((inp.shape[0], inp.shape[3], out_height, out_width))
        input_trans = inp.permute(0, 3, 1, 2)
        input_trans_square = torch.square(input_trans)

        for hi in range(out_height):
            h_start = hi * stride_y
            h_end = min(h_start + kernel_y + (kernel_y-1)*(dilation[0]-1), inp.shape[1])
            for wi in range(out_width):
                w_start = wi * stride_x
                w_end = min(w_start + kernel_x + (kernel_x-1)*(dilation[1]-1), inp.shape[2])
                sum_kernel = torch.sum(input_trans_square[:, :, h_start:h_end:dilation[0], w_start:w_end:dilation[1]],
                                       dim=(2, 3)).type(torch.float64)
                sum_kernel2 = torch.round(sum_kernel.type(torch.double) * scale / (2 ** shift)).type(torch.int32)
                sum_kernel2 = torch.clip(sum_kernel2, 0, 65535)
                tmp_shape = sum_kernel2.shape
                sum_kernel2 = sum_kernel2.flatten()
                sum_kernel2_lut = []
                # uint16 data, uint8 table
                for d in sum_kernel2:
                    lut_idx = (d >> 8) & 0xff
                    lut_delta = d & 0xff
                    dout = int(lut[lut_idx]) + ((int(lut[lut_idx + 1]) - int(lut[lut_idx])) * lut_delta >> 8)
                    sum_kernel2_lut.append(dout)
                sum_kernel2_lut = torch.asarray(sum_kernel2_lut, dtype=sum_kernel2.dtype)
                sum_kernel2_lut = sum_kernel2_lut.reshape(tmp_shape)
                output[:, :, hi, wi] = sum_kernel2_lut.type(torch.int32)
        y = output.permute(0, 2, 3, 1)
        y = clamp(y, self.outputs[0].dtype)
        self.outputs[0].irtensor = y
        return y

    elif self.params['method'] == 'AVG':
        i_b, i_h, i_w, i_c = inp.shape
        out_shape = self.attrs['_output_shapes'][0] if '_output_shapes' in self.attrs else self.outputs[0].shape
        out_shape = transfer_shape(out_shape, self.outputs[0].shape.layout.name, 'NHWC') \
            if self.outputs[0].shape.layout.name not in ['Flat', 'NHWC'] else out_shape
        out_shape = [out_shape[0]] + [1] + list(out_shape)[1:] if len(out_shape) == 3 else out_shape
        _, o_h, o_w, _ = out_shape
        ceil_mode_pad_bottom = 0
        ceil_mode_pad_right = 0
        if ceil_mode:
            if kernel_size[0] == 1 and stride[0] == 2:
                o_h = math.ceil((i_h + padding[0] + padding[1]-kernel_size[0])/stride[0])
                ceil_mode_pad_bottom = 0
            else:
                # When ceil_mode=True, Sliding windows that would start in the right padded region are ignored.
                # https://pytorch.org/docs/2.1/generated/torch.nn.AvgPool2d.html#torch.nn.AvgPool2d
                # o_h = math.ceil(
                #     (i_h + padding[0] + padding[1] - (1 + (kernel_size[0] - 1) * dilation[0])) / stride[0]) + 1
                ceil_mode_pad_bottom = (o_h - 1) * stride[0] + kernel_size[0] - (padding[0] + padding[1]) - i_h
            if kernel_size[1] == 1 and stride[1] == 2:
                o_w = math.ceil((i_w + padding[2] + padding[3]-kernel_size[1])/stride[1])
                ceil_mode_pad_right = 0
            else:
                # o_w = math.ceil(
                #     (i_w + padding[2] + padding[3] - (1 + (kernel_size[1] - 1) * dilation[1])) / stride[1]) + 1
                ceil_mode_pad_right = (o_w - 1) * stride[1] + kernel_size[1] - (padding[2] + padding[3]) - i_w

        # total_pad = [padding[0], padding[1] + ceil_mode_pad_bottom, padding[2], padding[3] + ceil_mode_pad_right]
        max_scale = 32768 if is_16b_scale else 256
        inp = inp.float()
        y = avg_pool(self, inp, kernel_size, stride, padding,
                     (ceil_mode_pad_bottom, ceil_mode_pad_right), max_scale, is_aiff)
        # after finish calculate the avg value of whole feature map, the result need to clip to the range of the output type
        y = clamp(y, self.outputs[0].dtype)
        if self.outputs[0].betensor.dim() == 3:
            y = torch.squeeze(y, dim=1)
        self.outputs[0].irtensor = y
        return y


def get_div_lut(kernel_size, max_scale):
    if kernel_size == 0:
        return 1, 0
    range_high = np.log2(max_scale) + 1
    scale = 1 / kernel_size
    for j in reversed(range(0, int(range_high))):
        do_scale = np.round(2 ** (np.log2(scale) - np.round(np.log2(scale)) + j)).astype(np.uint32)
        do_shift = abs(np.round(np.log2(scale)) - j).astype(np.uint32)
        if (do_scale < max_scale and do_scale % 2 != 0):
            break
    return do_scale, do_shift


def avg_pool(self, inputs, kernel, stride, pad_value, ceil_mod_pad, max_scale=256, is_aiff=False):
    # inputs alread pad the pads and ceil mode addtion pads
    i_b, i_h, i_w, i_c = inputs.shape
    ky = kernel[0]
    kx = kernel[1]
    sy = stride[0]
    sx = stride[1]
    # use pad value in ir, more than 'SAME' and VALID"
    pad_top, pad_bottom, pad_left, pad_right = pad_value
    pad_bottom += ceil_mod_pad[0]
    pad_right += ceil_mod_pad[1]
    o_h = math.floor((i_h - ky + pad_top + pad_bottom) / sy) + 1
    o_w = math.floor((i_w - kx + pad_left + pad_right) / sx) + 1

    # torch sum for fp16 data precision>np,cause mismatch
    if self.inputs[0].dtype.is_float:
        output = np.zeros((i_b, o_h, o_w, i_c))
        kernel_sum = np.zeros((i_b, o_h, o_w, i_c))
    else:
        output = torch.zeros((i_b, o_h, o_w, i_c), dtype=torch.float64)
        kernel_sum = torch.zeros((i_b, o_h, o_w, i_c), dtype=torch.float64)
    count_include_pad = self.params['count_include_pad'] if 'count_include_pad' in self.params else False
    ceil_mode = self.params['ceil_mode'] if 'ceil_mode' in self.params else False
    zerop = self.inputs[0].quantization.offset
    ceil_mode_pad_bottom = 0
    ceil_mode_pad_right = 0

    # height
    scales = np.zeros(o_h * o_w, dtype='int32')
    shifts = np.zeros(o_h * o_w, dtype='int32')
    box_sizes = np.zeros(o_h * o_w, dtype='int32')
    box_sizes_without_pad = np.zeros(o_h * o_w, dtype='int32')
    for ph in range(o_h):
        h_start = ph * sy - pad_top
        h_end = min(h_start + ky, i_h)
        h_start = max(h_start, 0)
        # width
        for pw in range(o_w):
            w_start = pw * sx - pad_left
            w_end = min(w_start + kx, i_w)
            w_start = max(w_start, 0)
            pos = ph * o_w + pw
            # create box/lut
            if count_include_pad:
                real_kernel_y = ky - ceil_mod_pad[0] if (ceil_mode and ph == o_h - 1) else ky
                real_kernel_x = kx - ceil_mod_pad[1] if (ceil_mode and pw == o_w - 1) else kx
                box_sizes[pos] = real_kernel_y * real_kernel_x
                box_sizes_without_pad[pos] = (h_end - h_start) * (w_end - w_start)
            else:
                box_sizes[pos] = (h_end - h_start) * (w_end - w_start)
                box_sizes_without_pad[pos] = box_sizes[pos]
            if not self.inputs[0].dtype.is_float:
                scales[pos], shifts[pos] = get_div_lut(box_sizes[pos], max_scale)
    # unify scale for aiff
    if is_aiff and not self.inputs[0].dtype.is_float:
        scales, shifts = _aiff_merge_shifts(torch.tensor(scales), torch.tensor(shifts))

    # height
    for ph in range(o_h):
        h_start = ph * sy - pad_top
        h_end = min(h_start + ky, i_h)
        h_start = max(h_start, 0)
        # width
        for pw in range(o_w):
            w_start = pw * sx - pad_left
            w_end = min(w_start + kx, i_w)
            w_start = max(w_start, 0)
            pos = ph * o_w + pw
            # sum
            if self.type == OpType.GlobalPool and self.inputs[0].dtype.is_int:
                kernel_sum[:, ph, pw, :] = torch.sum(
                    inputs[:, h_start:h_end, w_start:w_end, :].type(torch.int32), dim=(1, 2))
                output[:, ph, pw, :] = kernel_sum[:, ph, pw, :] / box_sizes[pos]
            elif self.inputs[0].dtype.is_float:
                kernel_sum[:, ph, pw, :] = np.sum(inputs.astype(
                    np.float32)[:, h_start:h_end, w_start:w_end, :], axis=(1, 2))
                output[:, ph, pw, :] = kernel_sum[:, ph, pw, :] / box_sizes[pos]
            else:
                scale = scales[pos]
                if isinstance(shifts, torch.Tensor) and shifts.dim() == 0:
                    shift = int(shifts)
                else:
                    shift = shifts[pos]
                if 'scale_value' in self.params and 'shift_value' in self.params:
                    scale = self.params['scale_value']
                    shift = self.params['shift_value']
                kernel_sum[:, ph, pw, :] = torch.sum(
                    inputs[:, h_start:h_end, w_start:w_end, :].type(torch.int32), dim=(1, 2))
                if count_include_pad:
                    kernel_sum[:, ph, pw, :] -= (box_sizes[pos] - box_sizes_without_pad[pos]) * zerop
                output[:, ph, pw, :] = torch.round(kernel_sum[:, ph, pw, :].type(
                    torch.double) * scale * (0.5 ** shift)).type(torch.int64)
    return output


def fp_pool(self):
    o_type = str(self.outputs[0].dtype)
    inp = self.inputs[0].irtensor

    # avgpool1d
    if inp.dim() == 3:
        inp = inp[:, None, :, :]

    count_include_pad = self.params.get('count_include_pad', default=True)
    kernel_size = self.params.get('kernel_y', 1), self.params.get('kernel_x', 1)
    stride = self.params.get("stride_y",  1), self.params.get("stride_x",  1)
    dilation = self.params.get("dilation_y",  1), self.params.get("dilation_x",  1)
    padding = (self.params.get('pad_top',  0),
               self.params.get('pad_bottom',  0),
               self.params.get('pad_left',  0),
               self.params.get('pad_right',  0))

    if self.type == OpType.GlobalPool:
        kernel_size = [inp.shape[1], inp.shape[2]]

    ceil_mode = self.params.get('ceil_mode',  default=False)
    inputs_type = self.inputs[0].dtype.torch
    if self.params['method'] == 'MAX':
        inp = layout_convert(inp, 'NHWC', 'NCHW')
        pad_val = torch.finfo(inputs_type).min
        padding_torch = padding[-2:] + padding[:-2]
        inp = torch.nn.functional.pad(inp, padding_torch, value=pad_val)
        y = torch.nn.functional.max_pool2d(inp.float(),
                                           kernel_size=kernel_size,
                                           stride=stride, padding=0,
                                           dilation=dilation,
                                           ceil_mode=ceil_mode)

        y = layout_convert(y, 'NCHW', 'NHWC')
        y = y.to(self.outputs[0].dtype.torch)

        self.outputs[0].irtensor = y
        return y

    elif self.params['method'] == 'MIN':
        inp = layout_convert(inp, 'NHWC', 'NCHW')
        pad_val = torch.finfo(inputs_type).max
        padding_torch = padding[-2:] + padding[:-2]
        inp = torch.nn.functional.pad(inp, padding_torch, value=pad_val)
        y = -torch.nn.functional.max_pool2d(-inp.float(),
                                            kernel_size=kernel_size,
                                            stride=stride, padding=0,
                                            dilation=dilation,
                                            ceil_mode=ceil_mode)

        y = layout_convert(y, 'NCHW', 'NHWC')
        y = y.to(self.outputs[0].dtype.torch)
        self.outputs[0].irtensor = y
        return y

    elif self.params['method'] == 'AVG':
        i_b, i_h, i_w, i_c = inp.shape
        ceil_mode_pad_bottom = 0
        ceil_mode_pad_right = 0
        if ceil_mode:
            o_h = math.ceil((i_h - kernel_size[0] + padding[0] + padding[1]) / stride[0]) + 1
            o_w = math.ceil((i_w - kernel_size[1] + padding[2] + padding[3]) / stride[1]) + 1
            ceil_mode_pad_bottom = (o_h - 1) * stride[0] + kernel_size[0] - (padding[0] + padding[1]) - i_h
            ceil_mode_pad_right = (o_w - 1) * stride[1] + kernel_size[1] - (padding[2] + padding[3]) - i_w

        # total_pad = [padding[0], padding[1] + ceil_mode_pad_bottom, padding[2], padding[3] + ceil_mode_pad_right]
        inp = inp.float().cpu().numpy()
        y = avg_pool(self, inp, kernel_size, stride, padding, (ceil_mode_pad_bottom, ceil_mode_pad_right), None)
        # after finish calculate the avg value of whole feature map, the result need to clip to the range of the output type
        y = y.astype(self.outputs[0].dtype.np)
        self.outputs[0].irtensor = y
        return y

    elif self.params['method'] in ["L1", "L2"]:
        p = 1 if self.params["method"] == "L1" else 2
        out_shape = self.outputs[0].irtensor.shape
        padding_lp = (self.params.get("pad_left", 0), 
                      self.params.get("pad_right", 0), 
                      self.params.get("pad_top", 0), 
                      self.params.get("pad_bottom", 0))
        inp = inp.permute(0, 3, 1, 2)
        inp = torch.nn.functional.pad(inp, padding_lp, value=0)
        batch = inp.shape[0]
        _, out_h, out_w, channel = out_shape
        inp_h, inp_w = inp.shape[2], inp.shape[3]
        psum_placeholder = torch.zeros((batch, channel), device=inp.device)
        output = torch.zeros((batch, channel, out_h, out_w), device=inp.device)
        feature = inp.to(torch.float32)
        if self.params["method"] == "L1":
            feature = torch.abs(feature)
        else:
            feature = feature * feature
        startX = torch.arange(out_w) * stride[1]
        startY = torch.arange(out_h) * stride[0]
        endY = startY + (kernel_size[0] - 1) * dilation[0] + 1
        endX = startX + (kernel_size[1] - 1) * dilation[1] + 1
        if ceil_mode:
            padding_right = max(0, endX.max() - inp_w)
            padding_bottom = max(0, endY.max() - inp_h)
            padding_diff = (0, padding_right, 0, padding_bottom)
            feature = torch.nn.functional.pad(feature, padding_diff, value=0)

        for r_idx in range(out_h):
            for c_idx in range(out_w):
                pool_value = feature[:, :, startY[r_idx] : endY[r_idx] : dilation[0], startX[c_idx] : endX[c_idx] : dilation[1]]
                poolsum = torch.sum(pool_value, dim=[2, 3])
                psum_placeholder = torch.cat((psum_placeholder, poolsum.reshape(batch, -1)), dim=-1)
                pool_output = torch.pow(poolsum, 1 / p)
                output[:, :, r_idx, c_idx] = pool_output
        outp = output.permute(0, 2, 3, 1)
        self.outputs[0].irtensor = outp

@register_gen_weight(OpType.Pooling)
def gen_random_weight(self):
    if 'sqrt_lut' in self.constants:  # for L2
        sqrt_lut = np.random.randint(0, 255, 256, dtype=np.uint8)
        sqrt_lut = np.sort(sqrt_lut)
        self.constants['sqrt_lut'].betensor = sqrt_lut


@register_op_metric(OpType.Pooling)
@register_op_metric(OpType.GlobalPool)
@check_fp_data
def metric(self, outputs, index=None):
    def call(r, u, c, t):
        if "float" in t:
            from AIPUBuilder.core import Target
            ulp_thresh = 3 if Target().name.startswith('X3') else 2  # CP-24410
            in_shape = self.inputs[0].shape
            ky = self.params.get('kernel_y', 0)
            kx = self.params.get('kernel_x', 0)
            if self.type == OpType.GlobalPool or (in_shape[1] == ky and in_shape[2] == kx) or kx * ky >=13*13:
                return np.all(u <= ulp_thresh)
            else:
                return np.all(u <= 1)
        else:
            return np.all(r == 0)

    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]

    layer_dtypes = [str(o.dtype) for o in self.outputs] if index is None else [str(self.outputs[index].dtype)]

    rel_diffs = relative_diff(outputs, groundtruths)
    ulp_diffs = ulp_diff(outputs, groundtruths)
    cos_dists = cos_distance(outputs, groundtruths)
    if self.outputs[0].dtype.is_float:
        return all(call(r, u, c, t) for r, u, c, t in zip(rel_diffs, ulp_diffs, cos_dists, layer_dtypes))

    return _OP_METRICS['_all'](self, outputs, index)
