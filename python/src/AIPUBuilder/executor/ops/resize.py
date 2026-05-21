import numpy as np
import torch
from .gt_util import math_utils as pyfma
from AIPUBuilder.core import OpType
from ..engine import register_gt, register_gen_weight, _OP_METRICS, register_op_metric
from .gt_util.metric import relative_diff, ulp_diff, cos_distance, check_fp_data, get_parent_node
from .gt_util.quant_utils import clamp, linear_requantize
from .gt_util.layout_transfer import transfer_shape
from .gt_util.dtype_utils import torch_to_numpy, numpy_to_torch
from .gt_util.math_utils import fp32_to_fp24


@register_gt(OpType.Resize)
def resize(self, *args, **kwargs):
    input_tensor = torch_to_numpy(self.inputs[0].irtensor, self.inputs[0].dtype.np)

    output_type = str(self.outputs[0].dtype)
    output_shape = self.outputs[0].irtensor.shape
    if '_output_shapes' in self.attrs:
        output_shape = self.attrs['_output_shapes'][0]
        if len(output_shape) == 5:
            output_shape = transfer_shape(output_shape, self.outputs[0].shape.layout.name, 'NHWC')

    method = self.params['method'].lower()
    if "nearest_mode" in self.params.keys() and self.params['nearest_mode']:
        nearest_mode = self.params['nearest_mode'].lower()
    else:
        nearest_mode = "round_prefer_floor"
    antialias = self.params.get('antialias', 'false')

    is_aiff = True
    tpc_or_aiff = 'aiff'
    if 'is_aiff' in kwargs.keys() and kwargs['is_aiff'] == False:
        is_aiff = False
        tpc_or_aiff = 'tpc'
    self.attrs["is_aiff"] = is_aiff
    print('run resize: %s version !' % tpc_or_aiff)

    if self.inputs[0].dtype.is_float:
        if method == 'bilinear' and str(antialias).lower() == 'true':
            output = resize_bilinear_antialias(self, input_tensor, output_shape)
        elif method == 'bilinear':
            output = accelerate_resize_bilinear(self, input_tensor, output_shape)
        elif method == 'nearest':
            output = nearest_resize(self, input_tensor, output_shape)
        else:
            raise ValueError('resize do not support method = {}.'.format(self.method))
    else:
        if method == 'bilinear' and str(antialias).lower() == 'true':
            output = resize_bilinear_antialias(self, input_tensor, output_shape)
            output = clamp(output, output_type)
        elif method == 'bilinear' or method == 'nearest':
            b, out_h, out_w, c = output_shape
            b, h, w, c = input_tensor.shape
            output = []
            for pb in range(b):
                input_data = input_tensor[pb].reshape((h, w, c))
                if ('simu_op' in kwargs.keys() and kwargs['simu_op'] == 'itp') or is_aiff:
                    output_batch = resize_with_method_itp(
                        self, input_data, [out_h, out_w], self.params['mode'].lower(), method=method)
                else:
                    output_batch = resize_with_method(self, input_data, [out_h, out_w], self.params['mode'].lower(),
                                                      nearest_mode, method=method)
                output_batch = np.clip(output_batch, np.iinfo(output_type).min, np.iinfo(output_type).max)
                output.append(output_batch.astype(np.int64))
            output = np.asarray(output)
            output = np.reshape(output, [b, out_h, out_w, c])
        else:
            # use upsample in benchmark actually
            input_tensor = input_tensor.repeat(self.params['height_factor'], axis=1)
            output = input_tensor.repeat(self.params['width_factor'], axis=2)
    self.outputs[0].irtensor = output
    return output


@register_gt(OpType.Resize, target='X3_1304')
@register_gt(OpType.Resize, target='X3_1302')
@register_gt(OpType.Resize, target='X3_1204')
@register_gt(OpType.Resize, target='X3_1202')
@register_gt(OpType.Resize, target='X3P_1304')
@register_gt(OpType.Resize, target='X3P_1302')
@register_gt(OpType.Resize, target='X3P_1204')
@register_gt(OpType.Resize, target='X3P_1202')
@register_gt(OpType.Resize, target='X3S_1304')
def resize(self, *args, **kwargs):
    input_tensor = torch_to_numpy(self.inputs[0].irtensor, self.inputs[0].dtype.np)

    output_shape = self.outputs[0].irtensor.shape
    if '_output_shapes' in self.attrs:
        output_shape = self.attrs['_output_shapes'][0]
        if len(output_shape) == 5:
            output_shape = transfer_shape(output_shape, self.outputs[0].shape.layout.name, 'NHWC')
    method = self.params['method'].lower()
    if "nearest_mode" in self.params.keys() and self.params['nearest_mode']:
        nearest_mode = self.params['nearest_mode'].lower()
    else:
        nearest_mode = "round_prefer_floor"
    antialias = self.params.get('antialias', 'false')
    is_float = self.inputs[0].dtype.is_float

    is_aiff = True
    tpc_or_aiff = 'aiff'
    if 'is_aiff' in kwargs.keys() and kwargs['is_aiff'] == False:
        is_aiff = False
        tpc_or_aiff = 'tpc'
    self.attrs["is_aiff"] = is_aiff
    print('run resize: %s version !' % tpc_or_aiff)

    # fp tpc version
    if is_float and not (is_aiff or ('simu_op' in kwargs.keys() and kwargs['simu_op'] == 'itp')):
        if method == 'bilinear' and str(antialias).lower() == 'true':
            output = resize_bilinear_antialias(self, input_tensor, output_shape)
        elif method == 'bilinear':
            output = accelerate_resize_bilinear(self, input_tensor, output_shape)
        elif method == 'nearest':
            output = nearest_resize(self, input_tensor, output_shape)
        else:
            raise ValueError('resize do not support method = {}.'.format(self.method))
    else:
        if method == 'bilinear' and str(antialias).lower() == 'true':
            output = resize_bilinear_antialias(self, input_tensor, output_shape)
            output = clamp(output, self.outputs[0].dtype)
        elif method == 'bilinear' or method == 'nearest':
            b, out_h, out_w, c = output_shape
            b, h, w, c = input_tensor.shape
            output = []
            for pb in range(b):
                input_data = input_tensor[pb].reshape((h, w, c))
                if ('simu_op' in kwargs.keys() and kwargs['simu_op'] == 'itp') or is_aiff:
                    # output_batch_ = resize_with_method_itp_x3(
                    #     self, input_data, [out_h, out_w], self.params['mode'].lower(), method, is_float)
                    output_batch = output_batch = resize_with_method_itp_x3_vec(
                        self, input_data, [out_h, out_w], self.params['mode'].lower(), method, is_float)
                else:
                    output_batch = resize_with_method(
                        self, input_data, [out_h, out_w], self.params['mode'].lower(), nearest_mode, method=method)
                if not is_float:
                    output_batch = np.clip(output_batch, np.iinfo(
                        str(self.outputs[0].dtype)).min, np.iinfo(str(self.outputs[0].dtype)).max)
                    output.append(output_batch.astype(np.int64))
                else:
                    output.append(output_batch.astype(np.float32))
            output = np.asarray(output)
            output = np.reshape(output, [b, out_h, out_w, c])
        else:
            # use upsample in benchmark actually
            input_tensor = input_tensor.repeat(self.params['height_factor'], axis=1)
            output = input_tensor.repeat(self.params['width_factor'], axis=2)
    self.outputs[0].irtensor = output
    return output

# will be remove after resize_with_method_itp_x3_vec is work right in more case


def resize_with_method_itp_x3(self, data, out, mode, method, is_float=False):
    input_h, input_w, output_h, output_w = data.shape[0], data.shape[1], out[0], out[1]
    ratio_y = self.params.get("ratio_y", 0)
    ratio_x = self.params.get("ratio_x", 0)
    input_h = get_real_input(input_h, output_h, 'ratio_y', self.params)
    input_w = get_real_input(input_w, output_w, 'ratio_x', self.params)
    interval_h = set_linear_interval(input_h, output_h, mode, ratio_y)
    interval_w = set_linear_interval(input_w, output_w, mode, ratio_x)
    offset_h = set_linear_offset(input_h, output_h, mode, ratio_y)
    offset_w = set_linear_offset(input_w, output_w, mode, ratio_x)
    input_dtype = self.inputs[0].irtensor.dtype

    res = []
    lut_data = data
    if is_float:
        lut_data = numpy_to_torch(data, self.inputs[0].dtype.np)
        lut_data = lut_data.to(torch.float32)
    else:
        lut_data = np.int32(data)
    inputs_shape = self.inputs[0].betensor.shape
    outputs_type = str(self.outputs[0].dtype)
    if len(inputs_shape) == 4:
        b, h, w, c = inputs_shape
    else:
        b, h, w, c = inputs_shape[0], inputs_shape[2], inputs_shape[3], inputs_shape[1] * inputs_shape[4]
    if method == 'bilinear':
        # bilinear do W lut first, get 16-bit intermediate result, than H lut
        # hardware use rounding to get 16-bit delta in X3
        for i in range(output_h):
            s_y = max(i * interval_h + offset_h, 0)
            y0 = min(max((s_y >> 24), 0), h - 1)
            y1 = min(y0 + 1, h - 1)
            delta_y = ((s_y & 0xffff00) >> 8) + ((s_y & 0x80) >> 7)

            for j in range(output_w):
                s_x = max(j * interval_w + offset_w, 0)
                x0 = min(max((s_x >> 24), 0), w - 1)
                x1 = min(x0 + 1, w - 1)
                delta_x = ((s_x & 0xffff00) >> 8) + ((s_x & 0x80) >> 7)

                # new feature in ITP interpolation:
                #   i8 data left shift 8 to int16 data for lut
                #   u8 data left shift 7 to int16 data for lut
                #   fp16 data in intermediate result for fp input
                if is_float:
                    w0 = fp32_to_fp24(fp32_to_fp24(
                        lut_data[y0, x1, ...] - lut_data[y0, x0, ...]) * delta_x) * (0.5 ** 16) + lut_data[y0, x0, ...]
                    w0 = fp32_to_fp24(w0).to(input_dtype)
                    w0 = w0.to(torch.float32)
                    w1 = fp32_to_fp24(fp32_to_fp24(
                        lut_data[y1, x1, ...] - lut_data[y1, x0, ...]) * delta_x) * (0.5 ** 16) + lut_data[y1, x0, ...]
                    w1 = fp32_to_fp24(w1).to(input_dtype)
                    w1 = w1.to(torch.float32)
                    result = fp32_to_fp24(fp32_to_fp24(w1 - w0) * delta_y) * (0.5 ** 16) + w0
                    result = fp32_to_fp24(result)
                    result = np.float32(result.cpu().numpy())
                else:
                    if outputs_type in ["int8"]:
                        w0 = (((lut_data[y0, x1, ...] << 8) - (lut_data[y0, x0, ...] << 8)) * delta_x + (
                            lut_data[y0, x0, ...] << (8 + 16)) >> 16)
                        w1 = (((lut_data[y1, x1, ...] << 8) - (lut_data[y1, x0, ...] << 8)) * delta_x + (
                            lut_data[y1, x0, ...] << (8 + 16)) >> 16)
                        result = np.float32((w1 - w0) * delta_y + (w0 << 16) >> 16)
                        result = np.round(result * (0.5 ** 8))
                    elif outputs_type in ["uint8"]:
                        w0 = (((lut_data[y0, x1, ...] << 7) - (lut_data[y0, x0, ...] << 7)) * delta_x + (
                            lut_data[y0, x0, ...] << (7 + 16)) >> 16)
                        w1 = (((lut_data[y1, x1, ...] << 7) - (lut_data[y1, x0, ...] << 7)) * delta_x + (
                            lut_data[y1, x0, ...] << (7 + 16)) >> 16)
                        result = np.float32((w1 - w0) * delta_y + (w0 << 16) >> 16)
                        result = np.round(result * (0.5 ** 7))
                    else:
                        w0 = ((lut_data[y0, x1, ...] - lut_data[y0, x0, ...]) * delta_x + (
                            lut_data[y0, x0, ...] << 16) >> 16)
                        w1 = ((lut_data[y1, x1, ...] - lut_data[y1, x0, ...]) * delta_x + (
                            lut_data[y1, x0, ...] << 16) >> 16)
                        result = ((w1 - w0) * delta_y + (w0 << 16) >> 16)
                res.append(result)

    if method == 'nearest':
        for i in range(output_h):
            s_y = max(i * interval_h + offset_h, 0)
            s_y = (s_y >> 24) + ((s_y >> 23) & 0x1)
            y = int(min(max(np.floor(s_y), 0), input_h - 1))
            for j in range(output_w):
                s_x = max(j * interval_w + offset_w, 0)
                s_x = (s_x >> 24) + ((s_x >> 23) & 0x1)
                x = int(min(max(np.floor(s_x), 0), input_w - 1))
                res.append(data[y, x, ...])

    return np.array(res)


def resize_with_method_itp_x3_vec(self, data, out, mode, method, is_float=False):
    input_h, input_w, output_h, output_w = data.shape[0], data.shape[1], out[0], out[1]
    ratio_y = self.params.get("ratio_y", 0)
    ratio_x = self.params.get("ratio_x", 0)
    input_h = get_real_input(input_h, output_h, 'ratio_y', self.params)
    input_w = get_real_input(input_w, output_w, 'ratio_x', self.params)
    interval_h = set_linear_interval(input_h, output_h, mode, ratio_y)
    interval_w = set_linear_interval(input_w, output_w, mode, ratio_x)
    offset_h = set_linear_offset(input_h, output_h, mode, ratio_y)
    offset_w = set_linear_offset(input_w, output_w, mode, ratio_x)
    input_dtype = self.inputs[0].irtensor.dtype

    res = []
    lut_data = data
    if is_float:
        lut_data = numpy_to_torch(data, self.inputs[0].dtype.np)
        lut_data = lut_data.to(torch.float32)
    else:
        lut_data = np.int32(data)
    inputs_shape = self.inputs[0].betensor.shape
    outputs_type = str(self.outputs[0].dtype)
    if len(inputs_shape) == 4:
        b, h, w, c = inputs_shape
    else:
        b, h, w, c = inputs_shape[0], inputs_shape[2], inputs_shape[3], inputs_shape[1] * inputs_shape[4]

    if method == 'bilinear':
        index_h = torch.arange(output_h)
        s_y = index_h * interval_h + offset_h
        s_y = torch.clamp(s_y, 0, s_y.max())
        y0 = torch.clamp(s_y >> 24, 0, h - 1)
        y1 = y0 + 1
        y1[y1 > h - 1] = h - 1
        delta_y = ((s_y & 0xffff00) >> 8) + ((s_y & 0x80) >> 7)

        index_w = torch.arange(output_w)
        s_x = index_w * interval_w + offset_w
        s_x = torch.clamp(s_x, 0, s_x.max())
        x0 = torch.clamp(s_x >> 24, 0, w - 1)
        x1 = x0 + 1
        x1[x1 > w - 1] = w - 1
        delta_x = ((s_x & 0xffff00) >> 8) + ((s_x & 0x80) >> 7)
        if is_float:
            w0 = fp32_to_fp24(fp32_to_fp24(lut_data[y0][:, x1, :] - lut_data[y0][:, x0, :])
                              * delta_x.reshape(1, -1, 1)) * (0.5 ** 16) + lut_data[y0][:, x0, :]
            w0 = fp32_to_fp24(w0).to(input_dtype)
            w0 = w0.to(torch.float32)
            w1 = fp32_to_fp24(fp32_to_fp24(lut_data[y1][:, x1, :] - lut_data[y1][:, x0, :])
                              * delta_x.reshape(1, -1, 1)) * (0.5 ** 16) + lut_data[y1][:, x0, :]
            w1 = fp32_to_fp24(w1).to(input_dtype)
            w1 = w1.to(torch.float32)
            result = fp32_to_fp24(fp32_to_fp24(w1 - w0) * delta_y.reshape(-1, 1, 1)) * (0.5 ** 16) + w0
            result = fp32_to_fp24(result)
            result = np.float32(result.cpu().numpy())
            return result
        else:
            delta_x = torch_to_numpy(delta_x, "int64")
            delta_y = torch_to_numpy(delta_y, "int64")
            if outputs_type in ["int8"]:
                w0 = (((lut_data[y0][:, x1, :] << 8) - (lut_data[y0][:, x0, :] << 8)) * delta_x.reshape(1, -1, 1) + (
                    lut_data[y0][:, x0, :] << (8 + 16)) >> 16)
                w1 = (((lut_data[y1][:, x1, :] << 8) - (lut_data[y1][:, x0, :] << 8)) * delta_x.reshape(1, -1, 1) + (
                    lut_data[y1][:, x0, :] << (8 + 16)) >> 16)
                result = np.float32((w1 - w0) * delta_y.reshape(-1, 1, 1) + (w0 << 16) >> 16)
                result = np.round(result * (0.5 ** 8))
                return result
            elif outputs_type in ["uint8"]:
                w0 = (((lut_data[y0][:, x1, :] << 7) - (lut_data[y0][:, x0, :] << 7)) * delta_x.reshape(1, -1, 1) + (
                    lut_data[y0][:, x0, :] << (7 + 16)) >> 16)
                w1 = (((lut_data[y1][:, x1, :] << 7) - (lut_data[y1][:, x0, :] << 7)) * delta_x.reshape(1, -1, 1) + (
                    lut_data[y1][:, x0, :] << (7 + 16)) >> 16)
                result = np.float32((w1 - w0) * delta_y.reshape(-1, 1, 1) + (w0 << 16) >> 16)
                result = np.round(result * (0.5 ** 7))
                return result
            else:
                w0 = ((lut_data[y0][:, x1, :] - lut_data[y0][:, x0, :]) * delta_x.reshape(1, -1, 1) + (
                    lut_data[y0][:, x0, :] << 16) >> 16)
                w1 = ((lut_data[y1][:, x1, :] - lut_data[y1][:, x0, :]) * delta_x.reshape(1, -1, 1) + (
                    lut_data[y1][:, x0, :] << 16) >> 16)
                result = ((w1 - w0) * delta_y.reshape(-1, 1, 1) + (w0 << 16) >> 16)
                return result

    if method == 'nearest':
        index_h = np.arange(output_h)
        s_y = index_h * interval_h + offset_h
        s_y = (s_y >> 24) + ((s_y >> 23) & 0x1)
        y = np.clip(np.floor(s_y), 0, input_h - 1)
        index_w = np.arange(output_w)
        s_x = index_w * interval_w + offset_w
        s_x = (s_x >> 24) + ((s_x >> 23) & 0x1)
        x = np.clip(np.floor(s_x), 0, input_w - 1)
        return data[y][:, x, :]

    return np.array(res)


def scale(self, inp, oup, mode):
    if mode == "align_corners" and oup > 1:
        return (inp - 1) / (oup - 1)
    else:
        return inp / oup


def scaler(self, i, scale, mode):
    if mode == "half_pixel":
        return (i + 0.5) * scale - 0.5
    elif mode == "pytorch_half_pixel":
        return 0
    elif mode == "tf_half_pixel_for_nn":
        return (i + 0.5) * scale
    # align_corners
    else:
        return i * scale


def convert_mode(self, oup, mode):
    if mode == "pytorch_half_pixel" and oup > 1:
        return "half_pixel"
    else:
        return mode


def nearest(self, origin, mode, is_down_sample):
    if mode == "floor":
        return np.floor(origin)
    elif mode == "ceil":
        return np.ceil(origin)
    elif mode == "round_prefer_ceil":
        return np.floor(origin + 0.5)
    elif mode == "simple":
        if is_down_sample:
            return np.ceil(origin)
        else:
            return origin
    # round_prefer_floor
    else:
        return np.ceil(origin - 0.5)


def resize_with_method(self, data, out, mode, nearest_mode, method='nearest'):
    """
    Calculate the value of each output point by nearest or bilinear interpolation method
    :param align_corners: https://aijishu.com/a/1060000000116597
    :param half_pixel:
    :param method:
    :return:
    """
    input_h, input_w, output_h, output_w = data.shape[0], data.shape[1], out[0], out[1]
    s_h = scale(self, input_h, output_h, mode)
    s_w = scale(self, input_w, output_w, mode)
    mode_h = convert_mode(self, output_h, mode)
    mode_w = convert_mode(self, output_w, mode)
    down_sample_h = output_h < input_h
    down_sample_w = output_w < input_w

    res = []

    # https://zhuanlan.zhihu.com/p/110754637
    if method == 'bilinear':
        scale_num = 2 ** self.params['interp_shift_value']
        for i in range(output_h):
            s_y = scaler(self, i, s_h, mode_h)
            y0 = int(max(np.floor(s_y), 0))
            y1 = int(min(np.ceil(s_y), input_h - 1))
            for j in range(output_w):
                s_x = scaler(self, j, s_w, mode_w)
                x0 = int(max(np.floor(s_x), 0))
                x1 = int(min(np.ceil(s_x), input_w - 1))

                Q11 = np.floor((1 - (s_x - x0)) * (1 - (s_y - y0)) * scale_num + 0.5)
                Q21 = np.floor((1 - (s_x - x0)) * (s_y - y0) * scale_num + 0.5)
                Q12 = np.floor((s_x - x0) * (1 - (s_y - y0)) * scale_num + 0.5)
                Q22 = np.floor((s_x - x0) * (s_y - y0) * scale_num + 0.5)

                result = data[y0, x0, ...] * Q11 + data[y1, x0, ...] * Q21 + \
                    data[y0, x1, ...] * Q12 + data[y1, x1, ...] * Q22
                res.append(np.around(result / scale_num))

    if method == 'nearest':
        for i in range(output_h):
            s_y = scaler(self, i, s_h, mode_h)
            y = int(min(max(nearest(self, s_y, nearest_mode, down_sample_h), 0), input_h - 1))
            for j in range(output_w):
                s_x = scaler(self, j, s_w, mode_w)
                x = int(min(max(nearest(self, s_x, nearest_mode, down_sample_w), 0), input_w - 1))
                res.append(data[y, x, ...])

    return np.array(res)


def resize_with_method_itp(self, data, out, mode, method='nearest'):
    input_h, input_w, output_h, output_w = data.shape[0], data.shape[1], out[0], out[1]
    ratio_y = self.params.get("ratio_y", 0)
    ratio_x = self.params.get("ratio_x", 0)
    input_h = get_real_input(input_h, output_h, 'ratio_y', self.params)
    input_w = get_real_input(input_w, output_w, 'ratio_x', self.params)
    interval_h = set_linear_interval(input_h, output_h, mode, ratio_y)
    interval_w = set_linear_interval(input_w, output_w, mode, ratio_x)
    offset_h = set_linear_offset(input_h, output_h, mode, ratio_y)
    offset_w = set_linear_offset(input_w, output_w, mode, ratio_x)

    res = []
    lut_data = np.int32(data)
    inputs_shape = self.inputs[0].betensor.shape
    outputs_type = str(self.outputs[0].dtype)
    if len(inputs_shape) == 4:
        b, h, w, c = inputs_shape
    else:
        b, h, w, c = inputs_shape[0], inputs_shape[2], inputs_shape[3], inputs_shape[1] * inputs_shape[4]
    if method == 'bilinear':
        # bilinear do W lut first, get 16-bit intermediate result, than H lut
        # hardware use rounding to get 8-bit delta
        for i in range(output_h):
            s_y = max(i * interval_h + offset_h, 0)
            y0 = min(max((s_y >> 24), 0), h - 1)
            y1 = min(y0 + 1, h - 1)
            delta_y = ((s_y & 0xff0000) >> 16) + ((s_y & 0x008000) >> 15)
            for j in range(output_w):
                s_x = max(j * interval_w + offset_w, 0)
                x0 = min(max((s_x >> 24), 0), w - 1)
                x1 = min(x0 + 1, w - 1)
                delta_x = ((s_x & 0xff0000) >> 16) + ((s_x & 0x008000) >> 15)
                # new feature in ITP interpolation: i8 data left shift 8 to int16 data for lut
                #                                   u8 data left shift 7 to int16 data for lut
                if outputs_type in ["int8"]:
                    w0 = (((lut_data[y0, x1, ...] << 8) - (lut_data[y0, x0, ...] << 8)) * delta_x + (
                        lut_data[y0, x0, ...] << 16) >> 8)
                    w1 = (((lut_data[y1, x1, ...] << 8) - (lut_data[y1, x0, ...] << 8)) * delta_x + (
                        lut_data[y1, x0, ...] << 16) >> 8)
                    result = np.float32((w1 - w0) * delta_y + (w0 << 8) >> 8)
                    result = np.round(result * (0.5 ** 8))
                elif outputs_type in ["uint8"]:
                    w0 = (((lut_data[y0, x1, ...] << 7) - (lut_data[y0, x0, ...] << 7)) * delta_x + (
                        lut_data[y0, x0, ...] << 15) >> 8)
                    w1 = (((lut_data[y1, x1, ...] << 7) - (lut_data[y1, x0, ...] << 7)) * delta_x + (
                        lut_data[y1, x0, ...] << 15) >> 8)
                    result = np.float32((w1 - w0) * delta_y + (w0 << 8) >> 8)
                    result = np.round(result * (0.5 ** 7))
                else:
                    w0 = ((lut_data[y0, x1, ...] - lut_data[y0, x0, ...]) * delta_x + (
                        lut_data[y0, x0, ...] << 8) >> 8)
                    w1 = ((lut_data[y1, x1, ...] - lut_data[y1, x0, ...]) * delta_x + (
                        lut_data[y1, x0, ...] << 8) >> 8)
                    result = ((w1 - w0) * delta_y + (w0 << 8) >> 8)
                res.append(result)

    if method == 'nearest':
        for i in range(output_h):
            s_y = max(i * interval_h + offset_h, 0)
            s_y = (s_y >> 24) + ((s_y >> 23) & 0x1)
            y = int(min(max(np.floor(s_y), 0), input_h - 1))
            for j in range(output_w):
                s_x = max(j * interval_w + offset_w, 0)
                s_x = (s_x >> 24) + ((s_x >> 23) & 0x1)
                x = int(min(max(np.floor(s_x), 0), input_w - 1))
                res.append(data[y, x, ...])

    return np.array(res)


def get_real_input(ix, ox, ratio_name, params):
    """
        support resize tiling
    """
    _ratio = 0.
    if ratio_name in params and params[ratio_name] is not None:
        _ratio = params[ratio_name]
    mode = params["mode"].lower()
    if mode != "align_corners" and _ratio != 0. and np.floor(_ratio * ix) != int(ox):
        assert np.floor(_ratio * (ix - 1)) == int(ox)
        return ix - 1
    return ix


def set_linear_interval(ix, ox, mode, ratio):
    interval = ix / ox
    if mode == "align_corners" and ix >= 2 and ox >= 2:
        interval = (ix - 1) / (ox - 1)
    elif mode == "half_pixel_symmetric" and ratio > 0:
        interval = 1 / ratio
    assert interval <= 16, "interval is too large: {}".format(interval)
    assert interval >= 1.0/64, "interval is too small: {}".format(interval)
    interval_i = int(round(interval * (1 << 24)))

    return interval_i


def set_linear_offset(ix, ox, mode, ratio):
    offset = 0.0
    if mode == "pytorch_half_pixel" and ox > 1:
        offset = ix / ox * 0.5 - 0.5
    elif mode == "half_pixel":
        offset = ix / ox * 0.5 - 0.5
    elif mode == "tf_half_pixel_for_nn":
        offset = ix / ox * 0.5
    elif mode == "half_pixel_symmetric":
        if ratio > 0:
            offset = (ix - 1 + (1 - ox) / ratio) / 2
        else:
            offset = ix / ox * 0.5 - 0.5
    offset_i = int(round(offset * (1 << 24)))

    return offset_i


def resize_bilinear_antialias(self, input_data, output_shape):
    input_data = torch.from_numpy(input_data.astype('float32'))
    in_shape = list(self.inputs[0].shape)
    dev = self.inputs[0].betensor.device
    mode = self.params['mode']
    exclude_outside = self.params['exclude_outside']
    coordination_shift = self.params.get('interp_shift_value', 0)
    coordination_shift = 0 if coordination_shift == None else coordination_shift
    input_height, input_width = in_shape[1:3]
    batch_size, output_height, output_width, channel = output_shape
    output_data = torch.zeros([batch_size, output_height, output_width, channel], device=input_data.device)

    weight_coefficients_x = self.constants['weight_coefficients_x'].betensor
    weight_coefficients_y = self.constants['weight_coefficients_y'].betensor
    bound_x_min = self.constants['bound_x_min'].betensor
    bound_x_max = self.constants['bound_x_max'].betensor
    bound_y_min = self.constants['bound_y_min'].betensor
    bound_y_max = self.constants['bound_y_max'].betensor
    windowsize_w = weight_coefficients_x.shape[0] // output_width
    windowsize_h = weight_coefficients_y.shape[0] // output_height
    image_temp_buffer = torch.zeros([batch_size, input_height, output_width, channel], device=input_data.device)

    # horizon interpolate
    for w in range(output_width):
        xmin = bound_x_min[w]
        xmax = bound_x_max[w]
        output = torch.zeros([batch_size, input_height, channel], device=input_data.device)
        for i in range(xmin, xmax):
            tmp_data = input_data[:, :, i, :] * weight_coefficients_x[w * windowsize_w + i - xmin]
            output += tmp_data
        image_temp_buffer[:, :, w, :] = output
    # to be consistent with lib
    if coordination_shift != 0:  # quant version
        image_temp_buffer = linear_requantize(image_temp_buffer, 1, coordination_shift, str(self.inputs[0].dtype))

    # vertical interpolate
    for h in range(output_height):
        xmin = bound_y_min[h]
        xmax = bound_y_max[h]
        tmp_data = (image_temp_buffer[:, xmin:xmax, :, :] * weight_coefficients_y[h *
                                                                                  windowsize_h: h*windowsize_h + xmax - xmin].reshape([1, -1, 1, 1]))
        output = np.sum(tmp_data.numpy(), axis=1, keepdims=False)
        output_data[:, h, :, :] = torch.from_numpy(output)

    if coordination_shift != 0:  # quant version
        output_data = linear_requantize(output_data, 1, coordination_shift, str(self.inputs[0].dtype))
    return output_data


def _scaler(i, ratio, mode, input_size, out_size):
    i = torch.tensor(i).double()
    if mode == 'half_pixel':
        return (i + 0.5) * (1. / ratio) - 0.5 if ratio is not None else (i + 0.5) * (input_size / out_size) - 0.5
    elif mode == 'align_corners':
        return i * 0. if out_size == 1 else (i * ((input_size - 1) / (out_size - 1)))
    elif mode == 'pytorch_half_pixel':
        if out_size > 1:
            return (i + 0.5) * (1. / ratio) - 0.5 if ratio is not None else (i + 0.5) * (input_size / out_size) - 0.5
        else:
            return i * 0.
    elif mode == 'tf_half_pixel_for_nn':
        return (i + 0.5) * (1. / ratio) if ratio is not None else (i + 0.5) * (input_size / out_size)
    else:  # asymmetric
        return i * (1. / ratio) if ratio is not None else i * (input_size / out_size)


# float version
def accelerate_resize_bilinear(self, in_data, output_shape):
    in_data = numpy_to_torch(in_data, self.inputs[0].dtype.np).float()
    mode = self.params['mode'].lower()
    coordination_shift = self.params.get('coordination_shift', 0)
    batch, out_h, out_w, out_c = output_shape
    in_h, in_w = in_data.shape[1:3]
    ratio_x = self.params.get('ratio_x', None)
    ratio_y = self.params.get('ratio_y', None)
    scale_num = 2 ** coordination_shift
    quantized = False if coordination_shift == 0 else True
    y = _scaler(torch.arange(out_h, device=in_data.device), ratio_y, mode, in_h, out_h)
    x = _scaler(torch.arange(out_w, device=in_data.device), ratio_x, mode, in_w, out_w)

    y_floor = torch.floor(y).int()
    y_ceil = torch.ceil(y).int()
    x_floor = torch.floor(x).int()
    x_ceil = torch.ceil(x).int()

    y0 = torch.maximum(y_floor, torch.tensor(0, device=in_data.device))
    y1 = torch.minimum(y_ceil, torch.tensor(in_h - 1, device=in_data.device))
    x0 = torch.maximum(x_floor, torch.tensor(0, device=in_data.device))
    x1 = torch.minimum(x_ceil, torch.tensor(in_w - 1, device=in_data.device))

    y0_x0 = torch.cartesian_prod(y0, x0).long()
    y0_x1 = torch.cartesian_prod(y0, x1).long()
    y1_x0 = torch.cartesian_prod(y1, x0).long()
    y1_x1 = torch.cartesian_prod(y1, x1).long()

    f00 = in_data[:, y0_x0[:, 0], y0_x0[:, 1], :]
    f01 = in_data[:, y0_x1[:, 0], y0_x1[:, 1], :]
    f10 = in_data[:, y1_x0[:, 0], y1_x0[:, 1], :]
    f11 = in_data[:, y1_x1[:, 0], y1_x1[:, 1], :]

    y0_lerp = y - y0
    x0_lerp = x - x0
    y1_lerp = 1 - y0_lerp
    x1_lerp = 1 - x0_lerp

    y0l_x0l = torch.cartesian_prod(y0_lerp, x0_lerp)
    y0l_x1l = torch.cartesian_prod(y0_lerp, x1_lerp)
    y1l_x0l = torch.cartesian_prod(y1_lerp, x0_lerp)
    y1l_x1l = torch.cartesian_prod(y1_lerp, x1_lerp)

    Q00 = y1l_x1l[:, 0] * y1l_x1l[:, 1] * scale_num
    Q10 = y0l_x1l[:, 0] * y0l_x1l[:, 1] * scale_num
    Q01 = y1l_x0l[:, 0] * y1l_x0l[:, 1] * scale_num
    Q11 = y0l_x0l[:, 0] * y0l_x0l[:, 1] * scale_num

    if quantized:
        Q00 = torch.floor(Q00 + 0.5)
        Q10 = torch.floor(Q10 + 0.5)
        Q01 = torch.floor(Q01 + 0.5)
        Q11 = torch.floor(Q11 + 0.5)

    Q00 = Q00.reshape([-1, 1]).repeat(batch, 1, out_c).float()
    Q01 = Q01.reshape([-1, 1]).repeat(batch, 1, out_c).float()
    Q10 = Q10.reshape([-1, 1]).repeat(batch, 1, out_c).float()
    Q11 = Q11.reshape([-1, 1]).repeat(batch, 1, out_c).float()

    # bilinear_out = (f00 * Q00 + f01 * Q01 + f10 * Q10 + f11 * Q11)
    bilinear_out = f00 * Q00
    bilinear_out = pyfma.fma(f01, Q01, bilinear_out)
    bilinear_out = pyfma.fma(f10, Q10, bilinear_out)
    bilinear_out = pyfma.fma(f11, Q11, bilinear_out)
    bilinear_out = torch.from_numpy(bilinear_out)

    bilinear_out = torch.round(bilinear_out) if quantized else bilinear_out
    outt = bilinear_out.reshape(*output_shape)
    return outt


def nearest_resize(self, input_data, output_shape):
    input_data = numpy_to_torch(input_data, self.inputs[0].dtype.np).float()
    mode = self.params['mode'].lower()
    nearest_mode = self.params.get('nearest_mode', 'round_prefer_floor').lower()
    batch, out_h, out_w, out_c = output_shape
    in_h, in_w = input_data.shape[1:3]
    ratio_x = self.params.get('ratio_x', None)
    ratio_y = self.params.get('ratio_y', None)

    def get_pixel(nearest_mode, original_pixel, down_sample):
        if nearest_mode == "round_prefer_floor":
            mask = torch.eq(original_pixel, torch.trunc(original_pixel) + 0.5)
            return torch.where(mask, torch.floor(original_pixel).int(), torch.round(original_pixel).int())
        elif nearest_mode == "floor":
            return torch.floor(original_pixel).int()
        elif nearest_mode == "ceil":
            return torch.ceil(original_pixel).int()
        elif nearest_mode == "simple":
            return torch.ceil(original_pixel).int() if down_sample else original_pixel.int()
        else:  # "round_prefer_ceil"
            return torch.floor(original_pixel + 0.5).int()

    y = _scaler(torch.arange(out_h, device=input_data.device), ratio_y, mode, in_h, out_h)
    x = _scaler(torch.arange(out_w, device=input_data.device), ratio_x, mode, in_w, out_w)
    map_inp_h = get_pixel(nearest_mode, y, out_h / in_h < 1)
    map_inp_w = get_pixel(nearest_mode, x, out_w / in_w < 1)
    map_inp_h = torch.maximum(map_inp_h, torch.tensor(0, device=input_data.device))
    map_inp_h = torch.minimum(map_inp_h, torch.tensor(in_h - 1, device=input_data.device))
    map_inp_w = torch.maximum(map_inp_w, torch.tensor(0, device=input_data.device))
    map_inp_w = torch.minimum(map_inp_w, torch.tensor(in_w - 1, device=input_data.device))

    input_index = torch.cartesian_prod(map_inp_h, map_inp_w).long()
    out_t = input_data[:, input_index[:, 0], input_index[:, 1], :].reshape(output_shape)
    return out_t


def compute_weight_coefficients(input_size, output_size, rscale, mode, exclude_outside, dev):
    def filter(x):
        if x < 0.0:
            x = -x
        if x < 1.0:
            return 1.0 - x
        return 0.0
    import math
    scale = 1/rscale
    support = scale if (scale >= 1.0) else 1.0
    windowsize = int(math.ceil(support)) * 2 + 1
    inv_scale = 1/(scale) if scale >= 1.0 else 1.0
    bound_w_min = []
    bound_w_max = []
    scale_buffer = torch.zeros([windowsize * output_size], device=dev)
    for coord in range(output_size):
        center = 0.5 + _scaler(coord, rscale, mode, input_size, output_size)
        total_weight = 0.0
        fmin = math.floor(center - support + 0.5)
        fmax = math.floor(center + support + 0.5)
        xmin_real = int(fmin)
        xmax_real = int(fmax)
        xmin_cut = max(0, xmin_real)
        xmax_cut = min(input_size, xmax_real)
        bound_w_min.append(xmin_cut)
        bound_w_max.append(xmax_cut)
        xmin = xmin_cut if exclude_outside else xmin_real
        xmax = xmax_cut if exclude_outside else xmax_real
        terp = xmax - xmin
        offset = coord*windowsize
        for x in range(terp):
            weight = filter((x + xmin - center + 0.5) * inv_scale)
            scale_buffer[offset + x] = weight
            total_weight += weight
        xmax -= xmin
        if not exclude_outside:
            neg_xsize = -xmin if xmin < 0 else 0
            for i in range(neg_xsize):
                scale_buffer[offset + neg_xsize] += scale_buffer[offset + i]
            bound_xsize = (xmax + xmin - input_size) if (xmax + xmin > input_size) else 0
            for x in range(xmax - bound_xsize, xmax):
                scale_buffer[offset+xmax - bound_xsize - 1] += scale_buffer[offset + x]
            x = 0
            while (neg_xsize | bound_xsize) > 0 and (x < xmax_cut - xmin_cut):
                scale_buffer[offset + x] = scale_buffer[offset + x + neg_xsize]
                x += 1
        total_weight_inv = 1.0 if total_weight == 0.0 else 1.0 / total_weight
        for x in range(xmax_cut - xmin_cut):
            scale_buffer[offset + x] *= total_weight_inv
    return scale_buffer, torch.tensor(bound_w_min, device=dev), torch.tensor(bound_w_max, device=dev)


@register_gen_weight(OpType.Resize)
def gen_random_weight(self):
    if 'weight_coefficients_x' in self.constants:
        in_shape = list(self.inputs[0].shape)
        out_shape = list(self.outputs[0].shape)
        dev = self.inputs[0].betensor.device
        mode = self.params['mode']
        exclude_outside = self.params['exclude_outside']
        coordination_shift = self.params.get('interp_shift_value', 0)
        input_height, input_width = in_shape[1:3]
        batch_size, output_height, output_width, channel = out_shape
        ratio_x = self.params.get('ratio_x', output_width / input_width)
        ratio_y = self.params.get('ratio_y', output_height / input_height)
        weight_coefficients_x, bound_x_min, bound_x_max = compute_weight_coefficients(
            input_width, output_width, ratio_x, mode, exclude_outside, dev)
        weight_coefficients_y, bound_y_min, bound_y_max = compute_weight_coefficients(
            input_height, output_height, ratio_y, mode, exclude_outside, dev)

        if self.constants['weight_coefficients_x'].dtype.is_float:
            self.constants['weight_coefficients_x'].betensor = weight_coefficients_x.to(
                self.constants['weight_coefficients_x'].dtype.torch)
            self.constants['weight_coefficients_y'].betensor = weight_coefficients_y.to(
                self.constants['weight_coefficients_x'].dtype.torch)
        else:
            self.constants['weight_coefficients_x'].betensor = (
                weight_coefficients_x * 255).cpu().numpy().astype('uint16')
            self.constants['weight_coefficients_y'].betensor = (
                weight_coefficients_y * 255).cpu().numpy().astype('uint16')
        self.constants['bound_x_min'].betensor = bound_x_min.cpu().numpy().astype('uint16')
        self.constants['bound_x_max'].betensor = bound_x_max.cpu().numpy().astype('uint16')
        self.constants['bound_y_min'].betensor = bound_y_min.cpu().numpy().astype('uint16')
        self.constants['bound_y_max'].betensor = bound_y_max.cpu().numpy().astype('uint16')


@check_fp_data
@register_op_metric(OpType.Resize)
def metric(self, outputs, index=None, real_parent_node_type=None):
    if self.inputs[0].dtype.is_int:
        return _OP_METRICS["_all"](self, outputs)
    else:
        if not self.attrs.get("is_aiff", False):
            return _OP_METRICS["_all"](self)  # tpc has no mismatch
        if index is None:
            groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
        else:
            groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]
        # rel_diffs = relative_diff(outputs, groundtruths)
        # ulp_diffs = ulp_diff(outputs, groundtruths)
        cos_dists = cos_distance(outputs, groundtruths)
        return cos_dists[0] > 0.999
