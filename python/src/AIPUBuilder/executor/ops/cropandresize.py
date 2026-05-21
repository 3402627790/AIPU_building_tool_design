import torch
from .gt_util import *
from ..engine import register_gt, register_gen_input
from AIPUBuilder.core import OpType
from AIPUBuilder.logger import WARN
from .gt_util.layout_transfer import check_out_datalayout,check_in_datalayout
from AIPUBuilder.core import  Target


@register_gt(OpType.CropAndResize)
def CropAndResize(self, *args, **kwargs):
    target = Target().name
    target = target.lower()
    feature = self.inputs[0].betensor.cpu().numpy().astype(str(self.inputs[0].dtype))
    nor_box = self.inputs[1].betensor.cpu().numpy().astype(str(self.inputs[1].dtype))
    box_indices = self.inputs[2].betensor.cpu().numpy().astype(str(self.inputs[2].dtype))

    if 'is_aiff' in kwargs.keys() and kwargs['is_aiff'] == False:
        print('CropAndResize tpc gt...')
        output = crop_and_resize_tpc(self, feature, nor_box, box_indices)
        output = clamp(output, self.outputs[0].dtype)
    else:
        print('CropAndResize aiff gt...')
        feature_new, nor_box_new, box_indices_new = check_in_datalayout(self, [feature, nor_box, box_indices], ['NHWC','Flat', 'Flat'])
        feature_new = feature_new.numpy()
        dtype = self.inputs[0].dtype
        feature_new = feature_new.astype(str(dtype))
        output = crop_and_resize_aiff(self, feature_new, nor_box, box_indices, target)
        output = check_out_datalayout(self, [output], ['NHWC'])[0]

    self.outputs[0].betensor = output
    return [o.betensor for o in self.outputs]


def crop_and_resize_tpc(self, feature, nor_box, box_indices):
    crop_height, crop_width = self.params['crop_size']
    extrapolation_value = self.params['extrapolation_value']
    method = self.params['method']
    scale_value = self.params['scale_value']
    shift_value = self.params['shift_value']
    is_perf_mode = self.params['is_perf_mode']

    batch, fm_height, fm_width, fm_depth = feature.shape
    box_size = nor_box.shape[0]

    qextrapolation_value = int(extrapolation_value * 2 ** shift_value / scale_value)

    out = np.zeros([box_size, crop_height, crop_width, fm_depth], 'int32')

    for b in range(box_size):
        box_idx = box_indices[b]
        y1, x1, y2, x2 = nor_box[b]

        qheight_scale = (((y2 - y1) * (fm_height - 1) * 256) // (crop_height - 1)) >> 8 if crop_height > 1 else 0
        qwidth_scale = (((x2 - x1) * (fm_width - 1) * 256) // (crop_width - 1)) >> 8 if crop_width > 1 else 0

        top_y_index = 0
        bottom_y_index = 0
        y_lerp = 0
        for y in range(crop_height):
            in_y = y1 * (fm_height - 1) + y * qheight_scale
            if in_y < 0 or (in_y * scale_value >> shift_value) > (fm_height - 1):
                padded_v = np.full([1, 1, out.shape[2], out.shape[3]], qextrapolation_value)
                out[b, y, :, :] = padded_v
                continue

            if method.lower() == 'bilinear':
                top_y_index = (in_y * scale_value >> shift_value)
                bottom_y_index = top_y_index + 1
                y_lerp = in_y - top_y_index * 2 ** shift_value // scale_value
            else:  # method == 'nearest':
                in_y = (np.int64((in_y * scale_value + 2 ** (shift_value - 1))) >> shift_value)
                in_y = np.clip(in_y, 0, fm_height - 1)

            for x in range(crop_width):
                in_x = x1 * (fm_width - 1) + x * qwidth_scale
                if in_x < 0 or ((in_x * scale_value) >> shift_value) > (fm_width - 1):
                    padded_v = np.full([1, 1, 1, out.shape[3]], qextrapolation_value)
                    out[b, y, x, :] = padded_v
                    continue

                if method.lower() == 'bilinear':
                    left_x_index = (in_x * scale_value >> shift_value)
                    right_x_index = left_x_index + 1
                    x_lerp = in_x - left_x_index * 2 ** shift_value // scale_value

                    top_y_index = np.clip(top_y_index, 0, fm_height - 1)
                    bottom_y_index = np.clip(bottom_y_index, 0, fm_height - 1)
                    left_x_index = np.clip(left_x_index, 0, fm_width - 1)
                    right_x_index = np.clip(right_x_index, 0, fm_width - 1)

                    top_left = feature[box_idx, top_y_index, left_x_index, :].astype(np.int32)
                    top_right = feature[box_idx, top_y_index, right_x_index, :].astype(np.int32)
                    bottom_left = feature[box_idx, bottom_y_index, left_x_index, :].astype(np.int32)
                    bottom_right = feature[box_idx, bottom_y_index, right_x_index, :].astype(np.int32)

                    top = top_left + (((top_right - top_left) * x_lerp * scale_value) >> shift_value)
                    bottom = bottom_left + (((bottom_right - bottom_left) * x_lerp * scale_value) >> shift_value)
                    out[b, y, x, :] = (top + (((bottom - top) * y_lerp * scale_value) >> shift_value))
                else:  # method == 'nearest':
                    in_x = (np.int64((in_x * scale_value + 2 ** (shift_value - 1))) >> shift_value)
                    in_x = np.clip(in_x, 0, fm_width - 1)
                    out[b, y, x, :] = feature[box_idx, in_y, in_x, :]
    return out


def crop_and_resize_aiff(self, feature, nor_box, box_indices, target):


    is_float = self.inputs[0].dtype.is_float
    crop_height, crop_width = self.params['crop_size']
    extrapolation_value = self.params['extrapolation_value']
    method = self.params['method'].lower()
    scale_value = self.params['scale_value']
    shift_value = self.params['shift_value']

    mode = "align_corners"  # todo change mode
    align_corners = False
    half_pixel = False

    if mode.lower() == 'half_pixel':
        half_pixel = True
    elif mode.lower() == 'align_corners':
        align_corners = True

    _, fm_height, fm_width, fm_depth = feature.shape
    box_size = nor_box.shape[0]

    # qextrapolation_value = int(self.extrapolation_value * 2 ** self.shift_value / self.scale_value)

    out = np.zeros([box_size, crop_height, crop_width, fm_depth], feature.dtype)

    for b in range(box_size):
        box_idx = box_indices[b]
        y1, x1, y2, x2 = nor_box[b]

        y_min = (y1 * (fm_height - 1) * scale_value) >> shift_value
        y_max = (y2 * (fm_height - 1) * scale_value) >> shift_value
        x_min = (x1 * (fm_width - 1) * scale_value) >> shift_value
        x_max = (x2 * (fm_width - 1) * scale_value) >> shift_value

        if x_min >= fm_width or x_max >= fm_width or y_min >= fm_height or y_max >= fm_height or x_min == x_max or y_min == y_max:
            print("aipu assert out of feature map, ARITHMETIC_ERROR 0x260480: box_index= {}".format(b))

        data = feature[box_idx][y_min:y_max + 1, x_min:x_max + 1, :]
        if "x3" in target:
            out_per_box = resize_with_method_itp_x3(self, data, [crop_height, crop_width], mode, method, is_float)
        else:
            out_per_box = resize_with_method_itp(data, [crop_height, crop_width], align_corners, half_pixel, method,
                                             "aipu assert out of interval, OUT_OF_AIFF_SPEC 0x260381: box_index= {}".format(
                                                 b))
        out[b] = out_per_box
    return out


def resize_with_method_itp(data: np.ndarray, out, align_corners=False, half_pixel=False, method='nearest', messg=None,
                           case_path=None):
    '''
        NOTE: this func will be use  CropAndResize op, if you modify it, make op-owner reviewed.
        data: shpae [h,w,c], dtype only support int8,uint8,int16
        out : [h_out,w_out]
        res : output result, shape [h_out,w_out,c], dtype is the same is data.dtype
    '''

    def set_linear_interval(ix, ox, align_corners):
        """
            NOTE: use "round" in this function to get 24bit fractional part of interval
            4bit integral part, 24bit fractional part. Zero means interval = 16???
        """
        max_interval = 16
        interval = 0
        if align_corners and ox > 1:
            # assert ((ix - 1) // (ox - 1)
            #         ) <= max_interval, "ix//ox is too large: {}".format(ix / ox)

            interval = ((ix - 1) << 25) // (ox - 1)
            interval = (interval >> 1) + (interval & 0x1)
        else:
            # assert (ix // ox) <= max_interval, "ix//ox is too large: {}".format(ix / ox)
            interval = (ix << 25) // ox
            interval = (interval >> 1) + (interval & 0x1)

        return interval

    def set_linear_offset(ix, ox, half_pixel):
        """
            NOTE: use "round" in this function to get 24bit fractional part of offset;
            not consider offset > 1 and sign bit as in AIFF, for python GT can use normal offset
        """
        offset = 0
        if half_pixel:
            offset = ((ix << 24) // ox) - (1 << 24)
            offset = (offset >> 1) + (offset & 0x1)

        return offset

    assert (str(data.dtype) in ["int8", "uint8", "in16"])
    input_h, input_w, output_h, output_w = data.shape[0], data.shape[1], out[0], out[1]
    interval_h = set_linear_interval(input_h, output_h, align_corners)
    interval_w = set_linear_interval(input_w, output_w, align_corners)

    # if interval_w >(16<<24) or interval_w<(1<<18):
    #     aipu_assert_false(messg, case_path)
    #
    # if interval_h > (16<<24) or interval_h<(1<<17):
    #     aipu_assert_false(messg, case_path)
    #     # pass

    offset_h = set_linear_offset(input_h, output_h, half_pixel)
    offset_w = set_linear_offset(input_w, output_w, half_pixel)

    res = []
    lut_data = np.int32(data)

    h, w, c = data.shape
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
                if data.dtype.name in ["int8"]:
                    w0 = (((lut_data[y0, x1, ...] << 8) - (lut_data[y0, x0, ...] << 8)) * delta_x + (
                        lut_data[y0, x0, ...] << 16) >> 8)
                    w1 = (((lut_data[y1, x1, ...] << 8) - (lut_data[y1, x0, ...] << 8)) * delta_x + (
                        lut_data[y1, x0, ...] << 16) >> 8)
                    result = np.float32((w1 - w0) * delta_y + (w0 << 8) >> 8)
                    result = np.round(result * (0.5 ** 8))
                elif data.dtype.name in ["uint8"]:
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
    res = np.array(res)
    res = res.reshape([output_h, output_w, c])
    clamp(res, data.dtype.name)
    # res=res.astype(TYPE_TO_DTYPE[data.dtype.name])

    return res


def set_linear_interval(ix, ox, align_corners):
    max_interval = 16
    interval = 0
    if align_corners and ox > 1:
        interval = ((ix - 1) << 25) // (ox - 1)
        interval = (interval >> 1) + (interval & 0x1)
    else:
        interval = (ix << 25) // ox
        interval = (interval >> 1) + (interval & 0x1)
    return interval


def get_real_input(ix, ox, ratio_name, params):
    """
        support resize tiling
    """
    _ratio = 0.
    mode = "align_corners"
    if ratio_name in params and params[ratio_name] is not None:
        _ratio = params[ratio_name]
    if "mode" in params and params["mode"] is not None:
        mode = params["mode"].lower()
    if mode != "align_corners" and _ratio != 0. and np.floor(_ratio * ix) != int(ox):
        assert np.floor(_ratio * (ix - 1)) == int(ox)
        return ix - 1
    return ix



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
    offset_i = int(round(offset * (1<<24)))

    return offset_i


def resize_with_method_itp_x3(self, data, out, mode, method, is_float=False):

    def set_linear_interval(ix, ox, mode, ratio):
        interval = ix / ox
        if mode == "align_corners" and ix >= 2 and ox >= 2:
            interval = (ix - 1) / (ox - 1)
        elif mode == "half_pixel_symmetric" and ratio > 0:
            interval = 1 / ratio
        assert interval <= 16, "interval is too large: {}".format(interval)
        assert interval >= 1.0/64, "interval is too small: {}".format(interval)
        interval_i = int(round(interval * (1<<24)))
        return interval_i

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
    lut_data = data
    if is_float:
        lut_data = np.float32(data)
    else:
        lut_data = np.int32(data)
    inputs_shape = data.shape
    outputs_type = str(self.outputs[0].dtype)
    h, w, c = inputs_shape
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
                    w0 = (lut_data[y0, x1, ...] - lut_data[y0, x0, ...]) * delta_x * (0.5 ** 16) + lut_data[y0, x0, ...]
                    w0 = np.float32(np.float16(w0))
                    w1 = (lut_data[y1, x1, ...] - lut_data[y1, x0, ...]) * delta_x * (0.5 ** 16) + lut_data[y1, x0, ...]
                    w1 = np.float32(np.float16(w1))
                    result = np.float32((w1 - w0) * delta_y * (0.5 ** 16) + w0)
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

    return np.array(res).reshape((output_h, output_w, -1))

@register_gen_input(OpType.CropAndResize)
def gen_input_data(self):
    import random
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_shape2 = list(self.inputs[2].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)
    in_type2 = str(self.inputs[2].dtype)

    prob_good = 1.0
    prob_airhmetic_error = 0.0
    prob_aiff_spec_error = 1 - prob_good - prob_airhmetic_error
    is_good_case = False
    is_arithmetic_error_case = False
    is_aiff_spec_error_case = False
    case_rand = float(np.random.uniform(0.0, 1.0, 1))
    if case_rand < prob_good:
        is_good_case = True
    elif case_rand < prob_good + prob_airhmetic_error:
        is_arithmetic_error_case = True
    else:
        is_aiff_spec_error_case = True
    box_num = in_shape1[0]
    test_arithmetic_error = [False] * box_num
    test_out_of_aiff_spec = [False] * box_num

    if is_good_case:
        pass
    elif is_arithmetic_error_case:
        box_index_test = int(np.random.randint(0, box_num, 1))
        test_arithmetic_error[box_index_test] = True
    elif is_aiff_spec_error_case:
        box_index_test = int(np.random.randint(0, box_num, 1))
        test_out_of_aiff_spec[box_index_test] = True
    else:
        assert False

    featuremap = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)
    out_shape = self.outputs[0].shape
    out_h = out_shape[-3]
    out_w = out_shape[-2]

    batch = in_shape0[0]
    y_max = max(2, in_shape0[1])
    x_max = max(2, in_shape0[2])

    input_box = []
    for i in range(box_num):
        counter = 0
        while True:
            counter += 1
            if test_arithmetic_error[i]:
                y = random.sample(range(0, 2 * y_max), 2)
                x = random.sample(range(0, 2 * x_max), 2)
            else:
                y = random.sample(range(0, y_max), 2)
                x = random.sample(range(0, x_max), 2)
            min_y = min(y)
            min_x = min(x)
            max_y = max(y)
            max_x = max(x)
            shift_value = int(self.params['shift_value'])
            scale_value = int(self.params['scale_value'])
            min_y = int(min(min_y * (2 ** shift_value) / (scale_value) / (y_max - 1), 2 ** 16 - 1))
            max_y = int(min(max_y * (2 ** shift_value) / (scale_value) / (y_max - 1), 2 ** 16 - 1))
            min_x = int(min(min_x * (2 ** shift_value) / (scale_value) / (x_max - 1), 2 ** 16 - 1))
            max_x = int(min(max_x * (2 ** shift_value) / (scale_value) / (x_max - 1), 2 ** 16 - 1))

            _min_y = (min_y * scale_value * (y_max - 1)) >> shift_value
            _max_y = (max_y * scale_value * (y_max - 1)) >> shift_value
            _min_x = (min_x * scale_value * (x_max - 1)) >> shift_value
            _max_x = (max_x * scale_value * (x_max - 1)) >> shift_value

            align_corners = True
            interval_w = set_linear_interval((_max_x - _min_x + 1), out_w, align_corners)
            interval_h = set_linear_interval((_max_y - _min_y + 1), out_h, align_corners)

            is_arithmetic_error = False
            is_out_of_aiff_spec = False
            if (_min_y >= y_max or _max_y >= y_max or _min_x >= x_max or _max_x >= x_max or _min_x == _max_x or _min_y == _max_y):
                is_arithmetic_error = True
            if interval_w > (16 << 24) or interval_w < (1 << 18) or interval_h > (16 << 24) or interval_h < (
                    1 << 17):
                if interval_h != 0 and interval_w != 0:
                    is_out_of_aiff_spec = True
            if (is_out_of_aiff_spec == test_out_of_aiff_spec[i]) and (
                    is_arithmetic_error == test_arithmetic_error[i]):
                break
            if counter >= 100:
                WARN("can not generate such a case")
                break

        bbox = [min_y, min_x, max_y, max_x]
        input_box.append(bbox)
    input_box = np.asarray(input_box, dtype=np.uint16)
    box_indices = np.random.randint(0, batch, size=in_shape2, dtype=in_type2)

    input_data = [featuremap, input_box, box_indices]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    self.inputs[2].betensor = input_data[2]
    return input_data
