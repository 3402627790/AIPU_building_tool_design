from ..engine import register_gt, register_gen_input
from AIPUBuilder.core import OpType
import numpy as np
import torch
from AIPUBuilder.logger import FATAL
from .gt_util.quant_utils import linear_requantize, clamp


@register_gt(OpType.PyramidROIAlign)
def PyramidROIAlign(self, *args, **kwargs):
    nor_box = self.inputs[0].betensor.float()
    feature_map0 = self.inputs[1].betensor.float()
    feature_map1 = self.inputs[2].betensor.float()
    feature_map2 = self.inputs[3].betensor.float()
    feature_map3 = self.inputs[4].betensor.float()
    feature_list = [feature_map0, feature_map1, feature_map2, feature_map3]

    proposal_normalized = self.params.get('proposal_normalized', False)

    if self.inputs[1].dtype.is_float:
        if proposal_normalized:
            output = local_roi_align_float(self, nor_box, feature_list)
        else:
            output = torch_roi_align_float(self, nor_box, feature_list)
    else:
        if proposal_normalized:
            output = local_roi_align(self, nor_box, feature_list)
        else:
            output = torch_roi_align(self, nor_box, feature_list)
        output = clamp(output, self.outputs[0].dtype)
    self.outputs[0].betensor = output
    return [self.outputs[i].betensor for i in range(len(self.outputs))]


def torch_roi_align(node, nor_box, feature_list):
    resize_height_ = node.outputs[0].shape[1]
    resize_width_ = node.outputs[0].shape[2]
    channel_ = node.outputs[0].shape[3]

    h_sample_ratio, w_sample_ratio = node.params['sample']
    out_height, out_width = node.params['resize_height'], node.params['resize_width']
    spatial_list = node.params['spatial_scale_value']
    coordinate_transformation_mode = node.params['coordinate_transformation_mode'].lower()
    _SUPPORT_COORDINATE_MODE = ['half_pixel', 'output_half_pixel']
    if coordinate_transformation_mode not in _SUPPORT_COORDINATE_MODE:
        FATAL(f"{node}, currently coordinate_transformation_mode only support {_SUPPORT_COORDINATE_MODE}, please check! ")
    is_half_pixel = True if coordinate_transformation_mode == 'half_pixel' else False

    y0, x0, y1, x1 = nor_box[0, :, 0], nor_box[0, :, 1], nor_box[0, :, 2], nor_box[0, :, 3]
    norbox_h = y1 - y0
    norbox_w = x1 - x0
    area = norbox_h * norbox_w
    feature_num = len(feature_list)

    target_lvls = torch.zeros_like(area, device=area.device)
    do_scales = node.params['scale_value']
    do_shifts = node.params['shift_value']
    input_scales = node.params["input_scale"]
    input_shifts = node.params["input_shift"]
    spatial_shift = node.params["spatial_shift"]
    L0, L1, L2 = node.params['levels']
    output = torch.zeros([nor_box.shape[1], resize_height_, resize_width_, channel_],
                         device=node.inputs[1].betensor.device)

    half_pixel_offset = 2 ** (spatial_shift - 1) if is_half_pixel else 0
    L0_mask = area < L0
    L1_mask = torch.bitwise_and(area >= L0, area < L1)
    L2_mask = torch.bitwise_and(area >= L1, area < L2)
    L3_mask = area >= L2
    target_lvls[L0_mask] = 0
    target_lvls[L1_mask] = 1
    target_lvls[L2_mask] = 2
    target_lvls[L3_mask] = 3
    for level in range(feature_num):
        feature_map = feature_list[level] + node.inputs[level + 1].quantization.offset
        dev = feature_map.device
        fm_height, fm_width = feature_map.shape[1:3]
        idx_in_level = torch.where(target_lvls == level)[0]
        roi_box = nor_box[0, idx_in_level, :] + node.inputs[0].quantization.offset
        current_box_num = roi_box.shape[0]
        nor_output = torch.zeros([current_box_num, resize_height_, resize_width_, channel_],
                                 device=node.inputs[1].betensor.device)
        clip_dtype = 'uint16'
        w_roi_start = linear_requantize(roi_box[:, 1], input_scales[level],
                                        input_shifts[level], clip_dtype).float() - half_pixel_offset  # [730,4]
        w_roi_end = linear_requantize(roi_box[:, 3], input_scales[level],
                                      input_shifts[level], clip_dtype).float() - half_pixel_offset
        h_roi_start = linear_requantize(roi_box[:, 0], input_scales[level],
                                        input_shifts[level], clip_dtype).float() - half_pixel_offset
        h_roi_end = linear_requantize(roi_box[:, 2], input_scales[level],
                                      input_shifts[level], clip_dtype).float() - half_pixel_offset

        roi_width = w_roi_end - w_roi_start
        roi_height = h_roi_end - h_roi_start
        if not is_half_pixel:
            # roi_width = torch.maximum((roi_width), torch.tensor(1.0, device=dev))
            # roi_height = torch.maximum((roi_height), torch.tensor(1.0, device=dev))
            roi_width = torch.maximum(roi_width, torch.tensor(2 ** spatial_shift, device=dev))
            roi_height = torch.maximum(roi_height, torch.tensor(2 ** spatial_shift, device=dev))

        w_sample_ratios = torch.ones([current_box_num], device=roi_box.device).int()
        if w_sample_ratio <= 0:
            w_sample_div = torch.div(roi_width, out_width * (2 ** spatial_shift), rounding_mode='trunc').int()
            w_sample_mod = roi_width - w_sample_div * (out_width * (2 ** spatial_shift))
            gt0_mask = w_sample_mod > 0
            lt0_mask = w_sample_mod <= 0
            w_sample_ratios[gt0_mask] = w_sample_div[gt0_mask] + 1
            w_sample_ratios[lt0_mask] = w_sample_div[lt0_mask]
        else:
            w_sample_ratios = w_sample_ratios * w_sample_ratio

        h_sample_ratios = torch.ones([current_box_num], device=roi_box.device).int()
        if h_sample_ratio <= 0:
            h_sample_div = torch.div(roi_height, out_height * (2 ** spatial_shift), rounding_mode='trunc').int()
            h_sample_mod = roi_height - h_sample_div * (out_height * (2 ** spatial_shift))
            gt0_mask = h_sample_mod > 0
            lt0_mask = h_sample_mod <= 0
            h_sample_ratios[gt0_mask] = h_sample_div[gt0_mask] + 1
            h_sample_ratios[lt0_mask] = h_sample_div[lt0_mask]
        else:
            h_sample_ratios = h_sample_ratios * h_sample_ratio

        w_step_size = torch.div(roi_width, out_width, rounding_mode='trunc')  # roi_width // out_width
        h_step_size = torch.div(roi_height, out_height, rounding_mode='trunc')  # roi_height // out_height
        # w_step_size = (roi_width * out_w_scale) >> out_w_shift
        # h_step_size = (roi_height * out_h_scale) >> out_h_shift

        # if sampling_ratio=0, use adaptive value of ceil(roi_width/out_width), same for height
        # w_sampling_ratio = w_sample_ratio if w_sample_ratio > 0 else int(math.ceil(w_step_size))
        # h_sampling_ratio = h_sample_ratio if h_sample_ratio > 0 else int(math.ceil(h_step_size))
        w_bin_size = torch.div(w_step_size, 2 * w_sample_ratios,
                               rounding_mode='trunc')  # w_step_size // (2*w_sample_ratio)  #[730]
        h_bin_size = torch.div(h_step_size, 2 * h_sample_ratios,
                               rounding_mode='trunc')  # h_step_size // (2*h_sample_ratio)
        # w_bin_size = (w_step_size.int() * sample_w_scale) >> sample_w_shift
        # h_bin_size = (h_step_size.int() * sample_h_scale) >> sample_h_shift

        for b in range(current_box_num):
            if roi_width[b] == 0 or roi_height[b] == 0:
                continue
            for i in range(out_height):
                h_start = (h_step_size[b] * i + h_roi_start[b]).long()
                for j in range(out_width):
                    w_start = (w_step_size[b] * j + w_roi_start[b]).long()
                    depth_output = torch.zeros([channel_], device=feature_map.device)
                    for y_ind in range(h_sample_ratios[b]):
                        # y = h_start[b] + h_bin_size / 2 + h_bin_size * y_ind
                        y = (h_start + h_bin_size[b] * (2 * y_ind + 1)).long()
                        for x_ind in range(w_sample_ratios[b]):
                            # x = w_start[b] + w_bin_size / 2 + w_bin_size * x_ind
                            x = (w_start + w_bin_size[b] * (2 * x_ind + 1)).long()
                            y_low = y >> spatial_shift
                            y_high = y_low + 1
                            dy1 = y - (y_low << spatial_shift)
                            x_low = x >> spatial_shift
                            x_high = x_low + 1
                            dx1 = x - (x_low << spatial_shift)

                            if y_low < -1 or y_low > fm_height or x_low < -1 or x_low > fm_width:
                                depth_output += 0
                            else:
                                y_low = min(max(y_low, 0), fm_height - 1)
                                x_low = min(max(x_low, 0), fm_width - 1)
                                x_high = min(max(x_high, 0), fm_width - 1)
                                y_high = min(max(y_high, 0), fm_height - 1)

                                dx2 = 2 ** spatial_shift - dx1
                                dy2 = 2 ** spatial_shift - dy1
                                ws = [dx2 * dy2, dx1 * dy2, dx2 * dy1, dx1 * dy1]
                                ws = [wss >> spatial_shift for wss in ws]
                                depth_output += (feature_map[0, y_low, x_low, :] * ws[0] +
                                                 feature_map[0, y_low, x_high, :] * ws[1] +
                                                 feature_map[0, y_high, x_low, :] * ws[2] +
                                                 feature_map[0, y_high, x_high, :] * ws[3])
                    # nor_output[b, i, j, :] = depth_output / (w_sampling_ratio * h_sampling_ratio)
                    if w_sample_ratio > 0 and h_sample_ratio > 0:
                        nor_output[b, i, j, :] = linear_requantize(depth_output, do_scales[level],
                                                                   do_shifts[level] + spatial_shift,
                                                                   str(node.outputs[0].dtype),
                                                                   node.outputs[0].quantization.offset)
                    else:
                        nor_output[b, i, j, :] = linear_requantize(
                            depth_output * do_scales[level] // (h_sample_ratios[b] * w_sample_ratios[b]), 1,
                            do_shifts[level] + spatial_shift, str(node.outputs[0].dtype), node.outputs[0].quantization.offset)
        output[idx_in_level, ...] = nor_output
    return output


def local_roi_align(node, nor_box, feature_list):
    L1_Q = 102703631
    L2_Q = 25675908
    L2_Q_h = 102703631
    L3_Q = 6418977
    L3_Q_h = 25675908
    L4_Q = 6418977

    out = node.outputs[0].betensor
    dev = node.inputs[0].betensor.device
    nor_box = nor_box + node.inputs[0].quantization.offset
    feature_maps = []
    for i, inp in enumerate(node.inputs):
        # (torch.tensor(0) if not self.quantized else torch.tensor(inp.zerop))
        feature_maps.append(inp.betensor.float() + inp.quantization.offset)
    resize_height_ = node.outputs[0].shape[1]
    resize_width_ = node.outputs[0].shape[2]
    channel_ = node.outputs[0].shape[3]

    y0, x0, y1, x1 = nor_box[0, :, 0], nor_box[0, :, 1], nor_box[0, :, 2], nor_box[0, :, 3]
    h = y1 - y0
    w = x1 - x0
    # Use shape of first image. Images in a batch must have the same size.
    # Equation 1 in the Feature Pyramid Networks paper. Account for
    # the fact that our coordinates are normalized here.
    # e.g. a 224x224 ROI (in pixels) maps to P4
    qvalue = node.params['box_input_qvalue']
    qmax = (1 << qvalue)-1
    # #quantized forward
    # image_area = (round((10 - 7.807) * 512) ) # 10 is log2(1024*1024), 7.807 is log2(224)
    # roi_level =torch.zeros((1000),dtype=torch.int16)
    # for i in range(nor_box.shape[1]):
    #     area = (h[i].int()*w[i].int())>>15
    #     area = area&0x7fff #to ensure area is postive
    #     level = ((image_area  + (mylog2(area)) // 2)+256)>>9
    #     level = min(5, max(
    #         2, 4 + (round(level))))
    #     roi_level[i]=level

    roi_level5 = torch.ones_like(h)*5
    roi_level4 = torch.ones_like(h)*4
    roi_level3 = torch.ones_like(h)*3
    roi_level2 = torch.ones_like(h)*2
    area = h.int()*w.int()
    area = area & 0x7fffFFFF
    f5 = (area > L1_Q)*roi_level5
    f4 = ((area < L2_Q_h) & (area > L2_Q))*roi_level4
    f3 = ((area < L3_Q_h) & (area > L3_Q))*roi_level3
    f2 = (area < L4_Q) * roi_level2
    roi_level = (f2+f3+f4+f5).int()

    resize_feature = torch.zeros((nor_box.shape[1], resize_height_, resize_width_, channel_), device=dev)
    # for batchidx in range(feature.shape[0]):
    batch_idx = 0
    for boxidx in range(nor_box.shape[1]):
        feature = feature_maps[roi_level[boxidx] - 1]

        image_width = feature.shape[1]
        image_height = feature.shape[2]
        channel = feature.shape[3]

        y0_q = nor_box[batch_idx, boxidx, 0].int()
        x0_q = nor_box[batch_idx, boxidx, 1].int()
        y1_q = nor_box[batch_idx, boxidx, 2].int()
        x1_q = nor_box[batch_idx, boxidx, 3].int()
        height_scale_q = (torch.div((y1_q - y0_q) * (image_height - 1) * 256,
                                    (resize_height_ - 1), rounding_mode='trunc')) >> 8  # Q15*Q0
        width_scale_q = (torch.div((x1_q - x0_q) * (image_width - 1) * 256,
                                   (resize_width_ - 1), rounding_mode='trunc')) >> 8

        x_q = (image_width-1) * x0_q + torch.arange(0, resize_width_, device=dev) * width_scale_q
        y_q = (image_height-1) * y0_q + torch.arange(0, resize_height_, device=dev) * height_scale_q

        yy_q = torch.clamp(y_q, 0,  (image_width - 1)*qmax)
        xx_q = torch.clamp(x_q, 0, (image_height - 1)*qmax)

        top_y_index_q = yy_q >> qvalue
        bottom_y_index_q = (yy_q+qmax) >> qvalue
        y_lerp_q = (yy_q & 0x7fff).reshape(resize_height_, 1).repeat(1, channel)
        # y_lerp = torch.repeat(y_lerp, channel).reshape(resize_height_, channel)
        left_x_index_q = xx_q >> qvalue
        right_x_index_q = (xx_q+qmax) >> qvalue
        x_lerp_q = (xx_q & 0x7fff).reshape(resize_width_, 1).repeat(1, channel)

        for idxh in range(resize_height_):
            for idxw in range(resize_width_):
                # get 4 point
                top_left = feature[0, top_y_index_q[idxh], left_x_index_q[idxw], :]  # Q12
                top_right = feature[0, top_y_index_q[idxh], right_x_index_q[idxw], :]  # Q22
                bottom_left = feature[0, bottom_y_index_q[idxh], left_x_index_q[idxw], :]  # Q11
                bottom_right = feature[0, bottom_y_index_q[idxh], right_x_index_q[idxw], :]  # Q21

                # bilinear interpretate
                # f(x,y)=Q12*(1-x_lerp)*(1-y_lerp)+Q22*x_lerp*(1-y_lerp)+Q11*(1-x_lerp)*y_lerp+Q21*x_lerp*y_lerp
                # Q11=bottom_left
                # Q12=top_left
                # Q21=bottom_right
                # Q22=top_right
                # data=top_left*(1-x_lerp[idxw,:])*(1-y_lerp[idxh,:])+top_right*(1-y_lerp[idxh,:])*x_lerp[idxw,:]
                # data=data+bottom_left*(1-x_lerp[idxw,:])*y_lerp[idxh,:]+bottom_right*x_lerp[idxw,:]*y_lerp[idxh,:]
                # xy=y_lerp[idxh,:]*x_lerp[idxw,:]
                # fourpoint_sum=(top_left+bottom_right-top_right-bottom_left)*xy
                # top = top_left + (top_right-top_left)* x_lerp[idxw,:]
                # bottom=(bottom_left - top_left) * y_lerp[idxh,:]
                # data = fourpoint_sum+top+bottom

                xy_q = y_lerp_q[idxh, :]*x_lerp_q[idxw, :] >> qvalue
                fourpoint_sum = (top_left+bottom_right-top_right-bottom_left)*xy_q
                top = (top_left.long() << qvalue) + ((top_right-top_left) * x_lerp_q[idxw, :])
                bottom = (bottom_left - top_left) * y_lerp_q[idxh, :]
                data_q = (fourpoint_sum+top+bottom).long() >> qvalue
                resize_feature[boxidx, idxh, idxw, :] = data_q
    return resize_feature


def mylog2(Am):
    logtab0 = [-32768, -28262, -24149, -20365, -16862, -13600, -10549, -7683, -4981, -2425]
    logtab1 = [22529, 20567, 18920, 17517, 16308, 15255, 14330, 13511, 12780, 12124]  # / *delta_y0 / delta_x0: Q13 * /
    logtab2 = [16384, 18022, 19660, 21299, 22938, 24576, 26214, 27853, 29491, 31130]  # / *x0: Q15 * /

    if Am == 0:
        return -32768
    else:
        point = 0
        var = Am
        while var < 16384:
            point = point + 1
            var = var << 1
    Am = (Am << point)
    point1 = (-point) * 512  # / *inputQ15, output Q9 * /
    index = (int(Am - 16384) * 20) >> 15  # / *tableindex * /
    dx = int(Am - logtab2[index])
    dy = (dx * logtab1[index]) >> 13
    y = (dy + logtab0[index]) >> 6  # / *Q9 * /
    y = point1 + y  # /*log2(x), Q9 * /
    return y


def torch_roi_align_float(node, nor_box, feature_list):
    resize_height_ = node.outputs[0].shape[1]
    resize_width_ = node.outputs[0].shape[2]
    channel_ = node.outputs[0].shape[3]

    h_sample_ratio, w_sample_ratio = node.params['sample']
    out_height, out_width = node.params['resize_height'], node.params['resize_width']
    spatial_list = node.params['spatial_scale_value']
    coordinate_transformation_mode = node.params['coordinate_transformation_mode'].lower()
    _SUPPORT_COORDINATE_MODE = ['half_pixel', 'output_half_pixel']
    if coordinate_transformation_mode not in _SUPPORT_COORDINATE_MODE:
        FATAL(f"{node}, currently coordinate_transformation_mode only support {_SUPPORT_COORDINATE_MODE}, please check! ")
    is_half_pixel = True if coordinate_transformation_mode == 'half_pixel' else False

    y0, x0, y1, x1 = nor_box[0, :, 0], nor_box[0, :, 1], nor_box[0, :, 2], nor_box[0, :, 3]
    norbox_h = y1 - y0
    norbox_w = x1 - x0
    area = norbox_h * norbox_w
    feature_num = len(feature_list)

    area = area  # * image_height * image_width
    sqrt_area = torch.sqrt(area)
    target_lvls = torch.floor(feature_num + torch.log2(sqrt_area / 224) + torch.tensor(1e-6, dtype=torch.float))
    target_lvls = torch.clamp(target_lvls, min=2, max=5)
    target_lvls = (target_lvls.to(torch.int64) - 2).to(torch.int64)
    output = torch.zeros([nor_box.shape[1], resize_height_, resize_width_, channel_],
                         device=node.inputs[1].betensor.device)
    for level in range(feature_num):
        feature_map = feature_list[level].float()
        dev = feature_map.device
        fm_height, fm_width = feature_map.shape[1:3]
        idx_in_level = torch.where(target_lvls == level)[0]
        roi_box = nor_box[0, idx_in_level, :].float()
        current_box_num = roi_box.shape[0]
        half_pixel_offset = 0.5 if is_half_pixel else 0
        nor_output = torch.zeros([current_box_num, resize_height_, resize_width_, channel_],
                                 device=node.inputs[1].betensor.device)

        w_roi_start = roi_box[..., 1] * spatial_list[level] - half_pixel_offset
        w_roi_end = roi_box[..., 3] * spatial_list[level] - half_pixel_offset
        h_roi_start = roi_box[..., 0] * spatial_list[level] - half_pixel_offset
        h_roi_end = roi_box[..., 2] * spatial_list[level] - half_pixel_offset

        roi_width = w_roi_end - w_roi_start
        roi_height = h_roi_end - h_roi_start
        if not is_half_pixel:
            roi_width = torch.maximum(roi_width, torch.tensor(1., device=dev))
            roi_height = torch.maximum(roi_height, torch.tensor(1., device=dev))

        w_step_size = roi_width / out_width
        h_step_size = roi_height / out_height

        if w_sample_ratio <= 0:
            w_sample_ratios = torch.ceil(w_step_size).int()
        else:
            w_sample_ratios = torch.ones([current_box_num], device=roi_box.device).int() * w_sample_ratio
        if h_sample_ratio <= 0:
            h_sample_ratios = torch.ceil(h_step_size).int()
        else:
            h_sample_ratios = torch.ones([current_box_num], device=roi_box.device).int() * h_sample_ratio
        w_bin_size = w_step_size / w_sample_ratios
        h_bin_size = h_step_size / h_sample_ratios

        for b in range(current_box_num):
            if roi_width[b] == 0 or roi_height[b] == 0:
                continue
            for i in range(out_height):
                h_start = h_step_size[b] * i + h_roi_start[b]
                for j in range(out_width):
                    w_start = w_step_size[b] * j + w_roi_start[b]  # [750]
                    depth_output = torch.zeros([channel_], device=feature_map.device)
                    for y_ind in range(h_sample_ratios[b]):
                        y = h_start + h_bin_size[b] / 2 + h_bin_size[b] * y_ind
                        for x_ind in range(w_sample_ratios[b]):
                            x = w_start + w_bin_size[b] / 2 + w_bin_size[b] * x_ind
                            if y < -1.0 or y > fm_height or x < -1.0 or x > fm_width:
                                # ws = [0., 0., 0., 0.]
                                # offset = [0, 0, 0, 0]
                                depth_output += 0
                            else:
                                y = torch.minimum(torch.maximum(y, torch.tensor(0., device=dev)),
                                                  torch.tensor(fm_height - 1, dtype=torch.float32, device=dev))
                                x = torch.minimum(torch.maximum(x, torch.tensor(0., device=dev)),
                                                  torch.tensor(fm_width - 1, dtype=torch.float32, device=dev))
                                x_low = torch.floor(x).long()
                                y_low = torch.floor(y).long()
                                x_high = torch.minimum((x_low + 1), torch.tensor(fm_width - 1, device=dev)).long()
                                y_high = torch.minimum((y_low + 1), torch.tensor(fm_height - 1, device=dev)).long()
                                dx1 = x - x_low
                                dy1 = y - y_low
                                dx2 = 1. - dx1
                                dy2 = 1. - dy1

                                ws = [dx2 * dy2, dx1 * dy2, dx2 * dy1, dx1 * dy1]
                                depth_output += (feature_map[0, y_low, x_low, :] * ws[0] +
                                                 feature_map[0, y_low, x_high, :] * ws[1] +
                                                 feature_map[0, y_high, x_low, :] * ws[2] +
                                                 feature_map[0, y_high, x_high, :] * ws[3])

                    nor_output[b, i, j, :] = depth_output / (w_sample_ratios[b] * h_sample_ratios[b])
        output[idx_in_level, ...] = nor_output
    return output


def local_roi_align_float(node, nor_box, feature_list):
    L1 = 0.09565604811391672
    L2 = 0.02391401202847918
    L2_h = 0.09565604811391672
    L3 = 0.005978503007119795
    L3_h = 0.02391401202847918
    L4 = 0.005978503007119795

    out = node.outputs[0].betensor
    dev = node.inputs[0].betensor.device
    nor_box = nor_box + node.inputs[0].quantization.offset
    nor_box = nor_box.double()  # todo delete

    feature_maps = []
    for i, inp in enumerate(node.inputs):
        feature_maps.append(inp.betensor.float())
    resize_height_ = node.outputs[0].shape[1]
    resize_width_ = node.outputs[0].shape[2]
    channel_ = node.outputs[0].shape[3]

    y0, x0, y1, x1 = nor_box[0, :, 0], nor_box[0, :, 1], nor_box[0, :, 2], nor_box[0, :, 3]
    h = y1 - y0
    w = x1 - x0
    # Use shape of first image. Images in a batch must have the same size.
    # Equation 1 in the Feature Pyramid Networks paper. Account for
    # the fact that our coordinates are normalized here.
    # e.g. a 224x224 ROI (in pixels) maps to P4

    # float32 forward
    # image_area = 10 #10 is log2(1024*1024), 7.807 is log2(224)
    # roi_level = image_area-7.807+ torch.log2(h * w)/2
    # roi_level = torch.minimum(torch.tensor(5).to(device), torch.maximum(
    #     torch.tensor(2).to(device), 4 + torch.round(roi_level).int()))
    roi_level5 = torch.ones_like(h) * 5
    roi_level4 = torch.ones_like(h) * 4
    roi_level3 = torch.ones_like(h) * 3
    roi_level2 = torch.ones_like(h) * 2
    area = h * w
    f5 = (area > L1) * roi_level5
    f4 = ((area < L2_h) & (area > L2)) * roi_level4
    f3 = ((area < L3_h) & (area > L3)) * roi_level3
    f2 = (area < L4) * roi_level2
    roi_level = (f2 + f3 + f4 + f5).int()

    resize_feature = torch.zeros((nor_box.shape[1], resize_height_, resize_width_, channel_), device=dev)
    # for batchidx in range(feature.shape[0]):
    batch_idx = 0
    for boxidx in range(nor_box.shape[1]):
        feature = feature_maps[roi_level[boxidx] - 1]

        y0 = nor_box[batch_idx, boxidx, 0]
        x0 = nor_box[batch_idx, boxidx, 1]
        y1 = nor_box[batch_idx, boxidx, 2]
        x1 = nor_box[batch_idx, boxidx, 3]
        image_width = feature.shape[1]
        image_height = feature.shape[2]
        channel = feature.shape[3]

        height_scale = (y1 - y0) * (image_height - 1) / (resize_height_ - 1)
        width_scale = (x1 - x0) * (image_width - 1) / (resize_width_ - 1)

        x = (image_width - 1) * x0 + torch.arange(0, resize_width_, device=dev) * width_scale
        y = (image_height - 1) * y0 + torch.arange(0, resize_height_, device=dev) * height_scale

        yy = torch.clamp(y, 0, image_width - 1)
        xx = torch.clamp(x, 0, image_height - 1)
        top_y_index = (torch.floor(yy)).int()
        bottom_y_index = (torch.ceil(yy)).int()
        y_lerp = (yy - top_y_index).reshape(resize_height_, 1).repeat(1, channel)
        left_x_index = (torch.floor(xx)).int()
        right_x_index = (torch.ceil(xx)).int()
        x_lerp = (xx - left_x_index).reshape(resize_width_, 1).repeat(1, channel)

        for idxh in range(resize_height_):
            for idxw in range(resize_width_):
                # get 4 point
                top_left = feature[0, top_y_index[idxh], left_x_index[idxw], :]  # Q12
                top_right = feature[0, top_y_index[idxh], right_x_index[idxw], :]  # Q22
                bottom_left = feature[0, bottom_y_index[idxh], left_x_index[idxw], :]  # Q11
                bottom_right = feature[0, bottom_y_index[idxh], right_x_index[idxw], :]  # Q21

                # bilinear interpretate
                # f(x,y)=Q12*(1-x_lerp)*(1-y_lerp)+Q22*x_lerp*(1-y_lerp)+Q11*(1-x_lerp)*y_lerp+Q21*x_lerp*y_lerp
                # Q11=bottom_left
                # Q12=top_left
                # Q21=bottom_right
                # Q22=top_right
                # data=top_left*(1-x_lerp[idxw,:])*(1-y_lerp[idxh,:])+top_right*(1-y_lerp[idxh,:])*x_lerp[idxw,:]
                # data=data+bottom_left*(1-x_lerp[idxw,:])*y_lerp[idxh,:]+bottom_right*x_lerp[idxw,:]*y_lerp[idxh,:]
                xy = y_lerp[idxh, :] * x_lerp[idxw, :]
                fourpoint_sum = (top_left + bottom_right - top_right - bottom_left) * xy
                top = top_left + (top_right - top_left) * x_lerp[idxw, :]
                bottom = (bottom_left - top_left) * y_lerp[idxh, :]
                data = fourpoint_sum + top + bottom

                resize_feature[boxidx, idxh, idxw, :] = data
    return resize_feature


@register_gen_input(OpType.PyramidROIAlign)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_shape2 = list(self.inputs[2].shape)
    in_shape3 = list(self.inputs[3].shape)
    in_shape4 = list(self.inputs[4].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)
    in_type2 = str(self.inputs[2].dtype)
    in_type3 = str(self.inputs[3].dtype)
    in_type4 = str(self.inputs[4].dtype)

    input0 = np.random.randint(0, 32768, size=in_shape0, dtype=in_type0)

    if self.inputs[1].dtype.is_float:
        input1 = (np.random.uniform(0, 10000, size=in_shape1)).astype(in_type1)
        input2 = (np.random.uniform(0, 10000, size=in_shape2)).astype(in_type2)
        input3 = (np.random.uniform(0, 10000, size=in_shape3)).astype(in_type3)
        input4 = (np.random.uniform(0, 10000, size=in_shape4)).astype(in_type4)
    else:
        input1 = np.random.randint(np.iinfo(in_type1).min, np.iinfo(in_type1).max, size=in_shape1, dtype=in_type1)
        input2 = np.random.randint(np.iinfo(in_type2).min, np.iinfo(in_type2).max, size=in_shape2, dtype=in_type2)
        input3 = np.random.randint(np.iinfo(in_type3).min, np.iinfo(in_type3).max, size=in_shape3, dtype=in_type3)
        input4 = np.random.randint(np.iinfo(in_type4).min, np.iinfo(in_type4).max, size=in_shape4, dtype=in_type4)

    input_data = [input0, input1, input2, input3, input4]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    self.inputs[2].betensor = input_data[2]
    self.inputs[3].betensor = input_data[3]
    self.inputs[4].betensor = input_data[4]
    return input_data
