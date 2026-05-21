import torch
from .gt_util import *
from ..engine import register_gt, register_gen_input
from AIPUBuilder.core import OpType
from AIPUBuilder.logger import WARN


@register_gt(OpType.MaxRoiPool)
def MaxRoiPool(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float and (not ('is_aiff' in kwargs.keys() and kwargs['is_aiff'])):
        output = maxroipool_float(self)
    else:
        input_tensor, rois = self.inputs[0].irtensor, self.inputs[1].irtensor
        input_tensor = torch_to_numpy(input_tensor, self.inputs[0].dtype.np)
        rois = torch_to_numpy(rois, self.inputs[1].dtype.np)

        if 'is_aiff' in kwargs.keys() and kwargs['is_aiff'] == True:
            print('MaxRoiPool aiff gt...')
            output = maxroipool_aiff(self, input_tensor, rois)
        else:
            print('MaxRoiPool tpc gt...')
            output = maxroipool_tpc_android(self, input_tensor, rois)  # tpc default

        if not self.inputs[0].dtype.is_float:
            output = clamp(output, self.outputs[0].dtype)

    self.outputs[0].irtensor = output
    return [o.betensor for o in self.outputs]


# discarded
def maxroipool_tpc(self, feature, rois_box):
    spatial = self.params['spatial']
    pooled_shape = self.params['pooled_shape']

    feature = torch.from_numpy(feature.astype(np.int32))
    rois_box = torch.from_numpy(rois_box.astype(np.int32))

    # to match some IR shape, no batch dimension
    if rois_box.ndim == 2:
        rois_box = rois_box.unsqueeze(0)
    rois = rois_box[..., 1:]
    spatial_scale_value = [spatial[1], spatial[0]]

    resize_height, resize_width = pooled_shape
    height, width, channel = feature.shape[1: 4]

    spatial_scale_int16 = [(spatial_scale_value[0] * 255) / 65535,
                           (spatial_scale_value[1] * 255) / 65535]
    top_data = torch.zeros((rois.shape[0], rois.shape[1], resize_height, resize_width, channel))
    # here is unreasonable, but lib remove a parameter which indicate box scale, lib maybe match specific case
    # but opt need match general cases test, so workaround like this
    box_qvalue = 0  # int(math.log2(self.inputs[1].scale))
    qvalue_one = 1 << (box_qvalue + 8)
    qvalue_half = 1 << (box_qvalue + 7)
    scaled_qvalue = box_qvalue + 8
    rois = rois.long() + self.inputs[1].quantization.offset
    for i in range(rois_box.shape[0]):
        batch_index = rois_box[i, :, 0] + self.inputs[1].quantization.offset
        for boxidx in range(rois.shape[1]):
            batch_idx = batch_index[boxidx].int()
            # rois is origin size, feature map size is reduced by 1/16, origin size=224, feature size=224/16/=14
            roi_start_w_q = ((rois[i, boxidx, 1] * spatial_scale_int16[1])).int()  # Q13=(8+box_qvalue)
            roi_end_w_q = ((rois[i, boxidx, 3] * spatial_scale_int16[1])).int()  # Q13
            roi_start_h_q = ((rois[i, boxidx, 0] * spatial_scale_int16[0])).int()
            roi_end_h_q = ((rois[i, boxidx, 2] * spatial_scale_int16[0])).int()

            roi_height_q = (max(roi_end_h_q - roi_start_h_q + qvalue_one,
                                qvalue_one) + qvalue_half) >> scaled_qvalue
            roi_width_q = (max(roi_end_w_q - roi_start_w_q + qvalue_one, qvalue_one) + qvalue_half) >> scaled_qvalue
            bin_size_h_q = int((roi_height_q * 256 / resize_height))
            bin_size_w_q = int((roi_width_q * 256 / resize_width))

            # have pooled size h,such as case, vector len is 7
            hstart_q = ((torch.arange(0, resize_height)) * bin_size_h_q) >> 8
            wstart_q = ((torch.arange(0, resize_width)) * bin_size_w_q) >> 8

            hend_q = ((torch.arange(0, resize_height)) * bin_size_h_q + bin_size_h_q)
            hend_q = (hend_q >> 8) + ((hend_q & 0xff) != 0).short()

            wend_q = ((torch.arange(0, resize_width)) * bin_size_w_q + bin_size_w_q)
            wend_q = (wend_q >> 8) + ((wend_q & 0xff) != 0).short()

            hstart_q = torch.clamp(hstart_q + ((roi_start_h_q + qvalue_half) >> scaled_qvalue),
                                   0, height)  # have pooled size h,such as case, vector len is 7
            hend_q = torch.clamp(hend_q + ((roi_start_h_q + qvalue_half) >> scaled_qvalue), 0, height)
            wstart_q = torch.clamp(wstart_q + ((roi_start_w_q + qvalue_half) >> scaled_qvalue), 0, width)
            wend_q = torch.clamp(wend_q + ((roi_start_w_q + qvalue_half) >> scaled_qvalue), 0, width)

            for ph in range(resize_height):
                for pw in range(resize_width):
                    if hstart_q[ph] >= hend_q[ph] or wstart_q[pw] >= wend_q[pw]:
                        continue
                    else:
                        top_data[i, boxidx, ph, pw, :] = torch.max(
                            torch.max(feature[batch_idx, hstart_q[ph]:hend_q[ph], wstart_q[pw]:wend_q[pw], :], 0)[0],
                            0)[0]

    output = top_data.cpu().numpy().astype(str(self.outputs[0].dtype)).reshape(self.outputs[0].shape)
    return output


def maxroipool_tpc_android(self, feature, rois_box):
    spatial = self.params['spatial']
    pooled_shape = self.params['pooled_shape']
    spatial_y, spatial_x = spatial

    rois = np.array(rois_box, dtype=np.uint32)
    rois = rois + self.inputs[1].quantization.offset
    # map roi
    rois[:, 1] = rois[:, 1] * np.uint32(spatial_y)
    rois[:, 2] = rois[:, 2] * np.uint32(spatial_x)
    rois[:, 3] = rois[:, 3] * np.uint32(spatial_y)
    rois[:, 4] = rois[:, 4] * np.uint32(spatial_x)

    rois_mod = rois[:, 1:] % 65536
    rois[:, 1:] = rois[:, 1:] // 65536
    flag = np.greater_equal(rois_mod, 32768)
    rois[:, 1:] = np.where(flag, rois[:, 1:] + 1, rois[:, 1:])

    input_tensor = np.int32(feature)
    output = pool(self, input_tensor, rois, pooled_shape, is_aiff=False)
    return output


def maxroipool_aiff(self, feature, rois_box):
    spatial = self.params['spatial']
    pooled_shape = self.params['pooled_shape']
    spatial_y, spatial_x = spatial
    input_tensor = np.int32(feature)
    if self.inputs[0].dtype.is_float:
        spatial_y = np.uint32(np.round(spatial_y * 65536))
        spatial_x = np.uint32(np.round(spatial_x * 65536))
        input_tensor = np.float32(feature)
    rois = np.array(rois_box, dtype=np.float32)
    rois = rois + self.inputs[1].quantization.offset
    # map roi
    rois[:, 1] = rois[:, 1] * np.uint32(spatial_y)
    rois[:, 2] = rois[:, 2] * np.uint32(spatial_x)
    rois[:, 3] = rois[:, 3] * np.uint32(spatial_y)
    rois[:, 4] = rois[:, 4] * np.uint32(spatial_x)

    rois[:, 1:] = np.round(rois[:, 1:] / 65536)
    rois = np.clip(rois, np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
    rois = rois.astype(np.uint32)

    output = pool(self, input_tensor, rois, pooled_shape)
    return output


def pool(self, src, rois, pooled_shape, is_aiff=True):
    """
    roi_pooling for each roi.
    :param src:
    :param rois: [batch_id, min_x, min_y, max_x, max_y]
    :param pooled_shape:
    :return:
    """
    y = np.zeros([rois.shape[0], pooled_shape[0], pooled_shape[1], src.shape[3]], dtype=np.int32)
    if self.inputs[0].dtype.is_float:
        y = np.zeros([rois.shape[0], pooled_shape[0], pooled_shape[1], src.shape[3]], dtype=np.float32)

    for index, rois_id in enumerate(rois):
        batch_id = rois_id[0]
        y1, x1, y2, x2 = rois_id[1:]
        x1 = np.int32(x1)
        y1 = np.int32(y1)
        x2 = np.int32(x2)
        y2 = np.int32(y2)
        width = max(1, int(x2) - int(x1) + 1)
        height = max(1, int(y2) - int(y1) + 1)
        bin_size_w = width // pooled_shape[1]  # calc output for each bin_h * bin_w
        bin_size_h = height // pooled_shape[0]
        bin_size_w_mod = width % pooled_shape[1]
        bin_size_h_mod = height % pooled_shape[0]

        # tpc version
        if not is_aiff:
            for channels in range(src.shape[3]):
                for block_height_id in range(pooled_shape[0]):
                    for block_width_id in range(pooled_shape[1]):
                        hstart = (block_height_id * bin_size_h + (bin_size_h_mod * block_height_id) // pooled_shape[0])
                        wstart = (block_width_id * bin_size_w + (bin_size_w_mod * block_width_id) // pooled_shape[1])
                        hend = (block_height_id + 1) * bin_size_h + (bin_size_h_mod * (block_height_id + 1)) // \
                            pooled_shape[0] + (
                            1 if (bin_size_h_mod * (block_height_id + 1)) % pooled_shape[0] else 0)
                        wend = (block_width_id + 1) * bin_size_w + (bin_size_w_mod * (block_width_id + 1)) // \
                            pooled_shape[1] + (1 if (bin_size_w_mod * (block_width_id + 1)) % pooled_shape[1] else 0)

                        hstart = (min(max(hstart + y1, 0), src.shape[1]))
                        hend = (min(max(hend + y1, 0), src.shape[1]))
                        wstart = (min(max(wstart + x1, 0), src.shape[2]))
                        wend = (min(max(wend + x1, 0), src.shape[2]))
                        is_empty = (hend <= hstart) or (wend <= wstart)

                        if (is_empty):
                            continue

                        y[index, block_height_id, block_width_id, channels] = np.max(src[batch_id,
                                                                                     hstart:hend,
                                                                                     wstart:wend,
                                                                                     channels], axis=(0, 1))
        # aiff version
        else:
            x1 = min(np.uint32(x1), src.shape[2] - 1)
            y1 = min(np.uint32(y1), src.shape[1] - 1)
            x2 = min(np.uint32(x2), src.shape[2] - 1)
            y2 = min(np.uint32(y2), src.shape[1] - 1)

            # todo: print some info
            # case_path = self.forward.case_path if hasattr(self.forward, 'case_path') else './'
            # temp_kn_x = max(1, width / self.layer_top_shape[0][2])
            # temp_kn_y = max(1, height / self.layer_top_shape[0][1])
            # if temp_kn_x * temp_kn_y > 256:
            #     aipu_assert_false("aipu assert out of aiff spec, ERROR_CODE 0x6d0381: roi_index= {}".format(index),
            #                       case_path)
            # if x1 > x2 or y1 > y2 or x1 >= src.shape[2] or y1 >= src.shape[1] or \
            #         x2 >= src.shape[2] or y2 >= src.shape[1]:
            #     aipu_assert_false("aipu assert out of feature map, ERROR_CODE 0x6d0381: roi_index= {}".format(index),
            #                       case_path)
            width = min(width, src.shape[2])
            height = min(height, src.shape[1])
            if ((width < (2 * pooled_shape[1] - 2)) or (width <= pooled_shape[1])) and \
                    ((height < (2 * pooled_shape[0] - 2)) or (height <= pooled_shape[0])):
                for block_height_id in range(pooled_shape[0]):
                    for block_width_id in range(pooled_shape[1]):
                        hstart = (block_height_id * height // pooled_shape[0]) + y1
                        hend = (hstart + 1) if (bin_size_h < 1) else (((block_height_id + 1) * height //
                                                                        pooled_shape[0]) + y1)
                        wstart = (block_width_id * width // pooled_shape[1]) + x1
                        wend = (wstart + 1) if (bin_size_w < 1) else ((block_width_id + 1) * width //
                                                                        pooled_shape[1]) + x1
                        if hend <= hstart or wend <= wstart:
                            continue
                        # print(index, block_height_id, block_width_id, channels)
                        # print(batch_id, hstart, hend, wstart, wend, channels)
                        y[index, block_height_id, block_width_id, :] = np.max(
                            src[batch_id, hstart:hend, wstart:wend, :], axis=(0, 1))
            else:
                # if kernel < 1, set width/height = pooled_shape[1]/pooled_shape[0]
                if bin_size_w == 0:
                    bin_size_w += 1
                    x1 -= min((pooled_shape[1] - width) // 2, x1)
                    width = pooled_shape[1]
                    x2 = x1 + width - 1
                    if x2 >= src.shape[2]:
                        x2 = src.shape[2] - 1
                        x1 = x2 + 1 - width
                if bin_size_h == 0:
                    bin_size_h += 1
                    y1 -= min((pooled_shape[0] - height) // 2, y1)
                    height = pooled_shape[0]
                    y2 = y1 + height - 1
                    if y2 >= src.shape[1]:
                        y2 = src.shape[1] - 1
                        y1 = y2 + 1 - height
                pad_l, pad_r, pad_t, pad_b = 0, 0, 0, 0
                bin_size_w_mod = width % pooled_shape[1]
                bin_size_h_mod = height % pooled_shape[0]
                # if need pad and pad < kernel, do pad; else do crop
                if bin_size_w_mod != 0:
                    bin_size_w += 1
                    if (pooled_shape[1] - bin_size_w_mod) > (bin_size_w * 2 - 2):
                        bin_size_w -= 1
                        x1 += (bin_size_w_mod // 2)
                        width -= bin_size_w_mod
                    else:
                        pad_l = (pooled_shape[1] - bin_size_w_mod) // 2
                        pad_r = (pooled_shape[1] - bin_size_w_mod) - pad_l
                if bin_size_h_mod != 0:
                    bin_size_h += 1
                    if (pooled_shape[0] - bin_size_h_mod) > (bin_size_h * 2 - 2):
                        bin_size_h -= 1
                        y1 += (bin_size_h_mod // 2)
                        height -= bin_size_h_mod
                    else:
                        pad_t = (pooled_shape[0] - bin_size_h_mod) // 2
                        pad_b = (pooled_shape[0] - bin_size_h_mod) - pad_t
                # AIFF need fixed kernel size and kernel size = stride in one roi
                x_end = min(x1 + width, src.shape[2])
                y_end = min(y1 + height, src.shape[1])
                for block_height_id in range(pooled_shape[0]):
                    for block_width_id in range(pooled_shape[1]):
                        # ignore pad part in pooling
                        hstart = max(block_height_id * bin_size_h + y1 - pad_t, y1)
                        wstart = max(block_width_id * bin_size_w + x1 - pad_l, x1)
                        hend = min((block_height_id + 1) * bin_size_h + y1 - pad_t, y_end)
                        wend = min((block_width_id + 1) * bin_size_w + x1 - pad_l, x_end)

                        if hend <= hstart or wend <= wstart:
                            continue
                        y[index, block_height_id, block_width_id, :] = np.max(
                            src[batch_id, hstart:hend, wstart:wend, :], axis=(0, 1))
    return y


def maxroipool_float(self):
    out = self.outputs[0].betensor
    feature = self.inputs[0].betensor
    rois_box = self.inputs[1].betensor

    resize_height = self.outputs[0].shape[1]
    resize_width = self.outputs[0].shape[2]
    channel = self.outputs[0].shape[3]
    height, width = feature.shape[1:3]
    # to match some IR shape,no batch dimension
    if rois_box.ndim == 2:
        rois_box = rois_box.unsqueeze(0)
    rois_box = rois_box.to(torch.float32)
    rois = rois_box[..., 1:]  # [:,0] stand for batch index

    if feature.shape[0] != rois_box.shape[0]:
        WARN("MaxRoiPool batch dim is not correct")
    spatial_scale_value = [self.params['spatial'][0], self.params['spatial'][1]]  # [spatial_y, spatial_x] in IR
    dev = feature.device
    top_data = torch.zeros((rois.shape[0], rois.shape[1], resize_height, resize_width, channel), device=dev)
    for i in range(rois_box.shape[0]):
        batch_index = rois_box[i, :, 0]
        for boxidx in range(rois.shape[1]):
            batch_idx = batch_index[boxidx].int()
            roi_start_w = (torch.floor(rois[i, boxidx, 1] * spatial_scale_value[1] + 0.5)).int()
            roi_end_w = (torch.floor(rois[i, boxidx, 3] * spatial_scale_value[1] + 0.5)).int()
            roi_start_h = (torch.floor(rois[i, boxidx, 0] * spatial_scale_value[0] + 0.5)).int()
            roi_end_h = (torch.floor(rois[i, boxidx, 2] * spatial_scale_value[0] + 0.5)).int()

            roi_height = max(roi_end_h - roi_start_h + 1, 1)
            roi_width = max(roi_end_w - roi_start_w + 1, 1)
            bin_size_h = roi_height / resize_height
            bin_size_w = roi_width / resize_width

            hstart = torch.floor((torch.arange(0, resize_height, device=out.device)) *
                                 bin_size_h)  # have pooled size h,such as case, vector len is 7
            wstart = torch.floor((torch.arange(0, resize_width, device=out.device)) * bin_size_w)
            hend = torch.ceil((torch.arange(1, resize_height + 1, device=out.device)) * bin_size_h)
            wend = torch.ceil((torch.arange(1, resize_width + 1, device=out.device)) * bin_size_w)

            # have pooled size h,such as case, vector len is 7
            hstart = (torch.clamp(hstart + roi_start_h, 0, height)).int()
            hend = (torch.clamp(hend + roi_start_h, 0, height)).int()
            wstart = (torch.clamp(wstart + roi_start_w, 0, width)).int()
            wend = (torch.clamp(wend + roi_start_w, 0, width)).int()

            for ph in range(resize_height):
                for pw in range(resize_width):
                    if hstart[ph] >= hend[ph] or wstart[pw] >= wend[pw]:
                        continue
                    else:
                        top_data[i, boxidx, ph, pw, :] = torch.max(
                            torch.max(feature[batch_idx, hstart[ph]:hend[ph], wstart[pw]:wend[pw], :], 0)[0], 0)[0]
    return top_data.reshape(list(self.outputs[0].shape))


@register_gen_input(OpType.MaxRoiPool)
def gen_input_data(self):
    import random
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)
    spatial_y, spatial_x = self.params['spatial']

    if self.inputs[0].dtype.is_float:
        feature_map = np.random.uniform(1, 10000, size=in_shape0).astype(in_type0)
        spatial_y = np.uint32(np.round(spatial_y * 65536))
        spatial_x = np.uint32(np.round(spatial_x * 65536))
    else:
        feature_map = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)
    batch = in_shape0[0]

    num_rois = in_shape1[0]
    y_max = in_shape0[1]
    x_max = in_shape0[2]

    rois = []
    for n in range(num_rois):
        counter = 0
        while True:
            out_of_spec = False
            out_of_map = False
            counter = counter+1
            if self.inputs[1].dtype.bits == 8:
                # uint8
                y = random.sample(range(min(0, -self.inputs[1].quantization.offset), min(y_max, 256)), 2)
                x = random.sample(range(min(0, -self.inputs[1].quantization.offset), min(x_max, 256)), 2)
            else:
                y = random.sample(range(min(0, -self.inputs[1].quantization.offset), y_max), 2)
                x = random.sample(range(min(0, -self.inputs[1].quantization.offset), x_max), 2)
            min_y = min(y)
            min_x = min(x)
            max_y = max(y)
            max_x = max(x)
            low = max(0, -self.inputs[1].quantization.offset)
            high = batch - self.inputs[1].quantization.offset
            batch_id = np.random.randint(low, high)
            # map roi
            min_y_temp = np.round(min_y * np.uint32(spatial_y)/65536).astype(np.uint32)
            min_x_temp = np.round(min_x * np.uint32(spatial_x)/65536).astype(np.uint32)
            max_y_temp = np.round(max_y * np.uint32(spatial_y)/65536).astype(np.uint32)
            max_x_temp = np.round(max_x * np.uint32(spatial_x)/65536).astype(np.uint32)
            x1 = min(np.uint32(min_x_temp), x_max - 1)
            y1 = min(np.uint32(min_y_temp), y_max - 1)
            x2 = min(np.uint32(max_x_temp), x_max - 1)
            y2 = min(np.uint32(max_y_temp), y_max - 1)
            width = max(1, int(x2) - int(x1) + 1)
            height = max(1, int(y2) - int(y1) + 1)
            temp_kn_x = max(1, width / self.outputs[0].shape[2])
            temp_kn_y = max(1, height / self.outputs[0].shape[1])
            if counter > 100:
                break
            if temp_kn_x * temp_kn_y > 256:
                out_of_spec = True
            if x1 >= x2 or y1 >= y2 or x1 >= x_max or y1 >= y_max or \
                    x2 >= x_max or y2 >= y_max:
                out_of_map = True
            if not out_of_map and not out_of_spec:
                break
        roi = [batch_id, min_y, min_x, max_y, max_x]
        rois.append(roi)
    rois = np.asarray(rois, dtype=in_type1)

    input_data = [feature_map, rois]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    return input_data
