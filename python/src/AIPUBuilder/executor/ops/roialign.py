from ..engine import register_gt, register_gen_input, _OP_METRICS, register_op_metric
from .gt_util.metric import relative_diff, ulp_diff, cos_distance, check_fp_data
from .gt_util.dtype_utils import torch_to_numpy
from AIPUBuilder.core import OpType
from AIPUBuilder.logger import DEBUG, WARN, ERROR
from .gt_util.quant_utils import clamp
import math
import numpy as np
import torch
import torch.nn.functional as F


@register_gt(OpType.RoiAlign)
def roialign(self, *args, **kwargs):
    if self.inputs[0].dtype.is_int:
        inp = self.inputs[0].betensor.cpu().numpy()
        rois = self.inputs[1].betensor.cpu().numpy()
        inp = inp.astype(np.int64)
        rois = rois.astype(np.int64)
        inp = inp + self.inputs[0].quantization.offset
        rois = rois + self.inputs[1].quantization.offset
        if 0 in self.params['sample']:
            output = int8Foward_old(self, inp, rois)
        else:
            output = int8Foward(self, inp, rois)
        output = clamp(output, self.outputs[0].dtype)
    else:
        # output = floatFoward(self, inp, rois)
        output = roi_align_float(self)
    out_shape = self.attrs["_output_shapes"][0] if "_output_shapes" in self.attrs else list(self.outputs[0].shape)
    output = np.asarray(output, dtype=self.outputs[0].dtype.np).reshape(out_shape)
    self.outputs[0].betensor = output


def _bilinear_interpolate(input, roi_batch_ind, y, x, ymask, xmask):
    _, channels, height, width = input.shape
    invalid_y = (y < -1) | (y > height)
    invalid_x = (x < -1) | (x > width)

    y = y.clip(min=0)  # [K, PH, IY]
    x = x.clip(min=0)  # [K, PW, IX]
    y_low = y.astype(np.int64)
    x_low = x.astype(np.int64)
    y_high = np.where(y_low >= height - 1, height - 1, y_low + 1)
    y_low = np.where(y_low >= height - 1, height - 1, y_low)
    y = np.where(y_low >= height - 1, y.astype(input.dtype), y)

    x_high = np.where(x_low >= width - 1, width - 1, x_low + 1)
    x_low = np.where(x_low >= width - 1, width - 1, x_low)
    x = np.where(x_low >= width - 1, x.astype(input.dtype), x)

    ly = y - y_low
    lx = x - x_low
    hy = 1.0 - ly
    hx = 1.0 - lx

    hy[invalid_y] = 0
    hx[invalid_x] = 0
    ly[invalid_y] = 0
    lx[invalid_x] = 0

    def masked_index(y_, x_):
        y_ = np.where(ymask[:, None, :], y_, 0)
        x_ = np.where(xmask[:, None, :], x_, 0)
        return input[
            roi_batch_ind[:, None, None, None, None, None],
            np.arange(channels)[None, :, None, None, None, None],
            y_[:, None, :, None, :, None],
            x_[:, None, None, :, None, :],
        ]  # [K, C, PH, PW, IY, IX]

    v1 = masked_index(y_low, x_low)
    v2 = masked_index(y_low, x_high)
    v3 = masked_index(y_high, x_low)
    v4 = masked_index(y_high, x_high)

    def outer_prod(y_, x_):
        return y_[:, None, :, None, :, None] * x_[:, None, None, :, None, :]

    w1 = outer_prod(hy, hx)  # [K, C, PH, PW, IY, IX]
    w2 = outer_prod(hy, lx)
    w3 = outer_prod(ly, hx)
    w4 = outer_prod(ly, lx)

    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    return val


def floatFoward(self, inp, rois):
    inp = inp.transpose((0, 3, 1, 2)).copy()

    _, channels, height, width = inp.shape
    rois_cnt, _ = rois.shape

    spatial_height, spatial_width = self.params["spatial_scale_value"]
    pooled_height, pooled_width = self.params["pooled_shape"]
    sample_height, sample_width = self.params["sample"]
    aligned = 'coordinate_transformation_mode' in self.params and self.params[
        'coordinate_transformation_mode'] == 'HALF_PIXEL'

    ph = np.arange(pooled_height)
    pw = np.arange(pooled_width)

    offset = 0.5 if aligned else 0.0
    roi_batch_ind = rois[:, 0].astype(np.int64)
    roi_start_h = rois[:, 1] * spatial_height - offset
    roi_start_w = rois[:, 2] * spatial_width - offset
    roi_end_h = rois[:, 3] * spatial_height - offset
    roi_end_w = rois[:, 4] * spatial_width - offset

    roi_height = roi_end_h - roi_start_h
    roi_width = roi_end_w - roi_start_w
    if not aligned:
        roi_height = roi_height.clip(min=1.0)
        roi_width = roi_width.clip(min=1.0)

    bin_size_h = roi_height / pooled_height
    bin_size_w = roi_width / pooled_width

    roi_bin_grid_h = np.full([rois_cnt], sample_height) if sample_height > 0 else np.ceil(roi_height / pooled_height)
    roi_bin_grid_w = np.full([rois_cnt], sample_width) if sample_width > 0 else np.ceil(roi_width / pooled_width)

    count = np.clip(roi_bin_grid_h * roi_bin_grid_w, 1, None)
    iy = np.arange(height)
    ix = np.arange(width)
    ymask = iy[None, :] < roi_bin_grid_h[:, None]
    xmask = ix[None, :] < roi_bin_grid_w[:, None]

    def from_K(t):
        return t[:, None, None]

    y = (
        from_K(roi_start_h)
        + ph[None, :, None] * from_K(bin_size_h)
        + (iy[None, None, :] + 0.5).astype(inp.dtype) * from_K(bin_size_h / roi_bin_grid_h)
    )  # [K, PH, IY]
    x = (
        from_K(roi_start_w)
        + pw[None, :, None] * from_K(bin_size_w)
        + (ix[None, None, :] + 0.5).astype(inp.dtype) * from_K(bin_size_w / roi_bin_grid_w)
    )  # [K, PW, IX]
    val = _bilinear_interpolate(inp, roi_batch_ind, y, x, ymask, xmask)  # [K, C, PH, PW, IY, IX]

    val = np.where(ymask[:, None, None, None, :, None], val, 0)
    val = np.where(xmask[:, None, None, None, None, :], val, 0)

    output = val.sum((-1, -2))
    output /= count[:, None, None, None]
    output = output
    return output.transpose((0, 2, 3, 1)).astype(inp.dtype)


def int8Foward_old(layer, inp, rois):
    # for sample_y or sample_x = 0
    q_data = inp.astype(np.int32)
    sampleRatio = layer.params['sample']
    exp = layer.params['spatial_shift_value']

    roiInfoLength = rois.shape[0]
    _, inHeigt, inWidth, c = inp.shape
    outWidth = layer.outputs[0].shape[2]
    outHeigt = layer.outputs[0].shape[1]

    if 'coordinate_transformation_mode' in layer.params and layer.params['coordinate_transformation_mode'] == 'HALF_PIXEL':
        offset = 2 ** (exp - 1)
    else:
        offset = 0

    outdata = []
    for k in range(0, roiInfoLength):
        batchId = rois[k, 0]
        if (batchId < 0):
            outdata = np.zeros([1, roiInfoLength * outWidth * outHeigt * c])
            return outdata

        data = q_data[batchId].reshape(-1)

        y_start = int(rois[k, 1] * layer.params['scale_value'][1] * layer.params['spatial_scale_value'][0]
                      * 0.5 ** layer.params['shift_value'][1]) - offset
        x_start = int(rois[k, 2] * layer.params['scale_value'][1] * layer.params['spatial_scale_value'][1]
                      * 0.5 ** layer.params['shift_value'][1]) - offset
        y_end = int(rois[k, 3] * layer.params['scale_value'][1] * layer.params['spatial_scale_value'][0]
                    * 0.5 ** layer.params['shift_value'][1]) - offset
        x_end = int(rois[k, 4] * layer.params['scale_value'][1] * layer.params['spatial_scale_value'][1]
                    * 0.5 ** layer.params['shift_value'][1]) - offset

        # Region out of bound: x1|x2|y1|y2 < 0 || x1|x2 > inWidth || y1|y2 > inHeight
        if ((x_end < x_start) | (y_end < y_start)):
            for idx in range(outWidth * outHeigt):
                outdata.append(np.zeros([c]) - layer.outputs[0].quantization.offset)
            continue
        if ((x_end > inWidth * (2 ** exp)) | (y_end > inHeigt * (2 ** exp))):
            for idx in range(outWidth * outHeigt):
                outdata.append(np.zeros([c]) - layer.outputs[0].quantization.offset)
            continue

        if 'coordinate_transformation_mode' in layer.params and layer.params['coordinate_transformation_mode'] == 'HALF_PIXEL':
            roiWidth = x_end - x_start
            roiHeight = y_end - y_start
        else:
            roiWidth = max(x_end - x_start, (2 ** exp))
            roiHeight = max(y_end - y_start, (2 ** exp))

        wStepSize = int(roiWidth // (outWidth * (2 ** exp)))
        hStepSize = int(roiHeight // (outHeigt * (2 ** exp)))
        wStepSizeMod = roiWidth - wStepSize * (outWidth * (2 ** exp))
        hStepSizeMod = roiHeight - hStepSize * (outHeigt * (2 ** exp))
        if (sampleRatio[1] > 0):
            wSamplingRatio = sampleRatio[1]
        else:
            if (wStepSizeMod > 0):
                wSamplingRatio = wStepSize + 1
            else:
                wSamplingRatio = wStepSize

        if (sampleRatio[0] > 0):
            hSamplingRatio = sampleRatio[0]
        else:
            if (hStepSizeMod > 0):
                hSamplingRatio = hStepSize + 1
            else:
                hSamplingRatio = hStepSize

        numSamplingPoints = wSamplingRatio * hSamplingRatio

        wBinSize = (wStepSize * outWidth * (2 ** exp) + wStepSizeMod) // (
            wSamplingRatio * outWidth * (2 ** exp))
        hBinSize = (hStepSize * outHeigt * (2 ** exp) + hStepSizeMod) // (
            hSamplingRatio * outHeigt * (2 ** exp))

        wBinSizeMod = wStepSize * outWidth * (
            2 ** exp) + wStepSizeMod - wBinSize * wSamplingRatio * outWidth * (2 ** exp)
        hBinSizeMod = hStepSize * outHeigt * (
            2 ** exp) + hStepSizeMod - hBinSize * hSamplingRatio * outHeigt * (2 ** exp)

        for i in range(0, outHeigt):
            for j in range(0, outWidth):
                wStart = wStepSize * j
                wStartMod = wStepSizeMod * j + x_start * outWidth
                hStart = hStepSize * i
                hStartMod = hStepSizeMod * i + y_start * outHeigt
                outdata_batch_l = []
                outdata_batch_h = []
                outdata_batch = []
                if layer.params['method'].upper() == 'MAX':
                    out_val = - 2 ** 31
                else:
                    out_val = 0
                for kk in range(0, c):
                    outdata_batch.append(0)
                    outdata_batch_l.append(0)
                    outdata_batch_h.append(out_val)

                # hSamplingRatio * wSamplingRatio(obtained by bilinear) to calc one output point
                for yInd in range(0, hSamplingRatio):
                    for xInd in range(0, wSamplingRatio):
                        xscale1 = outWidth * wSamplingRatio * (2 ** (exp + 1))
                        xscale2 = 2 * wSamplingRatio
                        xscale3 = outWidth * wSamplingRatio * (2 ** exp)

                        x = wStart + wBinSize * xInd
                        x_tmp1 = xscale2 * wStartMod + xscale3 * wBinSize + wBinSizeMod + 2 * wBinSizeMod * xInd
                        x_tmp2 = int(x_tmp1 / xscale1)

                        x1 = int(x + x_tmp2)
                        x2 = x1 + 1
                        dx1 = x_tmp1 - x_tmp2 * xscale1  # xscale1

                        yscale1 = outHeigt * hSamplingRatio * (2 ** (exp + 1))
                        yscale2 = 2 * hSamplingRatio
                        yscale3 = outHeigt * hSamplingRatio * (2 ** (exp))

                        y = hStart + hBinSize * yInd
                        y_tmp1 = yscale2 * hStartMod + yscale3 * hBinSize + hBinSizeMod + 2 * hBinSizeMod * yInd
                        y_tmp2 = int(y_tmp1 / yscale1)

                        y1 = int(y + y_tmp2)
                        y2 = y1 + 1
                        dy1 = y_tmp1 - y_tmp2 * yscale1  # yscale1

                        if ((x1 > inWidth - 1) | (x1 == inWidth - 1)):
                            x1 = x2 = inWidth - 1
                            dx1 = 0
                        dx2 = xscale1 - dx1

                        if ((y1 > inHeigt - 1) | (y1 == inHeigt - 1)):
                            y1 = y2 = inHeigt - 1
                            dy1 = 0
                        dy2 = yscale1 - dy1

                        if ((x1 < -1) | (y1 < -1) | (x1 > inWidth) | (y1 > inHeigt)):
                            ws = [0, 0, 0, 0]
                            offsets = [0, 0, 0, 0]
                        else:
                            if (x1 <= 0):
                                x1 = 0
                                x2 = 1
                                # dx2 = int(xscale1)

                            if (y1 <= 0):
                                y1 = 0
                                y2 = 1
                                # dy2 = int(yscale1)

                            # ongoning
                            dx1 = int(dx1 / 2)
                            dx2 = int(dx2 / 2)
                            dy1 = int(dy1 / 2)
                            dy2 = int(dy2 / 2)

                            # lib version
                            x_tmp1 = dx1 >> 16
                            dx1 = dx1 - (x_tmp1 << 16)
                            x_tmp2 = dx2 >> 16
                            dx2 = dx2 - (x_tmp2 << 16)

                            y_tmp1 = dy1 >> 16
                            dy1 = dy1 - (y_tmp1 << 16)
                            y_tmp2 = dy2 >> 16
                            dy2 = dy2 - (y_tmp2 << 16)

                            tmp1 = dx2 * dy2
                            tmp2 = x_tmp2 * dy2 + dx2 * y_tmp2
                            tmp3 = x_tmp2 * y_tmp2
                            tmp4 = (tmp1 >> exp) + \
                                (tmp2 << (16 - exp)) + (tmp3 << (32 - exp))
                            ws0 = tmp4

                            tmp1 = dx1 * dy2
                            tmp2 = x_tmp1 * dy2 + dx1 * y_tmp2
                            tmp3 = x_tmp1 * y_tmp2
                            tmp4 = (tmp1 >> exp) + (tmp2 << (16 - exp)) + (tmp3 << (32 - exp))
                            ws1 = tmp4

                            tmp1 = dx2 * dy1
                            tmp2 = x_tmp2 * dy1 + dx2 * y_tmp1
                            tmp3 = x_tmp2 * y_tmp1
                            tmp4 = (tmp1 >> exp) + (tmp2 << (16 - exp)) + (tmp3 << (32 - exp))
                            ws2 = tmp4

                            tmp1 = dx1 * dy1
                            tmp2 = x_tmp1 * dy1 + dx1 * y_tmp1
                            tmp3 = x_tmp1 * y_tmp1
                            tmp4 = (tmp1 >> exp) + (tmp2 << (16 - exp)) + (tmp3 << (32 - exp))
                            ws3 = tmp4
                            ws = (ws0, ws1, ws2, ws3)

                            # # opt version
                            # import math
                            # ws = (dx2 * dy2, dx1 * dy2, dx2 * dy1, dx1 * dy1)
                            # ws = [round(wss * (0.5 ** exp)) for wss in ws]

                            offsets = (y1 * inWidth * c + x1 * c,
                                       y1 * inWidth * c + x2 * c,
                                       y2 * inWidth * c + x1 * c,
                                       y2 * inWidth * c + x2 * c)

                        for kk in range(0, c):
                            if layer.params['method'].upper() == 'MAX':
                                interpolation_l = 0
                                interpolation_h = -2 ** 31
                            else:
                                interpolation_l = 0
                                interpolation_h = 0
                            for num in range(0, 4):
                                tmp = ws[num] - ((ws[num] >> 16) << 16)
                                # print(tmp)
                                # print(data[offsets[num] + kk])
                                if layer.params['method'].upper() == 'MAX':
                                    tmp_interpolation_l = tmp * data[offsets[num] + kk]
                                    tmp_interpolation_h = (ws[num] >> 16) * data[offsets[num] + kk]
                                    tmp_interpolation_h = tmp_interpolation_h + (tmp_interpolation_l >> 16)
                                    tmp_interpolation_l = tmp_interpolation_l - ((tmp_interpolation_l >> 16) << 16)
                                    if tmp_interpolation_h > interpolation_h:
                                        interpolation_h = tmp_interpolation_h
                                        interpolation_l = tmp_interpolation_l
                                    elif tmp_interpolation_h == interpolation_h:
                                        interpolation_h = tmp_interpolation_h
                                        interpolation_l = max(interpolation_l, tmp_interpolation_l)
                                else:
                                    interpolation_l += tmp * data[offsets[num] + kk]
                                    interpolation_h += (ws[num] >> 16) * data[offsets[num] + kk]

                            if layer.params['method'].upper() == 'MAX':
                                if interpolation_h > outdata_batch_h[kk]:
                                    outdata_batch_l[kk] = interpolation_l  # max
                                    outdata_batch_h[kk] = interpolation_h  # high & low bit
                                elif interpolation_h == outdata_batch_h[kk]:
                                    outdata_batch_l[kk] = max(np.uint32(outdata_batch_l[kk]),
                                                              np.uint32(interpolation_l))  # max
                                    outdata_batch_h[kk] = max(outdata_batch_h[kk], interpolation_h)  # high & low bit
                            else:
                                outdata_batch_l[kk] += interpolation_l  # avg
                                outdata_batch_h[kk] += interpolation_h  # high & low bit
                                if layer.outputs[0].dtype.bits == 16:
                                    tmp = outdata_batch_l[kk] >> 16
                                    outdata_batch_l[kk] = outdata_batch_l[kk] - (tmp << 16)
                                    outdata_batch_h[kk] = outdata_batch_h[kk] + tmp
                shift = layer.params['shift_value'][0] + exp
                for kk in range(0, c):
                    # lib version
                    tmp_out = 0
                    tmp = (outdata_batch_l[kk] >> 16)
                    temp1 = outdata_batch_l[kk] - (tmp << 16)
                    temp1 = temp1 * layer.params['scale_value'][0]
                    temp1 = (temp1 // (2 ** shift))
                    tmp_out = tmp_out + temp1
                    temp2 = tmp * layer.params['scale_value'][0]
                    if (shift > 16):
                        temp2 = temp2 // (2 ** (shift - 16))
                    else:
                        temp2 = temp2 * (2 ** (16 - shift))
                    tmp_out = tmp_out + temp2
                    tmp = (outdata_batch_h[kk] >> 16)
                    temp1 = outdata_batch_h[kk] - (tmp << 16)
                    temp1 = temp1 * layer.params['scale_value'][0]
                    if (shift > 16):
                        temp1 = temp1 // (2 ** (shift - 16))
                    else:
                        temp1 = temp1 * (2 ** (16 - shift))
                    tmp_out = tmp_out + temp1
                    temp2 = tmp * layer.params['scale_value'][0]
                    if (shift > 32):
                        temp2 = temp2 // (2 ** (shift - 32))
                    else:
                        temp2 = temp2 * (2 ** (32 - shift))
                    tmp_out = tmp_out + temp2

                    # # opt version
                    # temp3 = outdata_batch_h[kk] * (2 ** 16) + outdata_batch_l[kk]
                    # tmp_out = (temp3 * layer.params['scale_value'][0] / 2 ** shift)

                    if layer.params['method'].upper() == 'MAX':
                        # outHeigt * outWidth * numSamplingPoints from quant
                        temp3 = round(tmp_out / (numSamplingPoints * outHeigt * outWidth))
                    else:
                        temp3 = round(tmp_out / (numSamplingPoints * outHeigt * outWidth * numSamplingPoints))
                    outdata_batch[kk] = temp3
                outdata_batch = np.asarray(outdata_batch)
                outdata_batch -= layer.outputs[0].quantization.offset
                outdata.append(np.clip(np.asarray(outdata_batch), np.iinfo(layer.outputs[0].dtype.np).min,
                                       np.iinfo(layer.outputs[0].dtype.np).max).astype(layer.outputs[0].dtype.np))
    outdata = (np.asarray(outdata)).reshape(-1)
    return outdata


def int8Foward(layer, inp, rois):
    q_data = inp.astype(np.int32)
    sampleRatio = layer.params['sample']
    exp = layer.params['spatial_shift_value']

    roiInfoLength = rois.shape[0]
    _, inHeigt, inWidth, c = inp.shape
    outWidth = layer.outputs[0].shape[2]
    outHeigt = layer.outputs[0].shape[1]

    if 'coordinate_transformation_mode' in layer.params and layer.params['coordinate_transformation_mode'] == 'HALF_PIXEL':
        offset = 2 ** (exp - 1)
    else:
        offset = 0

    outdata = []
    for k in range(0, roiInfoLength):
        batchId = rois[k, 0]
        if (batchId < 0):
            outdata = np.zeros([1, roiInfoLength * outWidth * outHeigt * c])
            return outdata

        data = q_data[batchId].reshape(-1)

        y_start = int(rois[k, 1] * layer.params['scale_value'][1] * layer.params['spatial_scale_value'][0]
                      * 0.5 ** layer.params['shift_value'][1]) - offset
        x_start = int(rois[k, 2] * layer.params['scale_value'][1] * layer.params['spatial_scale_value'][1]
                      * 0.5 ** layer.params['shift_value'][1]) - offset
        y_end = int(rois[k, 3] * layer.params['scale_value'][1] * layer.params['spatial_scale_value'][0]
                    * 0.5 ** layer.params['shift_value'][1]) - offset
        x_end = int(rois[k, 4] * layer.params['scale_value'][1] * layer.params['spatial_scale_value'][1]
                    * 0.5 ** layer.params['shift_value'][1]) - offset

        # Region out of bound: x1|x2|y1|y2 < 0 || x1|x2 > inWidth || y1|y2 > inHeight
        if ((x_end < x_start) | (y_end < y_start)):
            for idx in range(outWidth * outHeigt):
                outdata.append(np.zeros([c]) - layer.outputs[0].quantization.offset)
            continue
        if ((x_end > inWidth * (2 ** exp)) | (y_end > inHeigt * (2 ** exp))):
            for idx in range(outWidth * outHeigt):
                outdata.append(np.zeros([c]) - layer.outputs[0].quantization.offset)
            continue

        if 'coordinate_transformation_mode' in layer.params and layer.params['coordinate_transformation_mode'] == 'HALF_PIXEL':
            roiWidth = x_end - x_start
            roiHeight = y_end - y_start
        else:
            roiWidth = max(x_end - x_start, (2 ** exp))
            roiHeight = max(y_end - y_start, (2 ** exp))

        wStepSize = int(roiWidth * layer.params['bin_scale_value'][1] /
                        2 ** (layer.params['bin_shift_value'][1]))   # bin
        hStepSize = int(roiHeight * layer.params['bin_scale_value'][0] / 2 ** (layer.params['bin_shift_value'][0]))

        wSamplingRatio = sampleRatio[1]
        hSamplingRatio = sampleRatio[0]

        wBinSize = int(wStepSize * layer.params['grid_scale_value'][1] /
                       2 ** (layer.params['grid_shift_value'][1]))  # grid
        hBinSize = int(hStepSize * layer.params['grid_scale_value'][0] / 2 ** (layer.params['grid_shift_value'][0]))

        for i in range(0, outHeigt):
            for j in range(0, outWidth):
                wStart = wStepSize * j + x_start
                hStart = hStepSize * i + y_start

                outdata_batch_l = []
                outdata_batch_h = []
                outdata_batch = []
                if layer.params['method'].upper() == 'MAX':
                    out_val = - 2 ** 31
                else:
                    out_val = 0
                for kk in range(0, c):
                    outdata_batch.append(0)
                    outdata_batch_l.append(0)
                    outdata_batch_h.append(out_val)

                # hSamplingRatio * wSamplingRatio(obtained by bilinear) to calc one output point
                for yInd in range(0, hSamplingRatio):
                    for xInd in range(0, wSamplingRatio):
                        y = hStart + hBinSize + 2 * hBinSize * yInd
                        x = wStart + wBinSize + 2 * wBinSize * xInd
                        x1 = int((x // (2 ** exp)))
                        y1 = int((y // (2 ** exp)))
                        x2 = x1 + 1
                        y2 = y1 + 1

                        dx1 = int(x - (x1 * (2 ** exp)))
                        dy1 = int(y - (y1 * (2 ** exp)))

                        if y1 < -1 or y1 > inHeigt or x1 < -1 or x1 > inWidth:
                            ws = [0, 0, 0, 0]
                            offsets = [0, 0, 0, 0]
                        else:
                            y1 = min(max(y1, 0), inHeigt - 1)
                            x1 = min(max(x1, 0), inWidth - 1)
                            if ((x1 >= inWidth - 1)):
                                x1 = x2 = inWidth - 1
                                dx1 = 0
                            if ((y1 >= inHeigt - 1)):
                                y1 = y2 = inHeigt - 1
                                dy1 = 0
                            dx2 = int((2 ** exp) - dx1)
                            dy2 = int((2 ** exp) - dy1)
                            ws = (dx2 * dy2 >> (exp),
                                  dx1 * dy2 >> (exp),
                                  dx2 * dy1 >> (exp),
                                  dx1 * dy1 >> (exp))

                            offsets = (y1 * inWidth * c + x1 * c,
                                       y1 * inWidth * c + x2 * c,
                                       y2 * inWidth * c + x1 * c,
                                       y2 * inWidth * c + x2 * c)

                        for kk in range(0, c):
                            if layer.params['method'].upper() == 'MAX':
                                interpolation_l = 0
                                interpolation_h = -2 ** 31
                            else:
                                interpolation_l = 0
                                interpolation_h = 0
                            for num in range(0, 4):
                                tmp = ws[num] - ((ws[num] >> 16) << 16)
                                # print(tmp)
                                # print(data[offsets[num] + kk])
                                if layer.params['method'].upper() == 'MAX':
                                    tmp_interpolation_l = tmp * data[offsets[num] + kk]
                                    tmp_interpolation_h = (ws[num] >> 16) * data[offsets[num] + kk]
                                    tmp_interpolation_h = tmp_interpolation_h + (tmp_interpolation_l >> 16)
                                    tmp_interpolation_l = tmp_interpolation_l - ((tmp_interpolation_l >> 16) << 16)
                                    if tmp_interpolation_h > interpolation_h:
                                        interpolation_h = tmp_interpolation_h
                                        interpolation_l = tmp_interpolation_l
                                    elif tmp_interpolation_h == interpolation_h:
                                        interpolation_h = tmp_interpolation_h
                                        interpolation_l = max(interpolation_l, tmp_interpolation_l)
                                else:
                                    interpolation_l += tmp * data[offsets[num] + kk]
                                    interpolation_h += (ws[num] >> 16) * data[offsets[num] + kk]

                            if layer.params['method'].upper() == 'MAX':
                                if interpolation_h > outdata_batch_h[kk]:
                                    outdata_batch_l[kk] = interpolation_l  # max
                                    outdata_batch_h[kk] = interpolation_h  # high & low bit
                                elif interpolation_h == outdata_batch_h[kk]:
                                    outdata_batch_l[kk] = max(np.uint32(outdata_batch_l[kk]),
                                                              np.uint32(interpolation_l))  # max
                                    outdata_batch_h[kk] = max(outdata_batch_h[kk], interpolation_h)  # high & low bit
                            else:
                                outdata_batch_l[kk] += interpolation_l  # avg
                                outdata_batch_h[kk] += interpolation_h  # high & low bit
                                if layer.outputs[0].dtype.bits == 16:
                                    tmp = outdata_batch_l[kk] >> 16
                                    outdata_batch_l[kk] = outdata_batch_l[kk] - (tmp << 16)
                                    outdata_batch_h[kk] = outdata_batch_h[kk] + tmp
                shift = layer.params['shift_value'][0] + exp
                for kk in range(0, c):
                    # lib version
                    tmp_out = 0
                    tmp = (outdata_batch_l[kk] >> 16)
                    temp1 = outdata_batch_l[kk] - (tmp << 16)
                    temp1 = temp1 * layer.params['scale_value'][0]
                    temp1 = np.round(temp1 * (0.5 ** shift))
                    tmp_out = tmp_out + temp1
                    temp2 = tmp * layer.params['scale_value'][0]
                    if (shift > 16):
                        temp2 = np.round(temp2 * (0.5 ** (shift - 16)))
                    else:
                        temp2 = temp2 * (2 ** (16 - shift))
                    tmp_out = tmp_out + temp2
                    tmp = (outdata_batch_h[kk] >> 16)
                    temp1 = outdata_batch_h[kk] - (tmp << 16)
                    temp1 = temp1 * layer.params['scale_value'][0]
                    if (shift > 16):
                        temp1 = np.round(temp1 * (0.5 ** (shift - 16)))
                    else:
                        temp1 = temp1 * (2 ** (16 - shift))
                    tmp_out = tmp_out + temp1
                    temp2 = tmp * layer.params['scale_value'][0]
                    if (shift > 32):
                        temp2 = np.round(temp2 * (0.5 ** (shift - 32)))
                    else:
                        temp2 = temp2 * (2 ** (32 - shift))
                    tmp_out = tmp_out + temp2
                    temp3 = tmp_out

                    # # opt version
                    # temp3 = outdata_batch_h[kk] * (2**16) + outdata_batch_l[kk]
                    # temp3 = round(temp3 * layer.params['scale_value'][0] / 2 ** shift)

                    # if self.method.upper() == 'MAX':
                    #     temp3 = int(tmp_out / (numSamplingPoints * outHeigt * outWidth))  # outHeigt * outWidth * numSamplingPoints from quant
                    # else:
                    #     temp3 = int(tmp_out / (numSamplingPoints * outHeigt * outWidth * numSamplingPoints))
                    outdata_batch[kk] = temp3
                outdata_batch = np.asarray(outdata_batch)
                outdata_batch -= layer.outputs[0].quantization.offset
                outdata.append(np.clip(np.asarray(outdata_batch), np.iinfo(layer.outputs[0].dtype.np).min,
                                       np.iinfo(layer.outputs[0].dtype.np).max).astype(layer.outputs[0].dtype.np))
    outdata = (np.asarray(outdata)).reshape(-1)
    return outdata


def roi_align_float(self):
    from torchvision.ops import roi_align

    sample_h, sample_w = self.params['sample']
    pool_height, pool_width = self.params['pooled_shape']
    method = self.params['method'].lower()
    spatial_y, spatial_x = self.params['spatial_scale_value']
    coordinate_transformation_mode = self.params['coordinate_transformation_mode']
    is_half_pixel = coordinate_transformation_mode == 'HALF_PIXEL'
    inp_d = self.inputs[0].betensor
    rois = self.inputs[1].betensor

    if method == 'avg':
        if spatial_x != spatial_y or sample_w != sample_h or coordinate_transformation_mode == 'HALF_PIXEL':
            params = {}
            params['output_size'] = self.params['pooled_shape']
            params['sample_ratio'] = self.params['sample']
            params['spatial_scale'] = self.params['spatial_scale_value']
            params['method'] = method
            params['is_half_pixel'] = is_half_pixel
            out = local_float_roi_align(inp_d, rois, params)
        else:
            split_rois = torch.split(rois, split_size_or_sections=1, dim=-1)
            rois = torch.cat([split_rois[0], split_rois[2], split_rois[1], split_rois[4], split_rois[3]], dim=-1)
            fm = torch.permute(inp_d, [0, 3, 1, 2])  # nhwc2nchw
            out = roi_align(fm.float(), rois.float(), [pool_height, pool_width],
                            spatial_scale=spatial_x, sampling_ratio=sample_w)
            out = torch.permute(out, [0, 2, 3, 1])  # nchw2nhwc
    else:  # method=='max'
        params = {}
        params['output_size'] = self.params['pooled_shape']
        params['sample_ratio'] = self.params['sample']
        params['spatial_scale'] = self.params['spatial_scale_value']
        params['method'] = method
        params['is_half_pixel'] = is_half_pixel
        out = local_float_roi_align(inp_d, rois, params)
    return out


def local_float_roi_align(fm, rois, params):
    out_height, out_width = params['output_size']
    spatial_y, spatial_x = params['spatial_scale']
    h_sample_ratio, w_sample_ratio = params['sample_ratio']
    method = params['method'].lower()
    is_half_pixel = params['is_half_pixel']

    # nhwc
    in_height, in_width, in_depth = fm.shape[1:]
    roi_num = rois.shape[0]
    dev = fm.device
    output_shape = [roi_num, out_height, out_width, fm.shape[-1]]
    out = torch.zeros(*output_shape)
    half_pixel_offset = 0.5 if is_half_pixel else 0.
    for r in range(roi_num):
        batch_idx = rois[r, 0]
        if batch_idx < 0:
            WARN(f"RoiAlign layer: the batch_index of box_id={r} is {batch_idx} < 0.")
            continue
        if batch_idx > fm.shape[0] - 1:
            ERROR(f"RoiAlign layer: batch_index={batch_idx} should be < the featuremap batch_size={fm.shape[0]}")
            continue

        batch_base = torch.flatten(fm[batch_idx.long()])
        w_roi_start = rois[r, 2] * spatial_x - half_pixel_offset
        w_roi_end = rois[r, 4] * spatial_x - half_pixel_offset
        h_roi_start = rois[r, 1] * spatial_y - half_pixel_offset
        h_roi_end = rois[r, 3] * spatial_y - half_pixel_offset

        if (w_roi_end < w_roi_start) or (h_roi_end < h_roi_start):
            continue
        if (w_roi_end.item() > in_width) or (h_roi_end.item() > in_height):
            continue

        roi_width = w_roi_end - w_roi_start
        roi_height = h_roi_end - h_roi_start
        if not is_half_pixel:
            roi_width = torch.maximum((roi_width), torch.tensor(1.0, device=dev))
            roi_height = torch.maximum((roi_height), torch.tensor(1.0, device=dev))

        w_step_size = roi_width / out_width
        h_step_size = roi_height / out_height

        # if sampling_ratio=0, use adaptive value of ceil(roi_width/out_width), same for height
        w_sampling_ratio = w_sample_ratio if w_sample_ratio > 0 else int(math.ceil(w_step_size))
        h_sampling_ratio = h_sample_ratio if h_sample_ratio > 0 else int(math.ceil(h_step_size))

        w_bin_size = w_step_size / w_sampling_ratio
        h_bin_size = h_step_size / h_sampling_ratio

        for i in range(out_height):
            for j in range(out_width):
                w_start = w_step_size * j + w_roi_start
                w_end = w_step_size * (j + 1) + w_roi_start
                h_start = h_step_size * i + h_roi_start
                h_end = h_step_size * (i + 1) + h_roi_start

                if method == 'avg':
                    out_k = torch.zeros(in_depth, device=dev)
                else:  # method == max
                    out_k = torch.full([in_depth], torch.finfo(torch.float32).min)
                for y_ind in range(h_sampling_ratio):
                    y = h_start + h_bin_size / 2 + h_bin_size * y_ind
                    for x_ind in range(w_sampling_ratio):

                        x = w_start + w_bin_size / 2 + w_bin_size * x_ind

                        if y < -1.0 or y > in_height or x < -1.0 or x > in_width:
                            ws = [0., 0., 0., 0.]
                            offset = [0, 0, 0, 0]
                        else:
                            y = torch.minimum(torch.maximum(y, torch.tensor(0., device=dev)),
                                              torch.tensor(in_height - 1, dtype=torch.float32, device=dev))
                            x = torch.minimum(torch.maximum(x, torch.tensor(0., device=dev)),
                                              torch.tensor(in_width - 1, dtype=torch.float32, device=dev))
                            x_low = torch.floor(x)
                            y_low = torch.floor(y)
                            x_high = torch.minimum((x_low + 1), torch.tensor(in_width - 1, device=dev))
                            y_high = torch.minimum((y_low + 1), torch.tensor(in_height - 1, device=dev))
                            dx1 = x - x_low
                            dy1 = y - y_low
                            dx2 = 1. - dx1
                            dy2 = 1. - dy1

                            ws = [dx2 * dy2, dx1 * dy2, dx2 * dy1, dx1 * dy1]
                            offset = [y_low * in_width * in_depth + x_low * in_depth,
                                      y_low * in_width * in_depth + x_high * in_depth,
                                      y_high * in_width * in_depth + x_low * in_depth,
                                      y_high * in_width * in_depth + x_high * in_depth
                                      ]
                            offset = [o.long() for o in offset]
                        for k in range(in_depth):
                            if method == 'avg':
                                interpolation = 0
                                for c in range(4):
                                    interpolation += ws[c] * batch_base[offset[c] + k]
                                out_k[k] += interpolation
                            else:  # max
                                max_4points = torch.max(torch.tensor([ws[0] * batch_base[offset[0] + k],
                                                                      ws[1] * batch_base[offset[1] + k],
                                                                      ws[2] * batch_base[offset[2] + k],
                                                                      ws[3] * batch_base[offset[3] + k],
                                                                      ]))
                                out_k[k] = torch.maximum(out_k[k], max_4points)

                if method == 'avg':
                    out_k = out_k / (w_sampling_ratio * h_sampling_ratio)
                else:  # max
                    pass
                out[r, i, j, :] = out_k
    return out.to(dev)


@register_gen_input(OpType.RoiAlign)
def gen_input_data(self):
    import random
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)

    if "int" in in_type0:
        info = np.iinfo(in_type0)
        feature_map = np.random.uniform(info.min, info.max, size=in_shape0).astype(in_type0)
    else:
        feature_map = np.random.uniform(0, 1, size=in_shape0).astype(in_type0)

    y_spatial, x_spatial = self.params["spatial_scale_value"]
    batch = in_shape0[0]
    num_rois = in_shape1[0]
    y_max = int((in_shape0[1] - 1) / y_spatial) if self.inputs[0].dtype.is_float else in_shape0[1] - 1
    x_max = int((in_shape0[2] - 1) / x_spatial) if self.inputs[0].dtype.is_float else in_shape0[2] - 1

    rois = []
    for n in range(num_rois):
        y = random.sample(range(0, y_max), 2)
        x = random.sample(range(0, x_max), 2)
        min_y = min(y)
        min_x = min(x)
        max_y = max(y)
        max_x = max(x)
        batch_id = np.random.choice(range(0, batch))
        roi = [batch_id, min_y, min_x, max_y, max_x]
        rois.append(roi)
    rois = np.asarray(rois, dtype=np.uint16)

    input_data = [feature_map, rois]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    return input_data


@register_op_metric(OpType.RoiAlign)
@check_fp_data
def metric(self, outputs, index=None):
    def call(r, u, c, t):
        if "float" in t:
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
