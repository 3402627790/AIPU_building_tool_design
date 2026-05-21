from ..engine import register_gt, register_gen_input
from AIPUBuilder.core import OpType
import torch
import numpy as np
from .gt_util.quant_utils import clamp


@register_gt(OpType.GridSample)
def grid_sample(self, *args, **kwargs):
    feature = self.inputs[0]
    grid = self.inputs[1]
    shifts = self.params['shift_value']
    do_scale = self.params['scale_value']
    method = self.params['method'].lower()
    padding_mode = self.params['padding_mode'].lower()
    align_corners = self.params['align_corners']

    if feature.betensor.dim() == 5:
        output = quant_grid_sample_5d(self,
                                      feature,
                                      grid,
                                      method,
                                      padding_mode,
                                      align_corners,
                                      do_scale,
                                      shifts
                                      )
    else:
        output = quant_grid_sample(self,
                                   feature,
                                   grid,
                                   method,
                                   padding_mode,
                                   align_corners,
                                   do_scale,
                                   shifts
                                   )
        if 'has_transpose' in self.params.keys():
            if self.params['has_transpose']:  # nhwc transpose to nchw
                output = output.permute(0, 3, 1, 2)
    output = clamp(output, self.outputs[0].dtype)
    self.outputs[0].betensor = output
    return self.outputs[0].betensor


def quant_grid_sample(self, inp0, inp1, method, padding_mode, align_corners, do_scale, shifts):
    feature = inp0.betensor.int()
    grid = inp1.betensor.int()
    if 'has_transpose' in self.params.keys():  # nchw transpose to nhwc
        if self.params['has_transpose']:
            feature = feature.permute(0, 2, 3, 1)
            grid = grid.permute(0, 2, 3, 1)

    feature_batch = feature.shape[0]
    feature_height = feature.shape[1]
    feature_width = feature.shape[2]
    feature_channel = feature.shape[3]
    resize_height = grid.shape[1]
    resize_width = grid.shape[2]
    offset_w = (feature_width - 1)
    offset_h = (feature_height - 1)

    grid += int(inp1.quantization.offset)
    feature_zp = int(inp0.quantization.offset)
    q16_qmin, q16_qmax = -2 ** 15, 2 ** 15 - 1
    do_scale0, do_scale1 = do_scale
    do_shift0, do_shift1 = shifts
    gridx_shift = self.params['coordinate_x_shift']
    gridy_shift = self.params['coordinate_y_shift']

    if align_corners:
        x_min = 0
        x_max = offset_w * (2 ** gridx_shift)
        y_min = 0
        y_max = offset_h * (2 ** gridy_shift)
    else:
        x_min = -2 ** (gridx_shift - 1)
        x_max = feature_width * (2 ** gridx_shift) - 2 ** (gridx_shift - 1)
        y_min = -2 ** (gridy_shift - 1)
        y_max = feature_height * (2 ** gridy_shift) - 2 ** (gridy_shift - 1)

    quant_output = torch.zeros((feature_batch, resize_height, resize_width,
                                feature_channel), device=inp0.betensor.device)
    for n in range(feature_batch):
        qgrid_x = grid[n, :, :, 0].reshape(-1, )
        qgrid_y = grid[n, :, :, 1].reshape(-1, )

        q_ix = (qgrid_x * do_scale0 >> do_shift0) + offset_w * (2 ** (gridx_shift - 1))
        q_iy = (qgrid_y * do_scale1 >> do_shift1) + offset_h * (2 ** (gridy_shift - 1))
        q_ix = torch.clamp(q_ix, q16_qmin, q16_qmax)
        q_iy = torch.clamp(q_iy, q16_qmin, q16_qmax)
        if method == 'nearest':
            q_ix += 2 ** (gridx_shift - 1)
            q_iy += 2 ** (gridy_shift - 1)
            q_ix = torch.clamp(q_ix, q16_qmin, q16_qmax)
            q_iy = torch.clamp(q_iy, q16_qmin, q16_qmax)
            left_x = (q_ix >> gridx_shift).long()
            top_y = (q_iy >> gridy_shift).long()
            quant_output[n, :, :, :] = PixelAtGrid(feature, n, feature_channel, top_y, left_x,
                                                   feature_height, feature_width, resize_height, resize_width,
                                                   padding_mode, feature_zp)
        else:
            # if padding_mode == 'border':
            #     q_ix = grid_clamp(q_ix, x_min, x_max, 0, offset_w * (2 ** gridx_shift))
            #     q_iy = grid_clamp(q_iy, y_min, y_max, 0, offset_h * (2 ** gridy_shift))
            left_x = (q_ix.long() >> gridx_shift).long()
            top_y = (q_iy.long() >> gridy_shift).long()
            right_x = left_x + 1
            bottom_y = top_y + 1
            x_terp = q_ix - left_x * 2 ** (gridx_shift)
            y_terp = q_iy - top_y * 2 ** (gridy_shift)
            x_terp = x_terp.reshape(resize_height, resize_width, 1)
            y_terp = y_terp.reshape(resize_height, resize_width, 1)

            top_left = PixelAtGrid(feature, n, feature_channel, top_y, left_x, feature_height,
                                   feature_width, resize_height, resize_width, padding_mode, feature_zp)
            top_right = PixelAtGrid(feature, n, feature_channel, top_y, right_x, feature_height,
                                    feature_width, resize_height, resize_width, padding_mode, feature_zp)
            bottom_left = PixelAtGrid(feature, n, feature_channel, bottom_y, left_x, feature_height,
                                      feature_width, resize_height, resize_width, padding_mode, feature_zp)
            bottom_right = PixelAtGrid(feature, n, feature_channel, bottom_y, right_x, feature_height,
                                       feature_width, resize_height, resize_width, padding_mode, feature_zp)

            top = top_left + (((top_right - top_left) * x_terp).long() >> (gridx_shift))
            bottom = bottom_left + (((bottom_right - bottom_left) * x_terp).long() >> gridx_shift)
            quant_output[n, :, :, :] = (top + (((bottom - top) * y_terp).long() >> gridy_shift))
    return quant_output


def quant_grid_sample_5d(self, inp0, inp1, method, padding_mode, align_corners, do_scale, shifts):
    feature = inp0.betensor.int()
    grid = inp1.betensor.int()
    feature_batch = feature.shape[0]
    feature_z = feature.shape[1]
    feature_height = feature.shape[2]
    feature_width = feature.shape[3]
    feature_channel = feature.shape[4]
    resize_z = grid.shape[1]
    resize_height = grid.shape[2]
    resize_width = grid.shape[3]
    offset_z = (feature_z - 1)
    offset_w = (feature_width - 1)
    offset_h = (feature_height - 1)

    grid += int(inp1.quantization.offset)
    feature_zp = int(inp0.quantization.offset)
    q16_qmin, q16_qmax = -2**15, 2**15-1
    do_scale0, do_scale1, do_scale2 = do_scale
    do_shift0, do_shift1, do_shift2 = shifts
    gridx_shift = self.params['coordinate_x_shift']
    gridy_shift = self.params['coordinate_y_shift']
    gridz_shift = self.params['coordinate_z_shift']

    horizontal_output = torch.zeros((feature_batch, resize_z, resize_height, resize_width,
                                     feature_channel), device=inp0.betensor.device).long()
    quant_output = torch.zeros((feature_batch, resize_z, resize_height, resize_width,
                                feature_channel), device=inp0.betensor.device).long()
    for n in range(feature_batch):
        qgrid_x = grid[n, :, :, :, 0].reshape(-1,)  # [2,3,4]
        qgrid_y = grid[n, :, :, :, 1].reshape(-1,)
        qgrid_z = grid[n, :, :, :, 2].reshape(-1,)
        q_ix = ((qgrid_x * do_scale0) >> do_shift0) + offset_w * (2**(gridx_shift-1))
        q_iy = ((qgrid_y * do_scale1) >> do_shift1) + offset_h * (2**(gridy_shift-1))
        q_iz = ((qgrid_z * do_scale2) >> do_shift2) + offset_z * (2**(gridz_shift-1))
        q_ix = torch.clamp(q_ix, q16_qmin, q16_qmax)
        q_iy = torch.clamp(q_iy, q16_qmin, q16_qmax)
        q_iz = torch.clamp(q_iz, q16_qmin, q16_qmax)
        if method == 'nearest':
            q_ix += 2 ** (gridx_shift-1)
            q_iy += 2 ** (gridy_shift-1)
            q_iz += 2 ** (gridz_shift-1)
            q_ix = torch.clamp(q_ix, q16_qmin, q16_qmax)
            q_iy = torch.clamp(q_iy, q16_qmin, q16_qmax)
            q_iz = torch.clamp(q_iz, q16_qmin, q16_qmax)
            x = (q_ix >> gridx_shift).long()
            y = (q_iy >> gridy_shift).long()
            z = (q_iz >> gridz_shift).long()
            quant_output[n, :, :, :, :] = PixelAtGrid3D(feature, n, feature_channel, z, y, x,
                                                        feature_z, feature_height, feature_width, resize_z, resize_height, resize_width, padding_mode, feature_zp)
        else:
            # if padding_mode == 'border':
            #     q_ix = grid_clamp(q_ix, x_min, x_max, 0, offset_w * (2**gridx_shift))
            #     q_iy = grid_clamp(q_iy, y_min, y_max, 0, offset_h * (2**gridy_shift))
            left_x = (q_ix.long() >> gridx_shift).long()
            top_y = (q_iy.long() >> gridy_shift).long()
            z1 = (q_iz.long() >> gridz_shift).long()
            right_x = left_x + 1
            bottom_y = top_y + 1
            z2 = z1 + 1
            x_terp = q_ix - left_x * 2 ** (gridx_shift)
            y_terp = q_iy - top_y * 2 ** (gridy_shift)
            z_terp = q_iz - z1 * 2 ** (gridz_shift)
            x_terp = x_terp.reshape(resize_z, resize_height, resize_width, 1)
            y_terp = y_terp.reshape(resize_z, resize_height, resize_width, 1)
            z_terp = z_terp.reshape(resize_z, resize_height, resize_width, 1)

            z1_top_left = PixelAtGrid3D(feature, n, feature_channel, z1, top_y, left_x, feature_z, feature_height,
                                        feature_width, resize_z, resize_height, resize_width, padding_mode, feature_zp)
            z1_top_right = PixelAtGrid3D(feature, n, feature_channel, z1, top_y, right_x, feature_z, feature_height,
                                         feature_width, resize_z, resize_height, resize_width, padding_mode, feature_zp)
            z1_bottom_left = PixelAtGrid3D(feature, n, feature_channel, z1, bottom_y, left_x, feature_z, feature_height,
                                           feature_width, resize_z, resize_height, resize_width, padding_mode, feature_zp)
            z1_bottom_right = PixelAtGrid3D(feature, n, feature_channel, z1, bottom_y, right_x, feature_z, feature_height,
                                            feature_width, resize_z, resize_height, resize_width, padding_mode, feature_zp)

            top = z1_top_left + (((z1_top_right - z1_top_left) * x_terp).long() >> (gridx_shift))
            bottom = z1_bottom_left + (((z1_bottom_right - z1_bottom_left) * x_terp).long() >> gridx_shift)
            horizontal_output[n, :, :, :, :] = (top + (((bottom - top) * y_terp).long() >> gridy_shift))

            z2_top_left = PixelAtGrid3D(feature, n, feature_channel, z2, top_y, left_x, feature_z, feature_height,
                                        feature_width, resize_z, resize_height, resize_width, padding_mode, feature_zp)
            z2_top_right = PixelAtGrid3D(feature, n, feature_channel, z2, top_y, right_x, feature_z, feature_height,
                                         feature_width, resize_z, resize_height, resize_width, padding_mode, feature_zp)
            z2_bottom_left = PixelAtGrid3D(feature, n, feature_channel, z2, bottom_y, left_x, feature_z, feature_height,
                                           feature_width, resize_z, resize_height, resize_width, padding_mode, feature_zp)
            z2_bottom_right = PixelAtGrid3D(feature, n, feature_channel, z2, bottom_y, right_x, feature_z, feature_height,
                                            feature_width, resize_z, resize_height, resize_width, padding_mode, feature_zp)

            top = z2_top_left + (((z2_top_right - z2_top_left) * x_terp).long() >> (gridx_shift))
            bottom = z2_bottom_left + (((z2_bottom_right - z2_bottom_left) * x_terp).long() >> gridx_shift)
            quant_output[n, :, :, :, :] = (top + (((bottom - top) * y_terp).long() >> gridy_shift))

            quant_output[n, :, :, :, :] = horizontal_output[n, :, :, :, :] + \
                (((quant_output[n, :, :, :, :] - horizontal_output[n, :, :, :, :]) * z_terp).long() >> gridz_shift)
    return quant_output


def grid_clamp(x, x_min, x_max, clip_xmin, clip_xmax):
    less_mask = x < x_min  # torch.bitwise_or(x < x_min, x>x_max)
    greater_mask = x > x_max
    x[less_mask] = clip_xmin
    x[greater_mask] = clip_xmax
    return x


def PixelAtGrid(feature, n, c, y, x, h, w, resize_height, resize_width, padding_mode, feature_zp):
    # default border
    y_less0_mask = y < 0
    y_greater_mask = y > (h - 1)
    x_less0_mask = x < 0
    x_greater_mask = x > (w - 1)
    y_c = y.clone()
    x_c = x.clone()
    y_c[y_less0_mask] = 0
    y_c[y_greater_mask] = h - 1
    x_c[x_less0_mask] = 0
    x_c[x_greater_mask] = w - 1
    out = feature[n, y_c.long(), x_c.long(), :]
    if padding_mode == "border":
        out = out.reshape(resize_height, resize_width, c)
    if padding_mode == "zeros":
        out = out.reshape(resize_height * resize_width, c)
        out[y_less0_mask, :] = -feature_zp
        out[y_greater_mask, :] = -feature_zp
        out[x_less0_mask, :] = -feature_zp
        out[x_greater_mask, :] = -feature_zp
        out = out.reshape(resize_height, resize_width, c)
    return out


def PixelAtGrid3D(feature, n, c, z, y, x, z_in, h_in, w_in, resize_z, resize_height, resize_width, padding_mode, feature_zp):
    # default border
    y_less0_mask = y < 0
    y_greater_mask = y > (h_in - 1)
    x_less0_mask = x < 0
    x_greater_mask = x > (w_in - 1)
    z_less0_mask = z < 0
    z_greater_mask = z > (z_in - 1)
    y_c = y.clone()
    x_c = x.clone()
    z_c = z.clone()
    y_c[y_less0_mask] = 0
    y_c[y_greater_mask] = h_in - 1
    x_c[x_less0_mask] = 0
    x_c[x_greater_mask] = w_in - 1
    z_c[z_less0_mask] = 0
    z_c[z_greater_mask] = z_in - 1
    out = feature[n, z_c, y_c, x_c, :]
    if padding_mode == "border":
        out = out.reshape(resize_z, resize_height, resize_width, c)
    if padding_mode == "zeros":
        out = out.reshape(resize_z * resize_height*resize_width, c)
        out[z_less0_mask, :] = -feature_zp
        out[z_greater_mask, :] = -feature_zp
        out[y_less0_mask, :] = -feature_zp
        out[y_greater_mask, :] = -feature_zp
        out[x_less0_mask, :] = -feature_zp
        out[x_greater_mask, :] = -feature_zp
        out = out.reshape(resize_z, resize_height, resize_width, c)
    return out


# @register_gen_input(OpType.GridSample)
# def gen_input_data(self):
#     in_shape0 = list(self.inputs[0].shape)
#     in_shape1 = list(self.inputs[1].shape)
#     in_type0 = str(self.inputs[0].dtype)
#     in_type1 = str(self.inputs[1].dtype)
#
#     if str(self.params['has_transpose']).lower() == 'true':
#         n, c, h, w = in_shape0
#     else:
#         n, h, w, c = in_shape0
#     offset_w = w - 1
#     offset_h = h - 1
#     offset_h = max(1, offset_h)
#     offset_w = max(1, offset_w)
#     method = self.params['method'].lower()
#     scale_value = self.params['scale_value']
#     shift_value = self.params['shift_value']
#     shift_diff = shift_value[1] - shift_value[0]
#
#     align_true_grid_x_min = ((-2 ** (31 + shift_value[0])) // offset_w - 2 ** shift_value[1]) / scale_value
#     align_true_grid_x_max = (((2 ** 31 - 1) * 2 ** shift_value[0]) // offset_w - 2 ** shift_value[1]) / scale_value
#     align_false_grid_x_min = ((-2 ** (31 + shift_value[0])) // w) / \
#         scale_value - 2 ** shift_value[1] / offset_w / scale_value
#     align_false_grid_x_max = (
#         ((2 ** 31 - 1) * 2 ** shift_value[0]) // offset_w) / scale_value - 2 ** shift_value[1] / w / scale_value
#     align_true_grid_y_min = ((-2 ** (31 + shift_value[0])) // offset_h - 2 ** shift_value[1]) / scale_value
#     align_true_grid_y_max = (((2 ** 31 - 1) * 2 ** shift_value[0]) // offset_h - 2 ** shift_value[1]) / scale_value
#     align_false_grid_y_min = ((-2 ** (31 + shift_value[0])) // h) / \
#         scale_value - 2 ** shift_value[1] / offset_h / scale_value
#     align_false_grid_y_max = (
#         ((2 ** 31 - 1) * 2 ** shift_value[0]) // offset_h) / scale_value - 2 ** shift_value[1] / h / scale_value
#
#     grid_min = max(np.iinfo(in_type1).min,
#                    max(align_true_grid_x_min, align_true_grid_y_min, align_false_grid_x_min, align_false_grid_y_min))
#     grid_max = min(np.iinfo(in_type1).max,
#                    min(align_true_grid_x_max, align_true_grid_y_max, align_false_grid_x_max, align_false_grid_y_max))
#     if grid_min > grid_max:
#         grid_min = np.iinfo(in_type1).min
#         grid_max = np.iinfo(in_type1).max
#     input0 = input1 = None
#     while input0 is None:
#         input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)
#         input1 = np.random.randint(grid_min, grid_max, size=in_shape1, dtype=in_type1)
#         out_h = in_shape1[1]
#         out_w = in_shape1[2]
#         offset_w = (w - 1)
#         offset_h = (h - 1)
#
#         input_1 = np.int64(input1) + self.inputs[1].quantization.offset
#         input_zp = self.inputs[0].quantization.offset
#
#         output = np.zeros([n, out_h, out_w, c], dtype=np.int64)
#         for i in range(n):
#             grid_x = input_1[i, :, :, 0].reshape(-1, )
#             grid_y = input_1[i, :, :, 1].reshape(-1, )
#             if self.params['align_corners']:
#                 ix = ((grid_x * scale_value) >> shift_value[0]) * offset_w + offset_w * 2 ** (shift_diff)
#                 iy = ((grid_y * scale_value) >> shift_value[0]) * offset_h + offset_h * 2 ** (shift_diff)
#                 if not ((ix > (-2 ** 31)).all() and (ix < (2 ** 31) - 1).all() and (iy > (-2 ** 31)).all() and (iy < (2 ** 31) - 1).all()):
#                     input0 = None
#                     break
#                 # 1. ((grid_x * scale_value) * 0.5 ** shift_0) * offset_w < 2 **31  2.offset_w * 2 ** (shift_diff) < 2 ** 31
#                 # grid_x < 2 ** 31 / offset_w * 2 ** shift_0 //scale_value   grid_x < 2 ** (31 + shift_0)//scale_value
#                 # 2. ((grid_y * scale_value) >> shift_value[0]) * offset_h < 2 **31  2.offset_h * 2 ** (shift_diff) < 2 ** 31
#             else:
#                 ix = ((grid_x * scale_value) >> shift_value[0]) * w + offset_w * 2 ** (shift_diff)
#                 iy = ((grid_y * scale_value) >> shift_value[0]) * h + offset_h * 2 ** (shift_diff)
#                 if not ((ix > (-2 ** 31)).all() and (ix < (2 ** 31) - 1).all() and (iy > (-2 ** 31)).all() and (iy < (2 ** 31) - 1).all()):
#                     input0 = None
#                     break
#
#             if method == 'nearest':
#                 ix += 2 ** shift_diff
#                 iy += 2 ** shift_diff
#                 if not ((ix > (-2 ** 31)).all() and (ix < (2 ** 31) - 1).all() and (iy > (-2 ** 31)).all() and (iy < (2 ** 31) - 1).all()):
#                     input0 = None
#                     break
#
#     input_data = [input0, input1]
#     self.inputs[0].betensor = input_data[0]
#     self.inputs[1].betensor = input_data[1]
#     return input_data
