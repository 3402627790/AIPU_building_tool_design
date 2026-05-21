import torch
from AIPUBuilder.core import OpType
from .gt_util import *
from .gt_util.quant_utils import _tpc_lut
from ..engine import register_gt


@register_gt(OpType.RgbToYuv)
def RgbToYuv(self, *args, **kwargs):
    input = self.inputs[0].betensor.cpu().numpy().astype(np.int32)

    output = []
    output_yuv444 = []
    for b in range(input.shape[0]):
        out_b, out_444_b = Rgb2Yuv(self, input[b])
        output.append(out_b)
        output_yuv444.append(out_444_b)
    output = clamp(np.asarray(output), self.outputs[0].dtype)
    output_yuv444 = clamp(np.asarray(output_yuv444), self.outputs[0].dtype)

    self.outputs[0].betensor = output
    return [o.betensor for o in self.outputs]


def Rgb2Yuv(self, input):
    """
    only support format=I420, conversion=BT709
    :param input: one batch of images. [h, w, 3]
    :return:
    """
    coefficient = self.params['coefficient']
    coefficient_shift = self.params['coefficient_shift']
    input_offset = np.asarray(coefficient[: 3]).reshape((3, 1))
    output_offset = np.asarray(coefficient[3: 6]).reshape((3, 1))
    coefficient = np.reshape(coefficient[6:], (3, 3))
    pix_num = np.prod(input.shape) // 3
    h, w, c = input.shape
    input_2d = input.reshape((-1, 3))
    input_2d = np.transpose(input_2d, (1, 0))  # [n, 3] -> [3, n]

    # matmul and shift
    out = np.matmul(coefficient, input_2d)
    out = output_offset + ((out + np.asarray([2 ** (coefficient_shift - 1)] * 3).reshape((3, 1))) >> coefficient_shift)
    out_444 = out.copy()
    out_444 = np.transpose(out_444)  # # [3, n] -> [n, 3] y0u0v0y1u1v1...

    # drop data according to I420.
    y_out = out[0]
    u_444 = np.reshape(out[1], (h, w))
    v_444 = np.reshape(out[2], (h, w))
    u_out = u_444[::2, ::2]  # pick odd rows and columns
    v_out = v_444[::2, ::2]
    yuv_out = np.concatenate([y_out, u_out.flatten(), v_out.flatten()], axis=0)

    return yuv_out, out_444
