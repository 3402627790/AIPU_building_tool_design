from .gt_util import *
from ..engine import register_gt
from AIPUBuilder.core import OpType


@register_gt(OpType.YuvToRgb)
def yuvtorgb(self, *args, **kwargs):
    input = self.inputs[0].betensor.cpu().numpy().astype(np.int32)
    coefficient = self.params['coefficient']
    coefficient_shift = self.params['coefficient_shift']
    output_type = str(self.outputs[0].dtype)
    output_shape = list(self.outputs[0].shape)
    coefficient = np.asarray(coefficient, dtype='int32')
    input_offset = coefficient[:3].reshape(3, 1)
    output_offset = coefficient[3: 6].reshape(3, 1)
    coefficient = coefficient[6:].reshape(3, 3)
    round_offset = np.asarray([2 ** (coefficient_shift - 1)] * 3).reshape(3, 1)
    N, H, W = output_shape[:3]
    output = []
    output_yuv444 = []
    for b in range(N):
        yuv444 = yuv420_to_444_per_batch(self, input[b], H, W, output_offset)
        out = np.matmul(coefficient, yuv444 - input_offset) + round_offset
        out = np.round(out * (0.5 ** coefficient_shift))
        out = np.transpose(out)
        out = out.reshape(H, W, 3)
        out = clamp(out, output_type)
        output.append(out)
        yuv444_b = np.transpose(yuv444.copy())
        output_yuv444.append(yuv444_b)
    output = np.asarray(output)
    self.outputs[0].betensor = output
    return output_yuv444


def yuv420_to_444_per_batch(self, yuv420, H, W, output_offset):
    y = yuv420[0: W * H].reshape(H, W)
    u = yuv420[W * H: W * H * 5 // 4].reshape(H // 2, W // 2)
    u0 = u.copy()
    v = yuv420[W * H * 5 // 4:].reshape(H // 2, W // 2)
    v0 = v.copy()

    # Convert channel U, V
    u444 = I420_to_444_single_channel(self, y, u, u0, H, W)
    v444 = I420_to_444_single_channel(self, y, v, v0, H, W)

    y444 = y.flatten()
    u444 = u444.flatten()
    v444 = v444.flatten()

    y444 = y444[np.newaxis, :]
    u444 = u444[np.newaxis, :]
    v444 = v444[np.newaxis, :]
    yuv444 = np.concatenate((y444, u444, v444), axis=0)
    yuv444 = yuv444 - output_offset

    return yuv444


def I420_to_444_single_channel(self, y, u, u0, H, W):
    for i in range(H // 2 - 1):
        ins_val_u = np.round((u0[i] + u0[i + 1]) * (0.5 ** 1))
        u = np.insert(u, 2 * i + 1, ins_val_u, axis=0)
    u = np.insert(u, u.shape[0], u0[H // 2 - 1], axis=0)

    u422 = u.copy()
    u444 = np.zeros_like(y)
    for j in range(W):
        if j == 0:
            u444[:, j] = u422[:, 0]
        elif j == W-1:
            u444[:, j] = u422[:, -1]
        elif j % 2 == 1:
            u444[:, j] = u422[:, (j - 1) // 2] * 3 + u422[:, (j + 1) // 2] * 1
            u444[:, j] = np.round(u444[:, j] * (0.5 ** 2))
        elif j % 2 == 0:
            u444[:, j] = u422[:, (j - 2) // 2] * 1 + u422[:, j // 2] * 3
            u444[:, j] = np.round(u444[:, j] * (0.5 ** 2))

    return u444
