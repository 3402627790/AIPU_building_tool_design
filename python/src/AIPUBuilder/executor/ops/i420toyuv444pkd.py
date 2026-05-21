import torch
from AIPUBuilder.core import OpType
from .gt_util import *
from ..engine import register_gt


@register_gt(OpType.I420ToYUV444pkd)
def I420ToYUV444pkd(self, *args, **kwargs):
    input = self.inputs[0].betensor.cpu().numpy().astype(np.int32)
    N, H, W = list(self.outputs[0].shape)[:3]
    output = []
    for b in range(N):
        out = yuv420_to_444_per_batch(input[b], H, W)
        out = np.round(out)
        out = clamp(out, self.outputs[0].dtype)
        output.append(out)
    output = np.asarray(output)

    self.outputs[0].betensor = output
    return [o.betensor for o in self.outputs]


def yuv420_to_444_per_batch(yuv420, H, W):
    y = yuv420[0: W * H].reshape(H, W)
    u = yuv420[W * H: W * H * 5 // 4].reshape(H // 2, W // 2)
    u0 = u.copy()
    v = yuv420[W * H * 5 // 4:].reshape(H // 2, W // 2)
    v0 = v.copy()

    ##############################
    # Channel: I420 -> YUV422
    ##############################
    def I420_to_444_single_channel(y, u, u0, H):
        for i in range(H // 2 - 1):
            ins_val_u = np.round((u0[i] + u0[i + 1]) * (0.5 ** 1))
            u = np.insert(u, 2 * i + 1, ins_val_u, axis=0)
        u = np.insert(u, u.shape[0], u0[H // 2 - 1], axis=0)

        u422 = u.copy()
        u444 = np.zeros_like(y)
        for j in range(W):
            if j == 0:
                u444[:, j] = u422[:, 0]
            elif j == W - 1:
                u444[:, j] = u422[:, -1]
            elif j % 2 == 1:
                u444[:, j] = u422[:, (j - 1) // 2] * 3 + u422[:, (j + 1) // 2] * 1
                u444[:, j] = np.round(u444[:, j] * (0.5 ** 2))
            elif j % 2 == 0:
                u444[:, j] = u422[:, (j - 2) // 2] * 1 + u422[:, j // 2] * 3
                u444[:, j] = np.round(u444[:, j] * (0.5 ** 2))

        return u444

    # Convert channel U, V
    u444 = I420_to_444_single_channel(y, u, u0, H)
    v444 = I420_to_444_single_channel(y, v, v0, H)

    # Concat Y, U, V channels
    yuv444 = np.zeros(shape=(H, W, 3))
    yuv444[:, :, 0] = y
    yuv444[:, :, 1] = u444
    yuv444[:, :, 2] = v444

    return yuv444
