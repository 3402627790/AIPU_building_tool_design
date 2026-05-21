import torch
from AIPUBuilder.core import OpType
from .gt_util import *
from ..engine import register_gt


@register_gt(OpType.YUV444pkdToI420)
def YUV444pkdToI420(self, *args, **kwargs):
    input_tensor = self.inputs[0].betensor.cpu().numpy().astype(np.int32)

    output = []
    for b in range(input_tensor.shape[0]):
        out_b = YUV444pkdToI420(input_tensor[b])
        output.append(out_b)
    output = clamp(np.asarray(output), self.outputs[0].dtype)

    self.outputs[0].betensor = output
    return [o.betensor for o in self.outputs]


def YUV444pkdToI420(input):
    """
    YUV444 -> I420
    :param input: one images. [h, w, 3]
    :return:
    """

    h, w, c = input.shape

    y_out = input[..., 0]
    u_444 = np.reshape(input[..., 1], (h, w))
    v_444 = np.reshape(input[..., 2], (h, w))
    u_out = u_444[::2, ::2]  # pick odd rows and columns
    v_out = v_444[::2, ::2]
    yuv_out = np.concatenate([y_out.flatten(), u_out.flatten(), v_out.flatten()], axis=0)

    return yuv_out
