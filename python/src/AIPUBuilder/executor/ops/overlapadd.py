import torch
from ..engine import register_gt
from AIPUBuilder.core import OpType
from .gt_util.quant_utils import clamp, linear_requantize
from AIPUBuilder.logger import ERROR


@register_gt(OpType.OverlapAdd)
def OverlapAdd(self, *args, **kwargs):
    inp = self.inputs[0].betensor
    output_type = str(self.outputs[0].dtype)
    if output_type != 'float16':
        inp += torch.tensor(self.inputs[0].quantization.offset, device=inp.device).int()

    if len(inp.shape) < 2:
        ERROR("input tensor must be at least rank 2")
    frame_step = self.params.get('frame_step', 1)
    shape_list = list(inp.size())
    frame_length = shape_list[-1]
    if frame_step > frame_length:
        ERROR("frame step must be less than or equal to frame length")
    frames = shape_list[-2]
    output_size = (frames - 1) * frame_step + frame_length
    output_shape = [shape_list[i] for i in range(len(shape_list) - 2)]
    output_shape.append(output_size)
    outp = torch.zeros(output_shape, device=inp.device)
    for j in range(frame_length):
        for index in range(frames):
            for i in range(output_size):
                if j + frame_step * index == i:
                    outp[..., i] += inp[..., index, j]
    if output_type != 'float16':
        scale_value = self.params['scale_value']
        shift_value = self.params['shift_value']
        outp = linear_requantize(outp, scale_value, shift_value, output_type, self.outputs[0].quantization.offset)
    outp = outp.half() if self.outputs[0].dtype.is_float else outp
    self.outputs[0].betensor = outp
