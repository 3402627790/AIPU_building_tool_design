import torch
import numpy as np
from ..engine import register_gt
from AIPUBuilder.core import OpType
from .gt_util.quant_utils import linear_requantize


@register_gt(OpType.Dilation)
def dilation(self, *args, **kwargs):
    input_data = self.inputs[0].betensor.double()
    weights = self.constants['weights'].betensor.clone().double().permute([3, 1, 2, 0])  # [C,H,W,1]=>[1,H,W,C]

    batch = input_data.shape[0]
    oh = self.outputs[0].shape[1]
    ow = self.outputs[0].shape[2]
    oc = self.outputs[0].shape[3]

    kernel_size = self.params['kernel_y'], self.params["kernel_x"]
    stride = self.params["stride_y"], self.params["stride_x"]
    dilation = self.params["dilation_y"], self.params["dilation_x"]
    pad_left = self.params['pad_left']
    pad_right = self.params['pad_right']
    pad_top = self.params['pad_top']
    pad_bottom = self.params['pad_bottom']

    padding = (0, 0,  # c
               pad_left, pad_right,  # w
               pad_top, pad_bottom  # h
               )

    inp_zerop = self.inputs[0].quantization.offset
    input_data = input_data + inp_zerop
    input_data = torch.nn.functional.pad(input_data, padding, value=torch.finfo(torch.double).min)

    out = torch.zeros([batch, oh, ow, oc], device=input_data.device)
    weights = torch.repeat_interleave(weights, batch, dim=0)
    startX = torch.arange(ow) * stride[1]
    startY = torch.arange(oh) * stride[0]
    endY = startY + (kernel_size[0]-1)*dilation[0]+1
    endX = startX + (kernel_size[1]-1)*dilation[1]+1

    scale, scale0, scale1 = self.params["scale_value"]
    shift = self.params["shift_value"]
    input_data = input_data * scale0
    weights = weights * scale1
    out_zerop = self.outputs[0].quantization.offset

    for cy in range(oh):
        for cx in range(ow):
            tmp_data = input_data[:, startY[cy]:endY[cy]:dilation[0], startX[cx]:endX[cx]:dilation[1], :]
            tmp_data = tmp_data + weights
            max_value = torch.amax(tmp_data, (1, 2))
            out[:, cy, cx, :] = max_value

    out = linear_requantize(out, scale, shift, str(self.outputs[0].dtype), out_zerop)

    self.outputs[0].betensor = out
