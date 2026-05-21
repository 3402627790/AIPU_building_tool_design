import torch
from .gt_util import *
from ..engine import register_gt
from AIPUBuilder.core import OpType
from AIPUBuilder.logger import DEBUG


@register_gt(OpType.MaxPoolingWithArgMax)
def MaxPoolingWithArgMax(self, *args, **kwargs):
    inp = self.inputs[0].betensor.float()
    n, h, w, c = inp.shape
    _, out_h, out_w, out_c = self.outputs[0].shape
    kh, kw = (self.params['kernel_y'], self.params['kernel_x'])
    sh, sw = (self.params['stride_y'], self.params['stride_x'])
    pt, pl = (self.params['pad_top'], self.params['pad_left'])
    pb, pr = (self.params['pad_bottom'], self.params['pad_right'])
    dh, dw = (self.params['dilation_y'], self.params['dilation_x'])
    flatten_dim = self.params['flatten_dim']
    ceil_mode = self.params['ceil_mode']
    storage_order = int(self.params['storage_order'])

    padding = (pl, pr, pt, pb)
    input_data = nhwc2nchw(inp)
    input_data = torch.nn.functional.pad(input_data, padding, value=-2. ** 31)

    # default is flatten_dim(HW)
    func = torch.nn.MaxPool2d(kernel_size=(kh, kw), stride=(sh, sw), padding=0,
                              dilation=(dh, dw), return_indices=True, ceil_mode=ceil_mode)
    value, indics = func(input_data)

    pad_h = h + pt + pb
    pad_w = w + pl + pr
    invalid_pad_mask = value == -2. ** 31
    value[invalid_pad_mask] = 0
    indics[invalid_pad_mask] = -(pad_h * pad_w * c * n + 2)

    # conner case under ceil_mode=True CTL-6524
    current_outh, current_outw = value.shape[2: 4]
    if current_outh < out_h or current_outw < out_w:
        DEBUG('MaxPoolingWithArgMax: conner case under ceil_mode=True.')
        auto_pad_h = out_h - current_outh
        auto_pad_w = out_w - current_outw
        padding = (0, auto_pad_w, 0, auto_pad_h)  # left, right, top, bottom
        pad_val = -self.inputs[0].quantization.offset
        value = torch.nn.functional.pad(value, padding, value=pad_val)
        indics = torch.nn.functional.pad(indics, padding, value=-(pad_h * pad_w * c * n + 2)).long()

    H = indics // pad_w
    W = indics % pad_w
    H = H - pt
    W = W - pl
    H = nchw2nhwc(H)
    W = nchw2nhwc(W)
    value = nchw2nhwc(value)

    # row-major when storage_order = 0; column-major when storage_order = 1
    indics = H * w + W
    if storage_order:
        # invalid indics will make H negative
        indics[H >= 0] = h * W[H >= 0] + H[H >= 0]

    if flatten_dim == 'HW':
        pass
    elif flatten_dim == 'HWC':
        for channel in range(indics.shape[3]):
            indics[:, :, :, channel] = indics[:, :, :, channel] * c + channel
    elif flatten_dim == 'NHWC':
        for batch in range(indics.shape[0]):
            for channel in range(indics.shape[3]):
                indics[batch, :, :, channel] = batch * h * w * c + indics[batch, :, :, channel] * c + channel
    elif flatten_dim == 'NCHW':
        for batch in range(indics.shape[0]):
            for channel in range(indics.shape[3]):
                indics[batch, :, :, channel] = h * w * c * batch + h * w * channel + indics[batch, :, :, channel]
    else:
        raise ValueError("unsupported method: %s for MaxPoolingWithArgMax(type)" % (flatten_dim))

    indics[indics < 0] = 0
    self.outputs[0].betensor = value
    self.outputs[1].betensor = indics

    return [o.betensor for o in self.outputs]
