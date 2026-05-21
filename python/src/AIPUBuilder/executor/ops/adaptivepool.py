import torch
from ..engine import register_gt
from AIPUBuilder.core import OpType, Dtype
from .gt_util.quant_utils import get_scale_approximation_params, linear_requantize
from .gt_util.dtype_utils import dtype2range
from .gt_util.layout_transfer import layout_convert


@register_gt(OpType.AdaptivePool)
def adaptivepool(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float:
        return fp_adaptivepool(self)
    method = self.params['method'].upper()
    inp = self.inputs[0].betensor
    out = self.outputs[0]

    is4dim = True if inp.ndim == 4 else False
    in_h, in_w = inp.shape[1:3] if is4dim else inp.shape[0:2]
    out_h, out_w = self.params['output_size']#out.ir_shape[1:3] if is4dim else out.ir_shape[0:2]

    channel = inp.shape[-1]
    out_shape = (inp.shape[0], out_h, out_w, channel) if is4dim else (1, out_h, out_w, channel)
    if not is4dim:
        inp = torch.unsqueeze(inp, dim = 0)
    output = torch.zeros(out_shape)

    h_start = torch.arange(out_h).int() // out_h * in_h + (torch.arange(out_h).int() % out_h) * in_h // out_h
    w_start = torch.arange(out_w).int() // out_w * in_w + (torch.arange(out_w).int() % out_w) * in_w // out_w
    h_end = 1 + ((torch.arange(out_h).int() + 1) * in_h - 1) // out_h
    w_end = 1 + ((torch.arange(out_w).int() + 1) * in_w - 1) // out_w
    mult_bits = 8 # the same as avg pool
    if method == 'AVG':
        kenel_dict = {}
        for oh in range(out_h):
            for ow in range(out_w):
                tmp_data = inp[:, h_start[oh]:h_end[oh], w_start[ow]:w_end[ow], :]
                tmp_sum = torch.sum(tmp_data, dim=[1, 2])
                kenel_area = (h_end[oh] - h_start[oh]) * (w_end[ow] - w_start[ow])
                if kenel_area not in kenel_dict:
                    do_scale, do_scale_type, do_shift, do_shift_type = get_scale_approximation_params(1. / kenel_area, mult_bits)
                    kenel_dict[kenel_area] = (do_scale, do_shift)
                else:
                    do_scale, do_shift = kenel_dict[kenel_area]
                dtype = self.inputs[0].dtype
                output[:, oh, ow, :] = linear_requantize(tmp_sum, do_scale, do_shift, self.inputs[0].dtype, 0)

    elif method == 'MAX':
        inp = inp.float()
        inp =  torch.permute(inp, [0, 3, 1, 2])
        func = torch.nn.AdaptiveMaxPool2d((out_h, out_w), return_indices=False)
        output = func(inp)
        output = torch.permute(output, [0, 2, 3, 1])
    else:
        raise ValueError("Invalid method, adaptivepool only support max/avg method")

    self.outputs[0].betensor = output if is4dim else output[0]

    return self.outputs[0].betensor

def fp_adaptivepool(self):
    method = self.params['method'].upper()
    inp = self.inputs[0].betensor
    is4dim = True if inp.ndim == 4 else False
    if not is4dim:
        inp = torch.unsqueeze(inp, dim = 0)

    inp =  torch.permute(inp, [0, 3, 1, 2])
    out_h, out_w = self.params['output_size']
    if method == 'MAX':
        func = torch.nn.AdaptiveMaxPool2d((out_h, out_w))
    elif method == 'AVG':
        func = torch.nn.AdaptiveAvgPool2d((out_h, out_w))
    else:
        raise ValueError("Invalid method, adaptivepool only support max/avg method")
    output = func(inp)
    output = torch.permute(output, [0, 2, 3, 1])
    self.outputs[0].betensor = output if is4dim else output[0]
    return self.outputs[0].betensor