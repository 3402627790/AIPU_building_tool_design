from ..engine import register_gt, register_gen_weight, _GEN_WEIGHTS
from AIPUBuilder.core import OpType
import torch
import numpy as np
from .gt_util.aiff import _aiff_compression_bias_combin, _aiff_merge_shifts
from .gt_util.layout_transfer import layout_convert
from .gt_util.aiff import _aiif_ahead_shift_biases
from .gt_util.aiff import _aiff_inner_activation, _aiff_final_output
from AIPUBuilder.core import Target
from .gt_util.aiff import clip_to_bit


@register_gt(OpType.Convolution3D)
def convolution3d(self, *args, **kwargs):
    self.sgattrs = kwargs['sgattrs'] if 'sgattrs' in kwargs else None
    if self.sgattrs is None:
        self.sgattrs = dict()

    inp = self.inputs[0].irtensor
    if self.inputs[0].dtype.is_float:
        inp = inp.to(torch.float32)
        biases = self.constants['biases'].betensor.to(torch.float32)
        weights = self.constants['weights'].betensor.to(torch.float32)
    else:
        inp = inp.long()
        biases = _aiff_compression_bias_combin(self.constants['biases'].betensor)
        weights = self.constants['weights'].betensor.long()
    group = self.params['group']
    stride = (self.params["stride_z"], self.params["stride_y"], self.params["stride_x"])
    dilation = (self.params['dilation_z'], self.params['dilation_y'], self.params['dilation_x'])

    padding = (self.params['pad_x_begin'], self.params['pad_x_end'],
               self.params['pad_y_begin'], self.params['pad_y_end'],
               self.params['pad_z_begin'], self.params['pad_z_end']
               )
    if self.inputs[0].dtype.is_float:
        pad_val = .0
    else:
        pad_val = -self.inputs[0].quantization.offset
    # inp:[b, d, h, w, c]==>>[b, c, d, h, w]
    inp = layout_convert(inp, _from='NDHWC', to='NCDHW')
    if self.inputs[0].dtype.is_float:
        inp = inp.float()
    inp = torch.nn.functional.pad(inp, padding, value=pad_val)
    # [woc, ky, kx, kz, wic] ==>> [woc, wic, kz, ky, kx]
    weights = torch.permute(weights, [0, 4, 3, 1, 2])

    # gpu do not support long/double in torch.conv3d
    if not self.inputs[0].dtype.is_float:
        inp = inp.float().to('cuda') if torch.cuda.is_available() else inp.double().to('cpu')
        weights = weights.float().to(
            'cuda') if torch.cuda.is_available() else weights.double().to('cpu')

    output = torch.nn.functional.conv3d(inp, weights, bias=None,
                                        stride=stride,
                                        padding=0,
                                        dilation=dilation,
                                        groups=group)
    output = layout_convert(output, _from='NCDHW', to='NDHWC')
    if self.inputs[0].dtype.is_float:
        output = output.float()
    # output = output.cpu().numpy().astype('int64')

    if self.inputs[0].dtype.is_float:
        output = output + biases
        if 'with_activation' in self.params.keys():
            with_activation = self.params['with_activation']
        else:
            with_activation = 'None'

        if with_activation.upper() == 'RELU':
            act_func = torch.nn.ReLU()
        elif with_activation.upper() == 'RELU6':
            act_func = torch.nn.ReLU6()
        elif with_activation.upper() == 'LEAKYRELU':
            if 'negative_slope' in self.params.keys():
                negative_slope = self.params['negative_slope']
            elif 'negative_slope_value' in self.params.keys():
                negative_slope = self.params['negative_slope_value']
            else:
                negative_slope = 1
            act_func = torch.nn.LeakyReLU(negative_slope)
        elif with_activation.upper() == 'PRELU':
            nslope = self.constants['negative_slope'].betensor
            act_func = torch.nn.PReLU(nslope)
        if with_activation.upper() == 'CLIP':
            clip_min = self.params['clip_min']
            clip_max = self.params['clip_max']
            output = torch.clip(output, clip_min, clip_max)
        elif with_activation.upper() != 'NONE':
            output = act_func(output)
    else:
        if 'scale_value' in self.params.keys():
            scale, shift = [self.params['scale_value'],
                            self.params['shift_value']]
        else:
            scale = self.constants['scale'].betensor.long()
            shift = self.constants['shift'].betensor.long()
            scale, shift = _aiff_merge_shifts(scale, shift)
        output, remain_shift = _aiif_ahead_shift_biases(self, output, origin_shift=shift, biases=biases)
        output, final_scale, final_shift = _aiff_inner_activation(self, output, scale, remain_shift)
        # final doquant and -zp
        output = _aiff_final_output(self, output, final_scale, final_shift)
        o_type = str(self.outputs[0].dtype)
        o_bit = eval(o_type.split('int')[-1])
        sign = 'u' not in o_type
        output = clip_to_bit(output, o_bit, sign=sign, to_type=o_type)
    self.outputs[0].irtensor = output
    return output


@register_gen_weight(OpType.Convolution3D)
def gen_random_weight(self):
    _GEN_WEIGHTS['_all'](self)
    if not self.inputs[0].dtype.is_float:
        self.constants['biases'].betensor = np.random.randint(-1000, 1000, size=self.constants['biases'].shape,
                                                              dtype=str(self.constants['biases'].dtype))
    if 'scale' in self.constants:
        self.constants['scale'].betensor = np.random.randint(1, 150, self.constants['scale'].shape,
                                                             str(self.constants['scale'].dtype))
    if 'shift' in self.constants:
        self.constants['shift'].betensor = np.random.randint(10, 20, self.constants['shift'].shape,
                                                             str(self.constants['shift'].dtype))
