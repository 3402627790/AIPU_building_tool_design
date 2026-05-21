from ..engine import register_gt, register_gen_weight, _GEN_WEIGHTS
from AIPUBuilder.core import OpType
import torch
import numpy as np
from .gt_util.aiff import _aiff_compression_bias_combin
from .gt_util.layout_transfer import layout_convert
from .gt_util.aiff import _aiif_ahead_shift_biases
from .gt_util.aiff import _aiff_inner_activation, _aiff_final_output
from AIPUBuilder.core import Target
from .gt_util.aiff import clip_to_bit


@register_gt(OpType.ConvTranspose3D)
def convtranspose3d(self, *args, **kwargs):
    self.sgattrs = []
    if 'sgattrs' in kwargs:
        self.sgattrs = kwargs['sgattrs']
    inp = self.inputs[0].irtensor.long()
    weights = self.constants['weights'].betensor.long()
    strides = [self.params['stride_z'], self.params['stride_y'], self.params['stride_x']]
    dilations = [self.params['dilation_z'], self.params['dilation_y'], self.params['dilation_x']]
    bias_type = self.constants['biases'].betensor_type
    biases = _aiff_compression_bias_combin(self.constants['biases'].betensor.type(bias_type))
    inp = layout_convert(inp, _from='NDHWC', to='NCDHW')
    # [woc, ky, kx, kz, wic] ==>> [wic, woc, kz, ky, kx]
    weights = torch.permute(weights, (4, 0, 3, 1, 2))
    output = torch.nn.functional.conv_transpose3d(inp.type(torch.float64),
                                                  weights.type(torch.float64),
                                                  stride=strides,
                                                  dilation=dilations)
    d, h, w = output.shape[-3:]
    output = layout_convert(output, _from='NCDHW', to='NDHWC')

    output = output[
        :,
        self.params['pad_z_begin']: d - self.params['pad_z_end'],
        self.params['pad_y_begin']: h - self.params['pad_y_end'],
        self.params['pad_x_begin']: w - self.params['pad_x_end'],
        :,
    ]
    scale, shift = [self.params['scale_value'],
                    self.params['shift_value']]
    output, remain_shift = _aiif_ahead_shift_biases(self, output, origin_shift=shift, biases=biases)
    output, final_scale, final_shift = _aiff_inner_activation(self, output, scale, remain_shift)
    output = _aiff_final_output(self, output, final_scale, final_shift)
    o_type = str(self.outputs[0].dtype)
    o_bit = eval(o_type.split('int')[-1])
    sign = 'u' not in o_type
    output = clip_to_bit(output, o_bit, sign=sign, to_type=o_type)
    self.outputs[0].irtensor = output
    return output


@register_gen_weight(OpType.ConvTranspose3D)
def gen_random_weight(self):
    _GEN_WEIGHTS['_all'](self)
    in_dtype = str(self.inputs[0].dtype)
    if not in_dtype.startswith('float'):
        self.constants['biases'].betensor = np.random.randint(-1000, 1000, size=self.constants['biases'].shape,
                                                              dtype=str(self.constants['biases'].dtype))
    if 'scale' in self.constants:
        self.constants['scale'].betensor = np.random.randint(1, 150, self.constants['scale'].shape,
                                                             str(self.constants['scale'].dtype))
    if 'shift' in self.constants:
        self.constants['shift'].betensor = np.random.randint(10, 20, self.constants['shift'].shape,
                                                             str(self.constants['shift'].dtype))
