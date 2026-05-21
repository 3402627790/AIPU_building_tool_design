from ..engine import register_gt, register_gen_weight, _GEN_WEIGHTS
from AIPUBuilder.core import OpType, Dtype
import torch
import numpy as np
from .gt_util.aiff import _aiff_compression_bias_combin, _aiff_merge_shifts, use_ptp_or_mtp
from .gt_util.layout_transfer import layout_convert, check_in_datalayout, check_out_datalayout
from .gt_util.aiff import _aiif_ahead_shift_biases, x3_aiff_fp32_2_fpx
from .gt_util.aiff import _aiff_inner_activation, _aiff_final_output
from .gt_util.aiff import clip_to_bit, pad_input
# from .conv import fp_conv # most likely due to a circular import
from .gt_util.quant_utils import fp_activation
from AIPUBuilder.core import Target


@register_gt(OpType.DepthwiseConv)
def depthwiseconv(self, *args, **kwargs):
    self.sgattrs = kwargs['sgattrs'] if 'sgattrs' in kwargs else None
    if self.sgattrs is None:
        self.sgattrs = dict()
    fuse = 'fuse' in kwargs.keys() and kwargs['fuse']

    if self.inputs[0].dtype.is_float:
        return fp_dwconv(self, fuse=fuse)

    target_name = Target().name
    inp = self.inputs[0].betensor.long()
    inp = check_in_datalayout(self, [inp], ['NHWC'])[0]
    stride = (self.params["stride_y"], self.params["stride_x"])
    dilation = (self.params['dilation_y'], self.params['dilation_x'])
    padding = (self.params['pad_left'], self.params['pad_right'],
               self.params['pad_top'], self.params['pad_bottom'])
    group = self.params['group']
    oc = self.outputs[0].shape[-1]

    bias_type = self.constants['biases'].betensor_type
    biases = _aiff_compression_bias_combin(self.constants['biases'].betensor.type(bias_type))
    weights = self.constants['weights']
    if 'pads_mode' in self.params:  # for simulator
        inp = pad_input(self, inp)
    inp = layout_convert(inp, _from='NHWC', to='NCHW')
    if 'pads_mode' not in self.params:
        pad_val = -self.inputs[0].quantization.offset
        inp = torch.nn.functional.pad(inp, padding, value=pad_val)
    weights = layout_convert(weights.betensor.long(), _from='NHWC', to='NCHW')
    # gpu do not support long/double in torch.conv2d
    inp = inp.float().to('cuda') if torch.cuda.is_available() else inp.long().to('cpu')
    weights = weights.float().to('cuda') if torch.cuda.is_available() else weights.long().to('cpu')

    outputs = torch.nn.functional.conv2d(inp,
                                         weights,
                                         bias=None,
                                         stride=stride,
                                         padding=0,
                                         dilation=dilation,
                                         groups=group)
    outputs = layout_convert(outputs, _from='NCHW', to='NHWC')

    scales = self.constants["scale"].betensor.long()
    shifts = self.constants["shift"].betensor.long()

    scale, shift = _aiff_merge_shifts(scales, shifts)

    use_ptp = use_ptp_or_mtp(self, target_name, fuse)
    if 'simu_op' in self.params and self.params['simu_op'] == 'ptp':
        use_ptp = True

    remain_shift = shift
    if use_ptp == True:
        outputs = clip_to_bit(outputs, 48, True)
        outputs = outputs.type(torch.int64)
        outputs += biases
    # use mtp
    else:
        outputs, remain_shift = _aiif_ahead_shift_biases(self, outputs, origin_shift=shift, biases=biases)
        # biases added in _aiif_ahead_shift_biases

    full_scale = []
    for s in scale:
        full_scale += [s] * (oc // self.constants["scale"].betensor.shape[0])
    outputs, final_scale, final_shift = _aiff_inner_activation(self, outputs, scale, remain_shift, use_ptp)
    # final doquant and -zp
    outputs = _aiff_final_output(self, outputs, final_scale, final_shift)

    skip_clamp = False

    if fuse:
        if len(self.children) > 0:
            if self.children[0].type.name == 'Eltwise' or \
                    self.children[0].type.name == 'Activation':
                skip_clamp = True

    if not skip_clamp:
        o_type = self.outputs[0].dtype.np
        o_bit = self.outputs[0].dtype.bits
        sign = 'u' not in str(self.outputs[0].dtype)
        outputs = clip_to_bit(outputs, o_bit, sign=sign, to_type=o_type)
    outputs = check_out_datalayout(self, [outputs], ['NHWC'])[0]
    self.outputs[0].betensor = outputs
    return outputs

def fp_dwconv(self, fuse=False):
    stride = (self.params['stride_y'], self.params['stride_x'])
    dilation = (self.params['dilation_y'], self.params['dilation_x'])
    padding = (self.params['pad_left'], self.params['pad_right'],
               self.params['pad_top'], self.params['pad_bottom'])

    inputs = self.inputs[0].betensor.to(torch.float32)
    inputs = check_in_datalayout(self, [inputs], ['NHWC'])[0]
    inputs = layout_convert(inputs, _from='NHWC', to='NCHW').float()
    inputs = torch.nn.functional.pad(inputs, padding, value=0)
    weights = self.constants['weights'].betensor.to(torch.float32)
    weights = layout_convert(weights, _from='NHWC', to='NCHW').float()
    biases = self.constants['biases'].betensor.to(torch.float32)
    output = torch.nn.functional.conv2d(inputs,
                                        weights,
                                        bias=biases,
                                        stride=stride,
                                        padding=0,
                                        groups=self.params['group'],
                                        dilation=dilation)
    output = output.permute([0, 2, 3, 1])
    output = fp_activation(output, self)

    from copy import deepcopy
    bak_out_dtype = deepcopy(self.outputs[0].dtype)
    if fuse:
        self.outputs[0].dtype = Dtype.FP32
    else:
        output = x3_aiff_fp32_2_fpx(output, fpx=self.outputs[0].dtype)
    output = check_out_datalayout(self, [output], ['NHWC'])[0]
    self.outputs[0].betensor = output
    if fuse:
        self.outputs[0].dtype = bak_out_dtype
    return output

@register_gen_weight(OpType.DepthwiseConv)
def gen_random_weight(self):
    _GEN_WEIGHTS['_all'](self)
    in_dtype = str(self.inputs[0].dtype)
    if not in_dtype.startswith(('float', 'bfloat')):
        self.constants['biases'].betensor = np.random.randint(-1000, 1000, size=self.constants['biases'].shape,
                                                              dtype=str(self.constants['biases'].dtype))
        self.constants['scale'].betensor = np.random.randint(1, 150, self.constants['scale'].shape,
                                                            str(self.constants['scale'].dtype))
        self.constants['shift'].betensor = np.random.randint(10, 20, self.constants['shift'].shape,
                                                            str(self.constants['shift'].dtype))
