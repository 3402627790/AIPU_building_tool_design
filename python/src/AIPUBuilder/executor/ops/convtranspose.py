from ..engine import register_gt, register_gen_weight, _GEN_WEIGHTS
from AIPUBuilder.core import OpType, Dtype
import torch
import numpy as np
from .gt_util.aiff import _aiff_compression_bias_combin, _aiff_merge_shifts, use_ptp_or_mtp
from .gt_util.layout_transfer import layout_convert
from .gt_util.aiff import _aiif_ahead_shift_biases, _aiif_ahead_shift_biases_for_merge
from .gt_util.aiff import _aiff_inner_activation, _aiff_final_output
from .gt_util.aiff import clip_to_bit, x3_aiff_fp32_2_fpx
from .gt_util.layout_transfer import check_in_datalayout, check_out_datalayout, transfer_shape
from .gt_util.quant_utils import fp_activation
from AIPUBuilder.core import Target


@register_gt(OpType.ConvTranspose)
def convtranspose(self, *args, **kwargs):
    self.sgattrs = kwargs['sgattrs'] if 'sgattrs' in kwargs else None
    if self.sgattrs is None:
        self.sgattrs = dict()
    fuse = 'fuse' in kwargs.keys() and kwargs['fuse']

    if self.inputs[0].dtype.is_float:
        return fp_deconv(self, fuse=fuse)

    inp = self.inputs[0].betensor.long()
    inp = check_in_datalayout(self, [inp], ['NHWC'])[0]

    stride = (self.params["stride_y"], self.params["stride_x"])
    dilation = (self.params['dilation_y'], self.params['dilation_x'])
    padding = (self.params['pad_left'], self.params['pad_right'],
               self.params['pad_top'], self.params['pad_bottom'])
    group = self.params['group']
    output_shape = self.outputs[0].shape
    if self.outputs[0].shape.layout.name != 'NHWC':
        output_shape = transfer_shape(output_shape, self.outputs[0].shape.layout.name, 'NHWC')

    output_shape = [
        output_shape[0],
        output_shape[1] + padding[2] + padding[3],
        output_shape[2] + padding[0] + padding[1],
        output_shape[3]
    ]
    bias_type = self.constants['biases'].betensor_type
    biases = _aiff_compression_bias_combin(self.constants['biases'].betensor.type(bias_type))
    weights = self.constants['weights'].betensor.long()
    oc = weights.shape[0]*group
    pad_val = -self.inputs[0].quantization.offset
    if group == 1 and 'shift_value' in self.params.keys():
        shift = self.params['shift_value']
    else:
        shift = self.constants['shift'].betensor

    if 'scale_value' in self.params.keys():
        scale = self.params['scale_value']
    else:
        scale = self.constants['scale'].betensor
    # print(self.name)
    if isinstance(scale, torch.Tensor) and isinstance(shift, torch.Tensor):
        scale, shift = _aiff_merge_shifts(scale, shift.long())
        shift = shift
        full_scale = []
        for s in scale:
            # for 5 dim c32 format, self.weights_shape[0]*self.group = out_c
            full_scale += [s] * (oc // len(scale))
        scale = torch.tensor(full_scale)

    if 'merge_mtp0_c' in self.sgattrs and 'merge_mtp1_c' in self.sgattrs \
            and group == 1 \
            and pad_val == 0:
        # TODO: group conv
        # TODO: asym bottom
        self.add_bias_after_merge = self.sgattrs['add_bias_after_merge'] if 'add_bias_after_merge' in self.sgattrs else False
        merge_mtp0_c = self.sgattrs['merge_mtp0_c']
        merge_mtp1_c = self.sgattrs['merge_mtp1_c']

    # asym
    if pad_val != 0:
        # TODO group deconv support asym
        assert self.params['group'] == 1

        new_pad_l = self.params['kernel_x'] - 1 - self.params['pad_left']
        new_pad_r = self.params['kernel_x'] - 1 - self.params['pad_right']
        new_pad_t = self.params['kernel_y'] - 1 - self.params['pad_top']
        new_pad_b = self.params['kernel_y'] - 1 - self.params['pad_bottom']

        new_fm_h = inp.shape[1] + (inp.shape[1] - 1) * (stride[0] - 1)
        new_fm_w = inp.shape[2] + (inp.shape[2] - 1) * (stride[1] - 1)
        #[b, h, w, c]
        new_input_tensor = torch.zeros((inp.shape[0], new_fm_h, new_fm_w, inp.shape[3])).fill_(pad_val)
        for i in range(inp.shape[1]):
            for j in range(inp.shape[2]):
                new_input_tensor[:, i*stride[0], j*stride[1], :] = inp[:, i, j, :]
        new_input_tensor = torch.nn.functional.pad(new_input_tensor,
                                                   (0, 0, new_pad_l, new_pad_r, new_pad_t, new_pad_b, 0, 0),
                                                   mode='constant',
                                                   value=pad_val)

        new_inp = new_input_tensor.permute([0, 3, 1, 2]).type(torch.float64)

        new_weights = torch.zeros(weights.shape)
        for i in range(self.params['kernel_y']):
            for j in range(self.params['kernel_x']):
                new_weights[:, self.params['kernel_y']-1-i, self.params['kernel_x']-1-j, :] = weights[:, i, j, :]
        # weight: [woc//self.group, ky, kx, wic] ==>>[oc, ic/groups, k_y, k_x]
        new_weights = new_weights.permute([0, 3, 1, 2]).type(torch.float64)
        outputs = torch.nn.functional.conv2d(new_inp,
                                             new_weights,
                                             bias=None,
                                             stride=[1, 1],
                                             padding=0,
                                             dilation=dilation)
        outputs = layout_convert(outputs, _from='NCHW', to='NHWC')

        outputs, remain_shift = _aiif_ahead_shift_biases(self, outputs, origin_shift=shift, biases=biases)
    else:
        split_oc = None
        if 'split_oc' in self.sgattrs:
            split_oc = self.sgattrs['split_oc']
        if split_oc is None or len(split_oc) == 0:
            split_oc = [weights.shape[0] * group]
        wsoes = list(torch.split(weights, [oc // group for oc in split_oc], dim=0))
        bsoes = list(torch.split(biases, split_oc, dim=0))

        # [oc/groups, k_y, k_x, ic]->[ic, oc/groups, k_y, k_x]
        wsoes = [wso.permute([3, 0, 1, 2]).type(torch.float64) for wso in wsoes]

        # # input: [NHWC]->[NCHW]
        inp = layout_convert(inp, _from='NHWC', to='NCHW')

        x0 = [None] * len(split_oc)
        x1 = [None] * len(split_oc)
        for i, oc in enumerate(split_oc):
            rounding_when_16bit_merge = False
            if 'rounding_when_16bit_merge' in self.sgattrs:
                rounding_when_16bit_merge = self.sgattrs['rounding_when_16bit_merge'][i]
            if rounding_when_16bit_merge:
                inps = torch.split(inp, (merge_mtp0_c, merge_mtp1_c), dim=1)
                wsies = torch.split(wsoes[i], (merge_mtp0_c, merge_mtp1_c), dim=0)

                inps = [_inp.float().to('cuda') if torch.cuda.is_available() else _inp.long().to('cpu') for _inp in inps]
                wsies = [wsi.float().to('cuda') if torch.cuda.is_available() else wsi.long().to('cpu') for wsi in wsies]

                x0[i] = torch.nn.functional.conv_transpose2d(inps[0], wsies[0],
                                                            None,
                                                            stride=stride,
                                                            groups=group,
                                                            dilation=dilation)
                # outputs [nchw]->[n,h,w,c]
                x0[i] = layout_convert(x0[i], _from='NCHW', to='NHWC')
                x0[i] = x0[i][
                    :,
                    padding[2]: output_shape[1] - padding[3],
                    padding[0]: output_shape[2] - padding[1],
                    :,
                ]

                x1[i] = torch.nn.functional.conv_transpose2d(inps[1], wsies[1],
                                                            None,
                                                            stride=stride,
                                                            groups=group,
                                                            dilation=dilation)
                # outputs [nchw]->[n,h,w,c]
                x1[i] = layout_convert(x1[i], _from='NCHW', to='NHWC')
                x1[i] = x1[i][
                    :,
                    padding[2]: output_shape[1] - padding[3],
                    padding[0]: output_shape[2] - padding[1],
                    :,
                ]
            else:
                # gpu do not support long/double in torch.conv_transpose2d
                inp = inp.float().to('cuda') if torch.cuda.is_available() else inp.long().to('cpu')
                wsoes[i] = wsoes[i].float().to('cuda') if torch.cuda.is_available() else wsoes[i].long().to('cpu')
                x0[i] = torch.nn.functional.conv_transpose2d(inp, wsoes[i],
                                                             None,
                                                             stride=stride,
                                                             groups=group,
                                                             dilation=dilation)
                # outputs [nchw]->[n,h,w,c]
                x0[i] = layout_convert(x0[i], _from='NCHW', to='NHWC')
                x0[i] = x0[i][
                    :,
                    padding[2]: output_shape[1] - padding[3],
                    padding[0]: output_shape[2] - padding[1],
                    :,
                ]

            if rounding_when_16bit_merge:
                x0[i], remain_shift = _aiif_ahead_shift_biases_for_merge(
                    self, x0[i], x1[i], origin_shift=shift, biases=bsoes[i])
            else:
                x0[i], remain_shift = _aiif_ahead_shift_biases(self, x0[i], origin_shift=shift, biases=bsoes[i])
        if len(x0) != 1:
            outputs = torch.cat(x0, axis=3)
        else:
            outputs = x0[0]

    outputs, final_scale, final_shift = _aiff_inner_activation(self, outputs, scale, remain_shift)
    # final doquant and -zp
    outputs = _aiff_final_output(self, outputs, final_scale, final_shift)
    outputs = clip_to_bit(outputs, 32, True)
    skip_clamp = False

    if fuse:
        if len(self.children) > 0:
            skip_clamp = True

    if not skip_clamp:
        o_type = str(self.outputs[0].dtype)
        o_bit = eval(o_type.split('int')[-1])
        sign = 'u' not in o_type
        outputs = clip_to_bit(outputs, o_bit, sign=sign, to_type=o_type)
    outputs = check_out_datalayout(self, [outputs], ['NHWC'])[0]
    self.outputs[0].betensor = outputs
    return outputs


@register_gen_weight(OpType.ConvTranspose)
def gen_random_weight(self):
    _GEN_WEIGHTS['_all'](self)
    if 'scale' in self.constants:
        self.constants['scale'].betensor = np.random.randint(1, 150, self.constants['scale'].shape,
                                                             self.constants['scale'].dtype.np)
    if 'shift' in self.constants:
        self.constants['shift'].betensor = np.random.randint(10, 20, self.constants['shift'].shape,
                                                             self.constants['shift'].dtype.np)

def fp_deconv(self, fuse=False):
    stride = (self.params['stride_y'], self.params['stride_x'])
    dilation = (self.params['dilation_y'], self.params['dilation_x'])
    padding = (self.params['pad_left'], self.params['pad_right'],
               self.params['pad_top'], self.params['pad_bottom'])

    inputs = self.inputs[0].betensor.to(torch.float32)
    inputs = check_in_datalayout(self, [inputs], ['NHWC'])[0]
    inputs = layout_convert(inputs, _from='NHWC', to='NCHW').float()
    weights = self.constants['weights'].betensor.to(torch.float32)
    weights = weights.permute([3, 0, 1, 2]).type(torch.float32)
    biases = self.constants['biases'].betensor.to(torch.float32)

    output_shape = self.outputs[0].shape
    if self.outputs[0].shape.layout.name != 'NHWC':
        output_shape = transfer_shape(output_shape, self.outputs[0].shape.layout.name, 'NHWC')
    output_shape = [
        output_shape[0],
        output_shape[1] + padding[2] + padding[3],
        output_shape[2] + padding[0] + padding[1],
        output_shape[3]
    ]
    output = torch.nn.functional.conv_transpose2d(inputs,
                                                  weights,
                                                  bias=biases,
                                                  stride=stride,
                                                  groups=self.params['group'],
                                                  dilation=dilation)
    output = output.permute([0, 2, 3, 1])
    output = output[
                :,
                padding[2]: output_shape[1] - padding[3],
                padding[0]: output_shape[2] - padding[1],
                :,
            ]
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