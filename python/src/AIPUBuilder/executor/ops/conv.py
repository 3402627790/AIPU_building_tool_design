from ..engine import register_gt, register_gen_weight, _GEN_WEIGHTS, _OP_METRICS, register_op_metric
from AIPUBuilder.core import OpType, Dtype
import torch
from .gt_util.aiff import _aiff_compression_bias_combin, _aiff_merge_shifts
from .gt_util.layout_transfer import layout_convert
from .gt_util.aiff import _aiif_ahead_shift_biases, _aiif_ahead_shift_biases_for_merge
from .gt_util.aiff import _aiff_inner_activation, _aiff_final_output
from .gt_util.aiff import clip_to_bit, _aiff_wt_dequant, pad_input, x3_aiff_fp32_2_fpx
from AIPUBuilder.core import Target
from .gt_util import check_in_datalayout, check_out_datalayout
from .gt_util.quant_utils import fp_activation
from .gt_util.metric import relative_diff, ulp_diff, cos_distance, check_fp_data
from .gt_util.dtype_utils import torch_to_numpy
from .gt_util.math_utils import fp32_to_fp24
from ml_dtypes import float4_e2m1fn
from .dwconv import depthwiseconv
import numpy as np
import time


@register_gt(OpType.Convolution)
def convolution(self, *args, **kwargs):
    self.sgattrs = kwargs['sgattrs'] if 'sgattrs' in kwargs else None
    if self.sgattrs is None:
        self.sgattrs = dict()
    fuse = True
    if not ('fuse' in kwargs and kwargs['fuse'] and
            (OpType.Eltwise in [c.type for c in self.children] or
             OpType.BatchNorm in [c.type for c in self.children] or
             OpType.Activation in [c.type for c in self.children])):
        fuse = False

    if self.inputs[0].dtype.is_float:
        return fp_conv(self, fuse=fuse)

    if 'is_dwconv' in self.attrs and self.attrs['is_dwconv']:
        x = depthwiseconv(self, *args, **kwargs)
        return x

    inp = self.inputs[0].betensor.long()
    inp = check_in_datalayout(self, [inp], ['NHWC'])[0]

    with_winograd = self.params['with_winograd'] if 'with_winograd' in self.params else False

    stride = (self.params["stride_y"], self.params["stride_x"])
    dilation = (self.params['dilation_y'], self.params['dilation_x'])
    padding = (self.params['pad_left'], self.params['pad_right'],
               self.params['pad_top'], self.params['pad_bottom'])

    bias_type = self.constants['biases'].betensor_type
    time0 = time.time()
    bias = _aiff_compression_bias_combin(torch.from_numpy(self.constants['biases'].np))
    requant_scale = 1
    requant_shift = 0
    if 'scale_value' in self.params:
        requant_scale = self.params['scale_value']
    elif "scale" in self.constants:
        requant_scale = self.constants["scale"].betensor
    if 'shift_value' in self.params:
        requant_shift = self.params['shift_value']
    elif "shift" in self.constants:
        requant_shift = self.constants["shift"].betensor
    if isinstance(requant_scale, torch.Tensor) and isinstance(requant_shift, torch.Tensor):
        requant_scale, requant_shift = _aiff_merge_shifts(requant_scale.long(), requant_shift.long())
    time1 = time.time()
    # print('time bias_compress: ', time1 - time0)
    weights = self.constants['weights']
    group = self.params['group']
    if 'pads_mode' in self.params:  # for simulator
        inp = pad_input(self, inp)
    inp = layout_convert(inp, _from='NHWC', to='NCHW')
    if 'pads_mode' not in self.params:
        pad_val = -self.inputs[0].quantization.offset
        inp = torch.nn.functional.pad(inp, padding, value=pad_val)
    weights = layout_convert(weights.np, _from='NHWC', to='NCHW')
    time2 = time.time()
    # print('time layout_convert: ', time2 - time1)

    split_oc = None
    if 'split_oc' in self.sgattrs:
        split_oc = self.sgattrs['split_oc']
    if split_oc is None or len(split_oc) == 0:
        split_oc = [weights.shape[0]]
    wsoes = list(torch.split(weights, split_oc, dim=0))
    bsoes = list(torch.split(bias, split_oc, dim=0))

    x0 = [None] * len(split_oc)
    x1 = [None] * len(split_oc)
    for i, oc in enumerate(split_oc):
        rounding_when_16bit_merge = False
        if 'rounding_when_16bit_merge' in self.sgattrs \
                and self.sgattrs['rounding_when_16bit_merge'][i] \
                and group == 1 \
                and not with_winograd:
            # TODO: group conv
            # TODO: winograd
            rounding_when_16bit_merge = True
            self.add_bias_after_merge = self.sgattrs['add_bias_after_merge'] if 'add_bias_after_merge' in self.sgattrs else False
            merge_mtp0_c = self.sgattrs['merge_mtp0_c']
            merge_mtp1_c = self.sgattrs['merge_mtp1_c']

            inps = torch.split(inp, (merge_mtp0_c, merge_mtp1_c), dim=1)
            wsies = torch.split(wsoes[i], (merge_mtp0_c, merge_mtp1_c), dim=1)

            inps = [_inp.float().to('cuda') if torch.cuda.is_available() else _inp.long().to('cpu') for _inp in inps]
            wsies = [wsi.float().to('cuda') if torch.cuda.is_available() else wsi.long().to('cpu') for wsi in wsies]

            x0[i] = torch.nn.functional.conv2d(inps[0], wsies[0],
                                               bias=None,
                                               stride=stride,
                                               padding=0,
                                               groups=group,
                                               dilation=dilation)
            x0[i] = layout_convert(x0[i], _from='NCHW', to='NHWC')
            x1[i] = torch.nn.functional.conv2d(inps[1], wsies[1],
                                               bias=None,
                                               stride=stride,
                                               padding=0,
                                               groups=group,
                                               dilation=dilation)
            x1[i] = layout_convert(x1[i], _from='NCHW', to='NHWC')

            # ir_out_shape = x0[i].shape  # x0[i] and x1[i] shape are same
        else:
            # gpu do not support long/double in torch.conv_transpose2d
            inp = inp.float().to('cuda') if torch.cuda.is_available() else inp.long().to('cpu')
            wsoes[i] = wsoes[i].float().to('cuda') if torch.cuda.is_available() else wsoes[i].long().to('cpu')

            # print('device inp2: ', inp.device)
            # print('device weights2: ', wsoes[i].device)

            time11 = time.time()
            x0[i] = torch.nn.functional.conv2d(inp, wsoes[i],
                                               bias=None,
                                               stride=stride,
                                               padding=0,
                                               dilation=dilation,
                                               groups=group)
            # print('device: ', x0[i].device)
            time3 = time.time()
            # print('time conv2d: ', time3 - time11)
            x0[i] = layout_convert(x0[i], _from='NCHW', to='NHWC')
            time4 = time.time()
            # print('time layout_convert: ', time4 - time3)
            # ir_out_shape = x0[i].shape

        if rounding_when_16bit_merge:
            # print("%s will use merge method" % self.name)
            x0[i], remain_shift = _aiif_ahead_shift_biases_for_merge(
                self, x0[i], x1[i], origin_shift=int(requant_shift), biases=bsoes[i])
        else:
            x0[i], remain_shift = _aiif_ahead_shift_biases(
                self, x0[i], origin_shift=int(requant_shift), biases=bsoes[i])
            x0[i] = clip_to_bit(x0[i], 33, True)  # for itp 33bit m0_alu
        time5 = time.time()
        # print('time _aiif_ahead_shift_biases: ', time5 - time4)
    if len(x0) != 1:
        x = torch.cat(x0, axis=3)
    else:
        x = x0[0]
    ir_out_shape = x.shape

    if isinstance(requant_scale, list):
        full_scale = []
        for s in requant_scale:
            full_scale += [s] * (ir_out_shape[-1] // len(requant_scale))
        requant_scale = full_scale
    x, final_scale, final_shift = _aiff_inner_activation(self, x, requant_scale, remain_shift)
    time6 = time.time()
    # print('time _aiff_inner_activation: ', time6 - time5)
    # final doquant and -zp
    x = _aiff_final_output(self, x, final_scale, final_shift, fuse=fuse)
    x = clip_to_bit(x, 32, True)
    time7 = time.time()
    # print('time _aiff_final_output: ', time7 - time6)
    o_type = str(self.outputs[0].dtype)
    o_bit = eval(o_type.split('int')[-1])
    sign = 'u' not in o_type
    #uint [0, 2**o_bit-1]
    #int: [-2**(obit-1), 2**(o_bit-1)-1]
    if not fuse:
        x = clip_to_bit(x, o_bit, sign=sign, to_type=o_type)
    # else:
    #     print("%s will fuse" % self.name)
    x = check_out_datalayout(self, [x], ['NHWC'])[0]
    time8 = time.time()
    # print('time check_out_datalayout: ', time8 - time7)
    self.outputs[0].betensor = x
    return x


@register_gen_weight(OpType.Convolution)
def gen_random_weight(self):
    _GEN_WEIGHTS['_all'](self)
    in_dtype = str(self.inputs[0].dtype)
    if 'float' not in in_dtype:
        self.constants['biases'].betensor = np.random.randint(-1000, 1000, size=self.constants['biases'].shape,
                                                              dtype=str(self.constants['biases'].dtype))
        if 'scale' in self.constants:
            self.constants['scale'].betensor = np.random.randint(1, 150, self.constants['scale'].shape,
                                                                str(self.constants['scale'].dtype))
        if 'shift' in self.constants:
            self.constants['shift'].betensor = np.random.randint(10, 20, self.constants['shift'].shape,
                                                             str(self.constants['shift'].dtype))


def fp_conv(self, fuse=False):
    stride = (self.params['stride_y'], self.params['stride_x'])
    dilation = (self.params['dilation_y'], self.params['dilation_x'])
    padding = (self.params['pad_left'], self.params['pad_right'],
               self.params['pad_top'], self.params['pad_bottom'])

    inputs = self.inputs[0].betensor.to(torch.float32)
    inputs = check_in_datalayout(self, [inputs], ['NHWC'])[0]
    inputs = layout_convert(inputs, _from='NHWC', to='NCHW').float()
    inputs = torch.nn.functional.pad(inputs, padding, value=0)

    weights = torch.tensor(self.constants['weights'].np.astype(np.float32)).to(inputs.device)  # [oc, kh, kw, ic]
    weights = weights.permute([1, 2, 3, 0])  # [kh, kw, ic, oc]
    # wt dequant
    weights = _aiff_wt_dequant(weights, self.constants, weights.shape[3], weights.shape[2], weights.shape[0], weights.shape[1])
    weights = weights.permute([3, 0, 1, 2])  # [oc, kh, kw, ic]
    weights = layout_convert(weights, _from='NHWC', to='NCHW').float()  # [ic, oc, kh, kw]

    biases = self.constants['biases'].betensor.to(torch.float32)

    output = torch.nn.functional.conv2d(inputs,
                                        weights,
                                        bias=None,
                                        stride=stride,
                                        padding=0,
                                        groups=self.params['group'],
                                        dilation=dilation)
    output = output.permute([0, 2, 3, 1])
    if 'scale1' in self.constants:
        output = output * self.constants['scale1'].betensor.flatten().float()  # [oc]

    output = output + biases
    output = fp_activation(output, self)

    scale = 1.0
    if 'scale' in self.constants:
        scale = self.constants['scale'].betensor.float()  # [oc]
    elif 'scale_value' in self.params.keys():
        scale = self.params['scale_value']
    output = output * scale

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


@register_op_metric(OpType.Convolution)
@check_fp_data
def metric(self, outputs, index=None):
    def call(r, u, c, t):
        if "float" in t:
            if np.all(u <= 1) or np.all(r <= 1e-4):
                return True
            else:
                return np.all(c > 0.999)
        return np.all(r == 0)

    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]

    layer_dtypes = [str(o.dtype) for o in self.outputs] if index is None else [str(self.outputs[index].dtype)]

    rel_diffs = relative_diff(outputs, groundtruths)
    ulp_diffs = ulp_diff(outputs, groundtruths)
    cos_dists = cos_distance(outputs, groundtruths)
    if self.outputs[0].dtype.is_float:
        return all(call(r, u, c, t) for r, u, c, t in zip(rel_diffs, ulp_diffs, cos_dists, layer_dtypes))

    return _OP_METRICS['_all'](self, outputs, index)
