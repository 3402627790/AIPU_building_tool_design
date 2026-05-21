import torch
from AIPUBuilder.core import OpType
from .gt_util import *
from .gt_util.quant_utils import _tpc_lut
from ..engine import register_gt, register_gen_input


@register_gt(OpType.MaxUnpool)
@register_gt(OpType.UpsampleByIndex)
def UpsampleByIndex(self, *args, **kwargs):
    """
    get the output according to the input and mask.
    :param
    input feature:     the feature map that will be upsampled
    input mask:        the out_hw_idx indices value used to mask `input` data
    output feature:    the output of upsampled feature map (`input`) by `mask` indices
    """
    flatten_dim = self.params['flatten_dim']
    input_image_data = torch_to_numpy(self.inputs[0].betensor, self.inputs[0].dtype.np).flatten()
    input_mask_data = torch_to_numpy(self.inputs[1].betensor, self.inputs[1].dtype.np).flatten()

    output_image_data = np.zeros(self.outputs[0].shape).flatten() - self.inputs[0].quantization.offset

    in_n, in_h, in_w, in_c = self.inputs[0].shape
    out_n, out_h, out_w, out_c = self.outputs[0].shape

    upsample_hw = out_h * out_w
    upsample_hwc = out_h * out_w * out_c
    upsample_nhwc = out_n * out_h * out_w * out_c

    for nidx in range(in_n):
        for hidx in range(in_h):
            for widx in range(in_w):
                for cidx in range(in_c):
                    mask_idx = nidx * (in_h * in_w * in_c) + (hidx * in_w + widx) * in_c + cidx
                    if mask_idx >= len(input_mask_data):
                        continue
                    mask_val = input_mask_data[mask_idx]
                    if flatten_dim == 'HW':
                        assert mask_val < upsample_hw, "check failed with mask_val < upsample_hw"
                        output_nhwc_idx = nidx * out_h * out_w * out_c + mask_val * in_c + cidx
                    elif flatten_dim == 'HWC':
                        assert mask_val < upsample_hwc, "check failed with mask_val < upsample_hwc"
                        output_nhwc_idx = nidx * out_h * out_w * out_c + mask_val
                    elif flatten_dim in ['NHWC', 'NCHW']:
                        assert mask_val < upsample_nhwc, "check failed with mask_val < upsample_nhwc"
                        output_nhwc_idx = mask_val  # output_nchw_idx actually when NCHW
                    else:
                        raise ValueError('wrong flatten_dim')
                    if output_nhwc_idx >= upsample_nhwc:
                        continue
                    output_image_data[int(output_nhwc_idx)] = input_image_data[mask_idx]

    if flatten_dim == 'NCHW':
        if 'storage_order' in self.params.keys() and self.params['storage_order'] == 1:  # ncwh -> nhwc
            tmp_shape = [out_n, out_c, out_w, out_h]
            output_image_data = output_image_data.reshape(tmp_shape)
            output_image_data = np.transpose(output_image_data, [0, 3, 2, 1])
        else:  # nchw -> nhwc
            tmp_shape = [out_n, out_c, out_h, out_w]
            output_image_data = output_image_data.reshape(tmp_shape)
            output_image_data = np.transpose(output_image_data, [0, 2, 3, 1])
    else:
        if 'storage_order' in self.params.keys() and self.params['storage_order'] == 1:  # nwhc -> nhwc
            tmp_shape = [out_n, out_w, out_h, out_c]
            output_image_data = output_image_data.reshape(tmp_shape)
            output_image_data = np.transpose(output_image_data, [0, 2, 1, 3])
        else:  # nhwc
            output_image_data = output_image_data.reshape(self.outputs[0].shape)

    output_image_data = clamp(output_image_data, self.outputs[0].dtype)
    self.outputs[0].betensor = output_image_data
    return [o.betensor for o in self.outputs]


@register_gen_input(OpType.MaxUnpool)
@register_gen_input(OpType.UpsampleByIndex)
def gen_input_data(self):
    out_shape = list(self.outputs[0].shape)
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)
    flatten_dim = self.params['flatten_dim']

    if self.inputs[0].dtype.is_float:
        input0 = (np.random.uniform(0, 10000, size=in_shape0)).astype(in_type0)
    else:
        input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)

    if flatten_dim == 'HW':
        max_index = out_shape[1] * out_shape[2]
    elif flatten_dim == 'HWC':
        max_index = out_shape[1] * out_shape[2] * out_shape[3]
    elif flatten_dim in ['NHWC', 'NCHW']:
        max_index = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3]
    else:
        raise TypeError('wrong flatten_dim')

    out_hw_idx = np.random.randint(0, max_index, size=in_shape1, dtype=in_type1)

    # the values corresponding to the same index should also be equal
    in_n, in_h, in_w, in_c = in_shape0
    out_n, out_h, out_w, out_c = out_shape
    upsample_nhwc = out_n * out_h * out_w * out_c

    input0 = input0.flatten()
    input_mask_data = out_hw_idx.flatten()

    output_nhwc_idx_dict = {}
    for nidx in range(in_n):
        for hidx in range(in_h):
            for widx in range(in_w):
                for cidx in range(in_c):
                    mask_idx = nidx * (in_h * in_w * in_c) + (hidx * in_w + widx) * in_c + cidx
                    mask_val = input_mask_data[mask_idx]
                    if flatten_dim == 'HW':
                        output_nhwc_idx = nidx * out_h * out_w * out_c + mask_val * in_c + cidx
                    elif flatten_dim == 'HWC':
                        output_nhwc_idx = nidx * out_h * out_w * out_c + mask_val
                    elif flatten_dim in ['NHWC', 'NCHW']:
                        output_nhwc_idx = mask_val  # output_nchw_idx actually when NCHW
                    else:
                        raise TypeError('wrong flatten_dim')
                    if output_nhwc_idx >= upsample_nhwc:
                        continue

                    if output_nhwc_idx in output_nhwc_idx_dict.keys():
                        input0[mask_idx] = input0[output_nhwc_idx_dict[output_nhwc_idx]]  # repeat value, replaced
                    output_nhwc_idx_dict[output_nhwc_idx] = mask_idx
    input0 = input0.reshape(in_shape0)
    input_data = [input0, out_hw_idx]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    return input_data
