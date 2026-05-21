# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils.quant_tool_utils import *
from AIPUBuilder.Optimizer.utils.dtype_utils import *
from AIPUBuilder.Optimizer.utils.string_utils import *


def _get_crops(node, crops):
    if 'dynamic_symbols' not in node.attrs:
        return crops

    dynamic_symbols_map = node.attrs['dynamic_symbols']
    if 'ds_crops' in node.params:
        crops = ds_expr_eval(node.params.get('ds_crops'), dynamic_symbols_map)
    return crops


def get_crops(node):
    import copy
    crops = copy.deepcopy(node.get_param('crops'))

    ds_parameter_constant = node.get_attrs("ds_parameter_constant", optional=True, default_value='disable')

    if ds_parameter_constant == 'default':
        node.params['ds_crops'] = crops
    else:
        crops = _get_crops(node, crops)
    return crops



@op_register(OpType.Crop)
def crop(self, *args):
    x = self.inputs[0].betensor
    begin_end_index = []
    crops = get_crops(self)
    # batch dim, need crop all batch
    batch_size_in_IR = self.get_attrs("batch_size_in_IR", optional=True, default_value=1)
    current_input_batch_size_in_ir = self.inputs[0].ir_shape[0]

    if batch_size_in_IR != 0 and crops[0] == [0, current_input_batch_size_in_ir]:
        crops[0][-1] = x.shape[0]
    if not isinstance(crops, list) or len(crops) != len(x.shape):
        OPT_ERROR("parameter crop should have length equals to inp rank")

    is_zero_dim = False
    for c_id, c in enumerate(crops):
        idx = c[0]
        per_axis = []
        if c[0] < 0:
            c[0] += x.shape[c_id]
        if c[1] < 0:
            c[1] += x.shape[c_id]
        c[1] = min(x.shape[c_id], c[1])
        if c[0] == c[1]:
            is_zero_dim = True
        else:
            if c[1] > x.shape[c_id] or c[0] > x.shape[c_id]-1:
                OPT_ERROR('layer_id=%s,Crop Op %d-dimension index is illegal, crop index is %s, and input shape is %s, Please Check!' %
                          (str(self.attrs['layer_id']), c_id, str(crops), str(x.shape)))
            for idx in range(c[0], c[1]):
                per_axis.append(idx)
        begin_end_index.append(per_axis)
    if is_zero_dim:
        zero_shape = []
        for index in begin_end_index:
            zero_shape.append(len(index))
        x = torch.zeros(zero_shape, device=x.device) + self.inputs[0].broadcast_zerop
    else:
        for axis in range(len(crops)):
            x = torch.index_select(x, axis, torch.tensor(begin_end_index[axis], device=x.device))

    self.outputs[0].betensor = x
    return x


@quant_register(OpType.Crop)
def crop_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qinvariant = inp.qinvariant
    if out.key_axis is not None and is_torch_tensor_with_multi_data(out.scale):
        crops = self.get_param('crops')[out.key_axis]
        out.scale = out.scale[crops[0]:crops[1]]
        out.zerop = out.zerop[crops[0]:crops[1]]
