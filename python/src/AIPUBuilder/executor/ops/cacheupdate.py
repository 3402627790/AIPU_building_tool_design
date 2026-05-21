import numpy as np
import torch
from AIPUBuilder.core import Target, OpType
from ..engine import register_gt, register_gen_input
from .gt_util.layout_transfer import layout_convert


@register_gt(OpType.CacheUpdate)
def cacheupdate(self, *args, **kwargs):
    blk_idx = self.params['block_index']
    axis = 2
    if 'axis' in self.params:
        axis = self.params['axis']
    assert axis in [2, 3]
    buf = self.inputs[0].betensor
    is_3dim_buf = len(buf.shape) == 3
    if is_3dim_buf:
        buf = buf.reshape(1, *buf.shape)
    idx = self.inputs[1].betensor.long()
    inp = self.inputs[2].betensor.type_as(buf)

    out = buf[blk_idx]
    max_len = buf.shape[2]
    bs = idx.shape[0]
    layout = self.inputs[2].shape.layout.name

    if layout != 'NHWC':
        n, c, h, w, cx = inp.shape
        out_shape = (n, max_len // cx, h, w, cx) if axis == 3 \
            else (n, c, h, max_len, cx)
        out = layout_convert(out.reshape(out_shape), layout, "NHWC")
        inp = layout_convert(inp, layout, "NHWC")
    else:
        out_shape = (*inp.shape[:-1], max_len) if axis == 3 \
            else (*inp.shape[:-2], max_len, -1)
        out = out.reshape(out_shape)

    inp = inp.reshape(bs, -1, *inp.shape[1:])
    out = out.reshape(bs, -1, *out.shape[1:])

    if axis == 3:
        if layout != 'NHWC':
            for i in range(bs):
                l = idx[i, 0]
                h = l + (inp.shape[4] - l % inp.shape[4])
                out[i, :, :, :, l: h] = inp[i, :, :, :, :h - l]
        else:
            for i in range(bs):
                out[i, :, :, :, idx[i, 0]: idx[i, 0] + inp.shape[3]] = inp[i]
    else:
        for i in range(bs):
            out[i, :, :, idx[i, 0]: idx[i, 0] + inp.shape[3], :] = inp[i]

    out = out.reshape(-1, *out.shape[2:]).detach()
    if layout != 'NHWC':
        out = layout_convert(out, "NHWC", layout)

    if is_3dim_buf:
        self.inputs[0].betensor = out.reshape(buf.shape[1:])
    else:
        self.inputs[0].betensor[blk_idx] = out.reshape(buf.shape[1:])
    self.outputs[0].betensor = out
    self.outputs[1].betensor = self.inputs[0].betensor
    return [x.betensor for x in self.outputs]


@register_gen_input(OpType.CacheUpdate)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_shape2 = list(self.inputs[2].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)
    in_type2 = str(self.inputs[2].dtype)

    max_len = in_shape0[-2]
    max_idx = max_len - in_shape2[-2]

    if self.inputs[0].dtype.is_float:
        input0 = (np.random.randn(*in_shape0) * 1000).astype(in_type0)
        input2 = (np.random.randn(*in_shape2) * 1000).astype(in_type2)
    else:
        input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)
        input2 = np.random.randint(np.iinfo(in_type2).min, np.iinfo(in_type2).max, size=in_shape2, dtype=in_type2)
    input1 = np.random.randint(0, max_idx, size=in_shape1, dtype=in_type1)

    input_data = [input0, input1, input2]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    self.inputs[2].betensor = input_data[2]
    return input_data
