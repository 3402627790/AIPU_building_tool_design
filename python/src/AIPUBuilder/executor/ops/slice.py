from ..engine import register_gt, register_gen_input, _GEN_INPUTS, _OP_METRICS, register_op_metric
from .gt_util.dtype_utils import torch_to_numpy, numpy_to_torch
from .gt_util.metric import check_fp_data
from AIPUBuilder.core import OpType
import torch
import random
import numpy as np


@register_gt(OpType.InternalSlice)
@register_gt(OpType.Slice)
def slice(self, *args, **kwargs):
    if len(self.inputs) > 1:
        ds_slice(self)
    else:
        x = self.inputs[0].irtensor

        strides = self.params['strides']
        begin = self.params['begin']
        end = self.params['end']

        real_shape = list(x.shape)
        for i in range(len(end)):
            end[i] = min(real_shape[i], end[i])

        for i in range(len(begin)):
            if (begin[i] >= real_shape[i] and end[i] >= real_shape[i]) or (
                    begin[i] < -real_shape[i] and end[i] < -real_shape[i]):
                begin[i] = end[i] = real_shape[i]
            else:
                begin[i] = max(0, min(real_shape[i] - 1, begin[i] + real_shape[i] if begin[i] < 0 else begin[i]))
                end[i] = max(-1, min(real_shape[i], end[i] + real_shape[i] if end[i] < 0 else end[i]))

        for axis in range(len(strides)):
            start_index = begin[axis]
            end_index = end[axis]
            stride = strides[axis]
            step = int((abs(end_index - start_index) - 1) / abs(stride))
            actual_index = [start_index + stride * s for s in range(step + 1)]
            actual_index = [] if start_index == end_index else actual_index
            x = x.index_select(axis, torch.tensor(actual_index, device=x.device).int())
        self.outputs[0].irtensor = x
        return x


def ds_slice(self):
    steps = []
    begin = []
    end = []
    dims = len(self.inputs[0].shape)
    axes = self.inputs[3].betensor
    axes = torch.where(axes < torch.zeros_like(axes), axes + dims, axes).clamp(0, dims - 1).tolist()
    for d in range(dims):
        predefined = False
        for ad in range(len(axes)):
            if d == axes[ad]:
                steps.append(int(self.inputs[4].betensor[ad]))
                begin.append(int(self.inputs[1].betensor[ad]))
                end.append(int(self.inputs[2].betensor[ad]))
                predefined = True
                break
        if not predefined:
            steps.append(1)
            begin.append(0)
            end.append(self.inputs[0].shape[d])
    self.params['strides'] = steps
    self.params['begin'] = begin
    self.params['end'] = end

    out = stridedslice(self)
    self.outputs[1].betensor = torch.tensor(out.shape, device=self.outputs[0].betensor.device)
    if '_output_shapes' not in self.attrs:  # without --enable_ds, pad 0, CP-20756
        out_pad = self.outputs[0].betensor.flatten()
        out_pad[: out.numel()] = out.flatten()
        out = out_pad.reshape(self.outputs[0].betensor.shape)
    self.outputs[0].betensor = out


def stridedslice(self):
    x = self.inputs[0].betensor
    strides = self.params['strides']
    begin = self.params['begin']
    end = self.params['end'][:]

    real_shape = list(x.shape)
    for i in range(len(end)):
        end[i] = min(real_shape[i], end[i])
    # upper_bond = self.params.get('upper_bound', False)
    # input_shape = self.inputs[0].shape
    for i in range(len(begin)):
        if (begin[i] >= real_shape[i] and end[i] >= real_shape[i]) or (begin[i] < -real_shape[i] and end[i] < -real_shape[i]):
            begin[i] = end[i] = real_shape[i]
        else:
            begin[i] = max(0, min(real_shape[i] - 1, begin[i] + real_shape[i] if begin[i] < 0 else begin[i]))
            end[i] = max(-1, min(real_shape[i], end[i] + real_shape[i] if end[i] < 0 else end[i]))

    for axis in range(len(strides)):
        start_index = begin[axis]
        end_index = end[axis]
        stride = strides[axis]
        step = int((abs(end_index - start_index) - 1) / abs(stride))
        actual_index = [start_index + stride * s for s in range(step + 1)]
        actual_index = [] if start_index == end_index else actual_index

        # remove the index out of bound
        tmp_size = x.shape[axis]
        inorder_tmp_size_idx = range(0, tmp_size, 1) if end_index > start_index else range(tmp_size - 1, -1, -1)
        for i in range(len(actual_index) - 1, -1, -1):
            tmp_idx = actual_index[i]
            if tmp_idx not in inorder_tmp_size_idx:
                actual_index.remove(tmp_idx)

        x = x.index_select(axis, torch.tensor(actual_index, device=x.device).int())
    return x


@register_gen_input(OpType.Slice)
def gen_input_data(self):
    if len(self.inputs) > 1:
        in_shape0 = list(self.inputs[0].shape)
        in_shape1 = list(self.inputs[1].shape)
        in_type0 = str(self.inputs[0].dtype)
        in_type1 = str(self.inputs[1].dtype)

        if self.inputs[0].dtype.is_float:
            input0 = np.float16(np.random.uniform(-1000, 1000, size=in_shape0))
            if in_type0 == 'bloat16':
                from ml_dtypes import bfloat16
                input0 = input0.astype(bfloat16)
        else:
            input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)

        axes = random.sample(range(0, len(in_shape0)), in_shape1[0])
        axes.sort()

        starts = []
        ends = []
        strides = []
        input_shape = in_shape0
        for i in range(len(axes)):
            # be = list(np.random.choice(range(0, max(2, input_shape[axes[i]] // 2 + 1)), size=2, replace=False))
            be = list(np.random.choice(range(0, input_shape[axes[i]] + 1), size=2, replace=False))  # maybe out of bound
            if (axes[i] == 0 and input_shape[axes[i]] == 1) or max(be) == input_shape[
                    axes[i]]:  # when b=1/max(be)==in[i], s must > 0.
                be_copy = be.copy()
                be[0] = min(be_copy)
                be[1] = max(be_copy)
            starts.append(be[0])
            ends.append(be[1])
            if be[1] > be[0]:
                strides.append(np.random.randint(1, (be[1] - be[0]) + 1))
            else:
                strides.append(np.random.randint((be[1] - be[0]), 0))
        starts = np.asarray(starts, dtype=in_type1)
        ends = np.asarray(ends, dtype=in_type1)
        strides = np.asarray(strides, dtype=in_type1)
        axes = np.asarray(axes, dtype=in_type1)

        input_data = [input0, starts, ends, axes, strides]
        self.inputs[0].betensor = input_data[0]
        self.inputs[1].betensor = input_data[1]
        self.inputs[2].betensor = input_data[2]
        self.inputs[3].betensor = input_data[3]
        self.inputs[4].betensor = input_data[4]
        return input_data
    else:
        _GEN_INPUTS[OpType.Input](self)


@register_op_metric(OpType.Slice)
@check_fp_data
def metric(self, outputs, index=None):
    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]

    if groundtruths[0].size == 0:
        print('slice output size is 0!')
        return True

    return _OP_METRICS['_all'](self, outputs, index)
