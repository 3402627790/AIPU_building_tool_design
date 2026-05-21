from .gt_util import *
from ..engine import register_gt, register_gen_input
from AIPUBuilder.core import OpType
import copy


@register_gt(OpType.ScatterElements)
def scatterelements(self, *args, **kwargs):
    if 'float' in str(self.inputs[0].dtype):
        output = scatterelements_float(self)
    else:
        reduction = self.params['reduction'].upper()
        inputs = self.inputs[0].betensor.cpu().numpy().astype('int64')
        indices = self.inputs[1].betensor.cpu().numpy().astype('int64')
        updates = self.inputs[2].betensor.cpu().numpy().astype('int64')
        self.params['scale_value'] = [1, 1, 1] if self.params.get(
            'scale_value', None) is None else self.params['scale_value']
        if reduction in ['ADD', 'NONE', 'MAX', 'MIN']:
            self.params['shift_value'] = 0 if self.params.get('shift_value') is None else self.params['shift_value']
            output = inferOP_int(self,
                                 inputs,
                                 indices,
                                 updates,
                                 self.params['scale_value'][0],
                                 self.params['scale_value'][1],
                                 self.params['scale_value'][2],
                                 self.params['shift_value'],
                                 0,
                                 0,
                                 reduction,
                                 self.params['axis'],
                                 str(self.inputs[0].dtype))
        elif reduction == 'MUL':
            self.params['shift_value'] = [0, 0, 0] if self.params['shift_value'] is None else self.params['shift_value']
            output = inferOP_int(self,
                                 inputs,
                                 indices,
                                 updates,
                                 self.params['scale_value'][0],
                                 self.params['scale_value'][1],
                                 self.params['scale_value'][2],
                                 self.params['shift_value'][0],
                                 self.params['shift_value'][1],
                                 self.params['shift_value'][2],
                                 reduction,
                                 self.params['axis'],
                                 str(self.inputs[0].dtype))
        else:
            raise ValueError('do not support such method')
    self.outputs[0].betensor = output


def inferOP_int(self,
                data: np.array,
                indices: np.array,
                updates: np.array,
                OUT_SCALE: int,  # output scale
                INPUTS_SCALE: int,  # input0 scale
                UPDATES_SCALE: int,  # input1 scale
                OUT_SHIFT: int,  # output shift
                INPUTS_SHIFT: int,  # inputs shift
                UPDATES_SHIFT: int,  # update shift
                reduction: str,
                axis: int,
                precision: str):

    inputs = np.int64(data)
    inputs = inputs + self.inputs[0].quantization.offset

    inputs = inputs * INPUTS_SCALE * 0.5 ** INPUTS_SHIFT  # inputs shift = 0, when method == 'ADD'/'NONE'
    inputs_bits = 8 if '8' in precision else 16
    if 'u' not in precision:
        inner_min, inner_max = -2 ** (inputs_bits * 2 - 1), 2 ** (inputs_bits * 2 - 1) - 1
    else:
        inner_min, inner_max = 0, 2 ** (inputs_bits * 2) - 1

    inputs = np.clip(np.floor(inputs), inner_min, inner_max)
    outputs = copy.deepcopy(inputs)

    updates = np.int64(updates)
    updates = updates + self.inputs[2].quantization.offset

    if axis < 0:
        axis = inputs.ndim + axis

    if reduction == 'NONE':
        updates = updates * UPDATES_SCALE
        updates = np.clip(updates, inner_min, inner_max)
        for idx in np.ndindex(indices.shape):
            updates_idx = list(idx).copy()
            updates_idx[axis] = indices[idx]
            outputs[tuple(updates_idx)] = updates[idx]

    elif reduction == 'MUL':
        outputs = outputs.astype('float64')
        updates = updates * UPDATES_SCALE * 0.5 ** UPDATES_SHIFT
        updates = np.clip(np.round(updates), inner_min, inner_max)
        for idx in np.ndindex(indices.shape):
            updates_idx = list(idx).copy()
            updates_idx[axis] = indices[idx]
            outputs[tuple(updates_idx)] *= updates[idx]
            outputs[tuple(updates_idx)] = np.clip(np.floor(outputs[tuple(updates_idx)]), inner_min, inner_max)

    elif reduction == 'ADD':
        updates = updates * UPDATES_SCALE
        updates = np.clip(updates, inner_min, inner_max)
        for idx in np.ndindex(indices.shape):
            updates_idx = list(idx).copy()
            updates_idx[axis] = indices[idx]
            outputs[tuple(updates_idx)] += updates[idx]
            outputs[tuple(updates_idx)] = np.clip(np.floor(outputs[tuple(updates_idx)]), inner_min, inner_max)

    elif reduction == 'MAX':
        updates = updates * UPDATES_SCALE
        updates = np.clip(updates, inner_min, inner_max)
        for idx in np.ndindex(indices.shape):
            updates_idx = list(idx).copy()
            updates_idx[axis] = indices[idx]
            outputs[tuple(updates_idx)] = max(outputs[tuple(updates_idx)], updates[idx])

    elif reduction == 'MIN':
        updates = updates * UPDATES_SCALE
        updates = np.clip(updates, inner_min, inner_max)
        for idx in np.ndindex(indices.shape):
            updates_idx = list(idx).copy()
            updates_idx[axis] = indices[idx]
            outputs[tuple(updates_idx)] = min(outputs[tuple(updates_idx)], updates[idx])

    else:
        print('Invalid reduction')
    output = np.int64(np.round(outputs * OUT_SCALE * 0.5 ** OUT_SHIFT))
    output -= self.outputs[0].quantization.offset
    output = np.clip(output, np.iinfo(precision).min, np.iinfo(precision).max)

    return output


def scatterelements_float(self):
    inputs = torch_to_numpy(self.inputs[0].betensor, self.inputs[0].dtype.np).astype('float32')
    indices = self.inputs[1].betensor.cpu().numpy().astype('int64')
    updates = torch_to_numpy(self.inputs[2].betensor, self.inputs[2].dtype.np)

    axis = self.params['axis']
    outputs = copy.deepcopy(inputs)

    if self.params['reduction'] == 'NONE':
        for idx in np.ndindex(indices.shape):
            updates_idx = list(idx).copy()
            updates_idx[axis] = indices[idx]
            outputs[tuple(updates_idx)] = updates[idx]
    elif self.params['reduction'] == 'MUL':
        outputs = outputs.astype('float64')
        for idx in np.ndindex(indices.shape):
            updates_idx = list(idx).copy()
            updates_idx[axis] = indices[idx]
            outputs[tuple(updates_idx)] *= updates[idx]
    elif self.params['reduction'] == 'ADD':
        for idx in np.ndindex(indices.shape):
            updates_idx = list(idx).copy()
            updates_idx[axis] = indices[idx]
            outputs[tuple(updates_idx)] += updates[idx]
    elif self.params['reduction'] == 'MAX':
        for idx in np.ndindex(indices.shape):
            updates_idx = list(idx).copy()
            updates_idx[axis] = indices[idx]
            outputs[tuple(updates_idx)] = max(outputs[tuple(updates_idx)], updates[idx])
    elif self.params['reduction'] == 'MIN':
        for idx in np.ndindex(indices.shape):
            updates_idx = list(idx).copy()
            updates_idx[axis] = indices[idx]
            outputs[tuple(updates_idx)] = min(outputs[tuple(updates_idx)], updates[idx])
    else:
        print('Invalid reduction')

    outputs = outputs.astype(str(self.outputs[0].dtype))
    return outputs


@register_gen_input(OpType.ScatterElements)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_shape2 = list(self.inputs[2].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)
    in_type2 = str(self.inputs[2].dtype)

    if self.inputs[0].dtype.is_float:
        input0 = (np.random.uniform(-10000, 10000, size=in_shape0)).astype(in_type0)
        update = (np.random.uniform(-10000, 10000, size=in_shape2)).astype(in_type2)
    else:
        input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)
        update = np.random.randint(np.iinfo(in_type2).min, np.iinfo(in_type2).max, size=in_shape2, dtype=in_type2)
        if self.params['reduction'] == 'ADD' and '16' in in_type0:
            input0 = np.random.randint(max(np.iinfo(in_type0).min, -10), min(np.iinfo(in_type0).max, 10),
                                       size=in_shape0, dtype=in_type0)  # restrict scatterelement add inputs
            update = np.random.randint(max(np.iinfo(in_type2).min, -10),
                                       min(np.iinfo(in_type2).max, 10), size=in_shape2, dtype=in_type2)
    axis = self.params['axis']
    min_v = -in_shape0[axis]
    max_v = in_shape0[axis] - 1
    if 'u' in in_type1:
        input1 = np.random.randint(0, max_v + 1, size=in_shape1, dtype=in_type1)
    else:
        input1 = np.random.randint(min_v, max_v + 1, size=in_shape1, dtype=in_type1)

    input_data = [input0, input1, update]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    self.inputs[2].betensor = input_data[2]
    return input_data
