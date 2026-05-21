from .gt_util import *
from ..engine import register_gt, register_gen_input
from AIPUBuilder.core import OpType
from .gt_util.dtype_utils import torch_to_numpy, numpy_to_torch


@register_gt(OpType.ScatterND)
def scatternd(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float:
        output = scatternd_float(self)
    else:
        reduction = self.params['reduction'].upper()

        inputs = self.inputs[0].irtensor.cpu().numpy().astype('int64')
        indices = self.inputs[1].irtensor.cpu().numpy().astype('int64')
        updates = self.inputs[2].irtensor.cpu().numpy().astype('int64')
        if reduction in ['ADD', 'NONE', 'MAX', 'MIN']:
            output = inferOP_int(self,
                                 inputs,
                                 indices,
                                 updates,
                                 self.params['scale_value'][0],
                                 self.params['scale_value'][1],
                                 self.params['scale_value'][2],
                                 self.params['shift_value'],
                                 1,
                                 reduction,
                                 str(self.inputs[0].dtype))
        elif reduction == 'MUL':
            output = inferOP_int(self,
                                 inputs,
                                 indices,
                                 updates,
                                 1,
                                 self.params['scale_value'][1],
                                 self.params['scale_value'][2],
                                 self.params['shift_value'][0],
                                 self.params['shift_value'][2],
                                 reduction,
                                 str(self.inputs[0].dtype))
        else:
            raise ValueError('no support such method')

    self.outputs[0].irtensor = output
    return output


def inferOP_int(self,
                data: np.array,
                indices: np.array,
                updates: np.array,
                OUT_SCALE: int,  # output scale
                INPUTS_SCALE: int,  # input0 scale
                UPDATES_SCALE: int,  # input1 scale
                OUT_SHIFT: int,  # output shift
                UPDATES_SHIFT: int,  # update shift
                reduction: str,
                precision: str):

    inputs = np.int64(data)
    inputs = inputs + self.inputs[0].quantization.offset
    input_shape = inputs.shape
    inputs = inputs * OUT_SCALE * INPUTS_SCALE

    indices_shape = indices.shape
    indices = indices.reshape(-1, indices_shape[-1])

    k = indices.shape[-1]
    q = indices.ndim
    updates = np.int64(updates)
    updates = updates + self.inputs[2].quantization.offset
    updates = updates * OUT_SCALE * UPDATES_SCALE
    updates_shape = indices.shape[0: q - 1] + input_shape[k:]
    updates = updates.reshape(updates_shape)
    updates_indices = indices.shape[:-1]

    if reduction == 'NONE':
        for idx in np.ndindex(updates_indices):
            inputs[tuple(indices[idx])] = updates[idx]
    elif reduction == 'MUL':
        inputs = np.float64(inputs)
        for idx in np.ndindex(updates_indices):
            inputs[tuple(indices[idx])] *= updates[idx]
            inputs[tuple(indices[idx])] *= 0.5 ** UPDATES_SHIFT
            inputs[tuple(indices[idx])] = np.round(inputs[tuple(indices[idx])])
            inputs[tuple(indices[idx])] = np.clip(inputs[tuple(indices[idx])], -2 ** 31, 2 ** 31 - 1)
    elif reduction == 'ADD':
        for idx in np.ndindex(updates_indices):
            inputs[tuple(indices[idx])] += updates[idx]
            if '8' in precision:
                inputs[tuple(indices[idx])] = np.clip(inputs[tuple(indices[idx])], -2 ** 31, 2 ** 31 - 1)
    elif reduction == 'MAX':
        for idx in np.ndindex(updates_indices):
            if updates.ndim > 1:
                temp = np.where(inputs[tuple(indices[idx])] < updates[idx])
                inputs[tuple(indices[idx])][temp] = updates[idx][temp]
            else:
                inputs[tuple(indices[idx])] = max(inputs[tuple(indices[idx])], updates[idx])
    elif reduction == 'MIN':
        for idx in np.ndindex(updates_indices):
            if updates.ndim > 1:
                temp = np.where(inputs[tuple(indices[idx])] > updates[idx])
                inputs[tuple(indices[idx])][temp] = updates[idx][temp]
            else:
                inputs[tuple(indices[idx])] = min(inputs[tuple(indices[idx])], updates[idx])
    else:
        raise ValueError('no support such method')
    if str(self.inputs[0].dtype).endswith('8'):
        output = np.int64(np.round(inputs * 0.5 ** OUT_SHIFT))  # only 8bit can align to opt
    else:
        output = np.int64(np.round(inputs)) >> OUT_SHIFT
    output -= self.outputs[0].quantization.offset
    output = np.clip(output, np.iinfo(precision).min, np.iinfo(precision).max)

    return output


def scatternd_float(self):
    inputs = torch_to_numpy(self.inputs[0].irtensor,  self.inputs[0].dtype.np).astype("float32").copy()
    indices = self.inputs[1].irtensor.cpu().numpy().astype('int64')
    updates = torch_to_numpy(self.inputs[2].irtensor,  self.inputs[0].dtype.np).astype("float32")

    input_shape = inputs.shape
    indices_shape = indices.shape
    indices = indices.reshape(int(np.prod(indices_shape[:-1])), indices_shape[-1])
    k = indices.shape[-1]
    q = indices.ndim
    updates_shape = indices.shape[0: q - 1] + input_shape[k:]
    updates = updates.reshape(updates_shape)
    updates_indices = indices.shape[:-1]

    if self.params['reduction'] == 'NONE':
        for idx in np.ndindex(updates_indices):
            inputs[tuple(indices[idx])] = updates[idx]
    elif self.params['reduction'] == 'MUL':
        for idx in np.ndindex(updates_indices):
            inputs[tuple(indices[idx])] *= updates[idx]
    elif self.params['reduction'] == 'ADD':
        for idx in np.ndindex(updates_indices):
            inputs[tuple(indices[idx])] += updates[idx]
    elif self.params['reduction'] == 'MAX':
        for idx in np.ndindex(updates_indices):
            if updates.ndim > 1:
                temp = np.where(inputs[tuple(indices[idx])] < updates[idx])
                inputs[tuple(indices[idx])][temp] = updates[idx][temp]
            else:
                inputs[tuple(indices[idx])] = max(inputs[tuple(indices[idx])], updates[idx])
    elif self.params['reduction'] == 'MIN':
        for idx in np.ndindex(updates_indices):
            if updates.ndim > 1:
                temp = np.where(inputs[tuple(indices[idx])] > updates[idx])
                inputs[tuple(indices[idx])][temp] = updates[idx][temp]
            else:
                inputs[tuple(indices[idx])] = min(inputs[tuple(indices[idx])], updates[idx])
    else:
        print('Invalid reduction')

    outputs = inputs.astype(self.outputs[0].dtype.np)
    self.outputs[0].irtensor = outputs
    return self.outputs[0].betensor


@register_gen_input(OpType.ScatterND)
def gen_input_data(self):
    import random
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_shape2 = list(self.inputs[2].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)
    in_type2 = str(self.inputs[2].dtype)

    if self.inputs[0].dtype.is_float:
        input0 = (np.random.randn(*in_shape0) * 10000).astype(in_type0)
        update = (np.random.randn(*in_shape2) * 10000).astype(in_type2)
    else:
        input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)
        update = np.random.randint(np.iinfo(in_type2).min, np.iinfo(in_type2).max, size=in_shape2, dtype=in_type2)

    index_shape = in_shape1
    tensor_shape = in_shape0
    index_batch = np.prod(index_shape[:-1])
    index_num = index_shape[-1]

    all_index = []
    for ids in np.ndindex(tuple(tensor_shape[: index_num])):
        all_index.append(ids)
    random.shuffle(all_index)
    index = all_index[:index_batch]
    index = np.asarray(index).astype(in_type1)
    index = index.reshape(index_shape)

    input_data = [input0, index, update]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    self.inputs[2].betensor = input_data[2]
    return input_data
