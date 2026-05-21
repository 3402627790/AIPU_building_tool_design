import torch
from .gt_util import *
from ..engine import register_gt, register_gen_weight, register_gen_input
from AIPUBuilder.core import OpType


@register_gt(OpType.Gather)
def Gather(self, *args, **kwargs):
    params = self.inputs[0].irtensor
    ids = self.inputs[1].betensor.long() if len(self.inputs) == 2 else self.constants['lut'].betensor
    ids_type = self.inputs[1].dtype if len(self.inputs) == 2 else self.constants['lut'].dtype
    out_shape_ = self.attrs["_output_shapes"][0] if "_output_shapes" in self.attrs else list(self.outputs[0].shape)

    method = self.params['method'] if 'method' in self.params.keys() else 'none'
    if 'lut' in self.constants.keys() and method != 'idx_lut':
        params = self.constants['lut'].betensor
        ids = self.inputs[0].betensor.long()
        ids_type = self.inputs[0].dtype

    # in ids, values after 0xffff(from Where op) are considered invalid(not work from 2024/5/9: CTL-8939)
    select_valid_index = False
    max_index = np.argwhere(ids.cpu().numpy() > np.iinfo(str(ids_type)).max)
    if max_index.size != 0:
        ids = ids[: max_index[0][0] + 1, : max_index[0][1]]
        select_valid_index = True

    axis = self.params['axis']
    axis = axis if axis >= 0 else len(params.shape) + axis
    batch_dims = self.params['batch_dims']
    virtual_batch = torch.prod(torch.tensor(params.shape[:batch_dims], device=ids.device)).int().item()

    if batch_dims > axis:
        raise ValueError("Please check your shape, batch dim should be less than axis ")
    else:
        axis = axis - batch_dims + 1
    newshape = (virtual_batch,) + params.shape[batch_dims:]
    params = params.reshape(newshape)
    # to get total indice length, it maybe be allocated in several dims, batch_dims is border
    # such as indice is [10,100,2], if batch_dims is 1, then [100,2] is indice, 10 is batch
    outshape = newshape[0:axis] + ids.shape[batch_dims:] + newshape[axis + 1:]
    temp_out = torch.zeros(outshape, device=ids.device)
    newshape = (virtual_batch,) + ids.shape[batch_dims:]

    ids = ids.reshape(newshape)
    padding = [0] * (params.dim() - axis) * 2
    padding_value = -self.inputs[0].quantization.offset
    for i in range(virtual_batch):
        # axis include dim, so need axis-1
        if torch.max(ids) <= params.shape[axis]:
            max_len = params[i].shape[axis - 1]
            pos = ids[i].flatten() != max_len
            inp_indice = ids[i].flatten()[pos].long()
        else:
            inp_indice = ids[i].flatten().long()
        # index_select not support negative index,so we need do some transformation

        neg_index = (inp_indice < 0) * params[i].shape[axis - 1]
        inp_indice = inp_indice + neg_index
        indice_len = valid_num = inp_indice.shape[0]

        out = torch.index_select(params[i], axis - 1, inp_indice)

        if select_valid_index:
            out_size = np.prod(out_shape_)
            temp_out = torch.zeros(out_size)
            temp_out[: out.cpu().numpy().size] = out.flatten()
        else:
            padding[-1] = indice_len - valid_num
            out = torch.nn.functional.pad(out, padding, value=padding_value)
            temp_out[i] = out.reshape(temp_out[i].shape)

    self.outputs[0].irtensor = temp_out.reshape(out_shape_).to(self.outputs[0].dtype.torch)
    # self.outputs[0].betensor = temp_out.reshape(list(self.outputs[0].shape))

    return [o.betensor for o in self.outputs]


@register_gen_weight(OpType.Gather)
def gen_random_weight(self):
    if 'lut' in self.constants:
        min_v = 0
        max_v = self.inputs[0].shape[self.params['axis']]
        self.constants['lut'].betensor = np.random.randint(min_v, max_v, size=self.constants['lut'].shape,
                                                           dtype=str(self.constants['lut'].dtype))


@register_gen_input(OpType.Gather)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_type0 = str(self.inputs[0].dtype)

    axis = self.params['axis']
    min_v = 0
    max_v = in_shape0[axis]
    if self.inputs[0].dtype.is_float:
        input0 = (np.random.randn(*in_shape0) * 1000).astype(in_type0)
    else:
        input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)
    if len(self.inputs) == 2:
        in_type1 = str(self.inputs[1].dtype)
        in_shape1 = list(self.inputs[1].shape)
        input1 = np.random.randint(min_v, max_v, size=in_shape1, dtype=in_type1)
        input_data = [input0, input1]
        self.inputs[1].betensor = input_data[1]
    else:
        input_data = [input0]

    self.inputs[0].betensor = input_data[0]
    return input_data
