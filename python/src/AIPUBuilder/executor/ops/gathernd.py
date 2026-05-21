import torch
from .gt_util import *
from ..engine import register_gt, register_gen_input
from AIPUBuilder.core import OpType


@register_gt(OpType.GatherND)
def GatherND(self, *args, **kwargs):
    # indices defines slices into the first N dimensions of params, where N = indices.shape[-1]
    # input data last dim is keeped, if input dim == indice dim, input's shape[-1] as 1
    inp_betensors = self.inputs[0].betensor
    inp_indice = (self.inputs[1].betensor).long()

    batch_dims = self.params.get('batch_dims', 0)

    if batch_dims == 0:
        if inp_indice.ndim == 1:
            inp_indice = inp_indice.unsqueeze(0)
        out = singleGatherNd(inp_betensors, inp_indice)
    else:  # batch dim==1
        index_depth = inp_indice.shape[-1]
        batch_shape = inp_indice.shape[:batch_dims]
        batchnum = torch.prod(torch.tensor(batch_shape, device=inp_betensors.device))
        outer_shape = inp_indice.shape[batch_dims:-1]
        inner_shape = inp_betensors.shape[batch_dims + index_depth:]
        newshape = batch_shape + outer_shape + inner_shape
        out = torch.zeros(newshape, device=inp_betensors.device)
        if out.ndim == 1:
            out = out.unsqueeze(0)
        new_indice = inp_indice.reshape((batchnum,) + inp_indice.shape[batch_dims:])
        new_inp = inp_betensors.reshape((batchnum,) + inp_betensors.shape[batch_dims:])
        out = out.reshape((batchnum,) + out.shape[batch_dims:])
        for batch in range(batchnum):
            indice_batch = new_indice[batch, ...]
            inp_batch = new_inp[batch, ...]
            if indice_batch.ndim == 1:
                indice_batch = indice_batch.unsqueeze(0)
            out[batch, ...] = singleGatherNd(inp_batch, indice_batch).reshape(out[batch, ...].shape)
    out_shape = self.attrs["_output_shapes"][0] if "_output_shapes" in self.attrs else list(self.outputs[0].shape)
    self.outputs[0].betensor = out.reshape(out_shape)

    return [o.betensor for o in self.outputs]


def singleGatherNd(inp_betensors, inp_indice):
    # input data last dim is keeped, if input dim == indice dim, input's shape[-1] as 1
    # indice's last dim indicate how many input's dim need be indiced
    newshape = inp_indice.shape[:-1]+inp_betensors.shape[inp_indice.shape[-1]:]
    out = torch.zeros(newshape, device=inp_betensors.device)
    if out.ndim == 1:
        out = out.unsqueeze(-1)
    if inp_indice.ndim > 2:
        m = 1
        for i in range(inp_indice.ndim-1):
            m = m*inp_indice.shape[i]
        inp_indice = inp_indice.reshape(m, inp_indice.shape[-1])
        newshape = (m,)+inp_betensors.shape[inp_indice.shape[-1]:]
        out = torch.zeros(newshape, device=inp_betensors.device)

    last_dims = inp_indice.shape[-1]
    for dim in range(last_dims):
        inp_indice[..., dim] = torch.minimum(torch.tensor(inp_betensors.shape[dim] - 1, device=inp_betensors.device),
                                             inp_indice[..., dim])

    if(inp_betensors.ndim == inp_indice.shape[-1]):
        inp_betensors = inp_betensors.unsqueeze(-1)

    if inp_indice.shape[-1] <= 5:
        for seq in range(newshape[0]):
            out[seq, ...] = inp_betensors[tuple(inp_indice[seq])]
    else:
        raise ValueError("need indice data dim only is 2 or 3,4,5 ")
    return out


@register_gen_input(OpType.GatherND)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)

    # todo: modify max_v
    min_v = 0
    max_v = min(in_shape0)
    max_v = max(1, max_v // 2)  # for ds
    if self.inputs[0].dtype.is_float:
        input0 = (np.random.randn(*in_shape0) * 1000).astype(in_type0)
    else:
        input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)
    input1 = np.random.randint(min_v, max_v, size=in_shape1, dtype=in_type1)

    input_data = [input0, input1]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    return input_data
