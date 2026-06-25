# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *


@op_register(OpType.Gather)
# IR
# layer_bottom=rpn_class/concat_0,topk_indice
# layer_bottom_shape=[1,261888,2],[1,6000]
# layer_bottom_type=float32,float32
# layer_top=gather_prob_tensor
# layer_top_shape=[1,6000,2]
# layer_top_type=float32
# axis=1
# batch_dims=1
# bathc_dims is less than indice'dim, means how many dim as batch
def gather(self, *args):
    try:
        inp0_betensors = self.inputs[0].betensor.clone()
        axis = self.get_param('axis')
        if axis < 0:
            axis += inp0_betensors.dim()
        output_axis = axis
        batch_dims = self.get_param('batch_dims', optional=True, default_value=0)
        indice_betensor = None
        indice_dtype = None
        # if lut is in IR, one of gather's two inputs is from lut
        # if method =='idx_lut', indice from lut,or gather's object from lut
        if len(self.inputs) < 2:
            lut_idx = self.get_param('method')
            if 'lut' not in self.constants:
                OPT_FATAL("Please check your input or lut should be included in IR")
            else:
                if lut_idx == 'idx_lut':
                    indice_dtype = self.constants['lut'].dtype
                    indice_betensor = self.constants['lut'].betensor.long()
                else:
                    inp0_betensors = self.constants['lut'].betensor
                    indice_dtype = self.inputs[0].dtype
                    indice_betensor = self.inputs[0].betensor.long()
        else:
            indice_dtype = self.inputs[1].dtype
            indice_betensor = self.inputs[1].betensor.clone().long()
        select_valid_index = False
        invalid_index = None
        if indice_dtype in [Dtype.UINT32, Dtype.INT32, Dtype.INT64, Dtype.UINT64]:
            _, invalid_index = dtype2range(indice_dtype)
            select_valid_index = True
        virtual_batch = torch.prod(torch.tensor(
            inp0_betensors.shape[:batch_dims], device=indice_betensor.device)).int().item()
        # for i in range(indice_betensor.ndim-1):
        #     if indice_betensor.shape[i] == inp0_betensors.shape[i] and i<axis:
        #         virtual_batch = virtual_batch*indice_betensor.shape[i]
        #         batch_dims = batch_dims + 1
        if batch_dims > axis:
            OPT_FATAL("Please check your shape, batch dim should be less than axis ")
        else:
            axis = axis - batch_dims + 1
        newshape = (virtual_batch,) + inp0_betensors.shape[batch_dims:]
        inp0_betensors = inp0_betensors.reshape(newshape)
        # to get total indice length, it maybe be allocated in seveal dims, batch_dims is border
        # such as indice is [10,100,2], if batch_dims is 1, then [100,2] is indice,10 is batch
        outshape = newshape[0:axis] + indice_betensor.shape[batch_dims:] + newshape[axis + 1:]
        temp_out = torch.zeros(outshape, device=inp0_betensors.device, dtype=inp0_betensors.dtype)
        newshape = (virtual_batch,) + indice_betensor.shape[batch_dims:]

        indice_betensor = indice_betensor.reshape(newshape)
        padding = [0] * (inp0_betensors.dim() - axis) * 2
        # TODO: support per-channel zerop and pad the per-channel zerop
        if self.quantized:
            padding_value = -self.inputs[0].zerop[0]
        else:
            padding_value = 0
        for i in range(virtual_batch):
            # axis include dim, so need axis-1
            max_len = inp0_betensors[i].shape[axis - 1]
            current_indice = indice_betensor[i].flatten()
            if current_indice.numel() > 0 and torch.max(current_indice) <= max_len:
                pos = current_indice != max_len
                inp_indice = current_indice[pos].long()
            else:
                inp_indice = current_indice.long()
            # index_select not support negative index,so we need do some transformation

            neg_index = (inp_indice < 0) * inp0_betensors[i].shape[axis - 1]
            inp_indice = inp_indice + neg_index
            indice_len = valid_num = inp_indice.shape[0]

            # Remove the invalid index
            if select_valid_index:
                invalid_pos = torch.where(inp_indice == invalid_index)[0]
                if invalid_pos.numel() > 0:
                    inp_indice = inp_indice[:invalid_pos[0].item()]
                    valid_num = inp_indice.shape[0]
            out = torch.index_select(inp0_betensors[i], axis - 1, inp_indice)
            padding[-1] = indice_len - valid_num
            out = torch.nn.functional.pad(out, padding, value=padding_value)
            # out shape maybe not match temp_out shape due to some inddex including max index range
            # so only fill actual data to temp_out
            tmp = temp_out[i].flatten()
            tmp[:out.flatten().shape[0]] = out.flatten()
            temp_out[i] = tmp.reshape(temp_out[i].shape)
        ir_shape = list(self.outputs[0].ir_shape)
        t_shape = list(temp_out.shape)
        if 'ds_output_shape' in self.attrs:
            ds_output_shape = self.attrs['ds_output_shape']
            ir_shape = ds_output_shape
        elif ir_shape != t_shape:
            if output_axis >= len(ir_shape):
                OPT_FATAL("Please check your shape, axis is out of range for output shape")
            ir_shape[output_axis] = -1
        self.outputs[0].betensor = temp_out.reshape(ir_shape)

    except Exception as e:
        OPT_DEBUG(f"try normal impl gather failed, now try the dynamic shape impl: {e}")
        inp0_betensors = self.inputs[0].betensor.clone()
        axis = self.get_param('axis')
        if axis < 0:
            axis += inp0_betensors.dim()
        if len(self.inputs) < 2:
            lut_idx = self.get_param('method')
            if 'lut' not in self.constants:
                OPT_FATAL("Please check your input or lut should be included in IR")
            if lut_idx == 'idx_lut':
                index = self.constants['lut'].betensor.clone().long()
            else:
                inp0_betensors = self.constants['lut'].betensor.clone()
                index = self.inputs[0].betensor.clone().long()
        else:
            index = self.inputs[1].betensor.clone().long()
        # for exposing potential IR errors, do not clamp the index
        # index = index.clamp(-inp0_betensors.shape[axis], inp0_betensors.shape[axis]-1)
        if axis != 0:
            self.outputs[0].betensor = torch.index_select(inp0_betensors, axis, index)
        else:
            if index.numel() == 1 and len(self.inputs[0].ir_shape) - len(self.outputs[0].ir_shape) == 1:
                index = index.item()
            self.outputs[0].betensor = inp0_betensors[index]

    return self.outputs[0].betensor


@quant_register(OpType.Gather)
def gather_quantize(self, *args):
    inp = self.inputs[0]
    out = self.outputs[0]
    out.dtype = inp.dtype
    out.scale = inp.scale
    out.zerop = inp.zerop
    out.qbits = inp.qbits
    out.qinvariant = inp.qinvariant
