from ..engine import register_gt, register_gen_input
from .gt_util.quant_utils import clamp
from .gt_util.dtype_utils import str2dtype, dtype2range
from AIPUBuilder.core import OpType
import torch
import numpy as np


@register_gt(OpType.Where)
def where(self, *args, **kwargs):
    if self.outputs[0].dtype.is_float:
        return fp_where(self)
    output_type = str(self.outputs[0].dtype)
    condition = self.inputs[0].betensor.long()
    # condition no quant, if condition_zp != 0 raise error
    if self.inputs[0].quantization.offset != 0:
        print("ERROR, condition should not quant")
        # return False

    if len(self.inputs) == 1:
        indexes = torch.where(condition)
        output = torch.tensor([t.cpu().numpy() for t in indexes])
        output = output.permute(1, 0).flatten()
        dim = len(list(condition.shape))

        invalid_data_num = condition.numel() * dim - output.numel()
        max_value_num = 0 if invalid_data_num == 0 else condition.ndim
        invalid_data = torch.ones(max_value_num, dtype=output.dtype) * np.iinfo(self.outputs[0].dtype.np).max
        invalid_data_zero = torch.zeros(max(0, invalid_data_num - condition.ndim), dtype=self.outputs[0].betensor.dtype)
        output = torch.cat((output, invalid_data, invalid_data_zero))

    else:
        input_data1 = self.inputs[1].betensor.double()
        input_data2 = self.inputs[2].betensor.double()
        input_data1 = input_data1 + self.inputs[1].quantization.offset
        input_data2 = input_data2 + self.inputs[2].quantization.offset

        input_data1 = torch.round(
            input_data1 * self.params['scale_value'][0] * 0.5 ** self.params['shift_value'][0])
        input_data2 = torch.round(
            input_data2 * self.params['scale_value'][1] * 0.5 ** self.params['shift_value'][1])

        output = torch.where(condition != 0, input_data1, input_data2).long()
        output -= self.outputs[0].quantization.offset
        output = clamp(output, self.outputs[0].dtype)

    output_shape = self.attrs['_output_shapes'][0] if '_output_shapes' in self.attrs else list(self.outputs[0].shape)
    output = output.reshape(output_shape)
    self.outputs[0].betensor = output
    return self.outputs[0].betensor


@register_gen_input(OpType.Where)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_type0 = str(self.inputs[0].dtype)

    if len(self.inputs) == 1:
        condition = np.random.randint(0, 2, size=in_shape0, dtype=in_type0)
        input_data = [condition]
    else:
        in_shape1 = list(self.inputs[1].shape)
        in_type1 = str(self.inputs[1].dtype)
        in_shape2 = list(self.inputs[2].shape)
        in_type2 = str(self.inputs[2].dtype)

        condition = np.random.randint(0, 2, size=in_shape0, dtype=in_type0)
        if self.inputs[1].dtype.is_float:
            input1 = (1000 * np.random.random(in_shape1)).astype(in_type1)
        else:
            input1 = np.random.randint(np.iinfo(in_type1).min, np.iinfo(in_type1).max, size=in_shape1, dtype=in_type1)
        if self.inputs[2].dtype.is_float:
            input2 = (1000 * np.random.random(in_shape2)).astype(in_type2)
        else:
            input2 = np.random.randint(np.iinfo(in_type2).min, np.iinfo(in_type2).max, size=in_shape2, dtype=in_type2)
        input_data = [condition, input1, input2]
        self.inputs[1].betensor = input_data[1]
        self.inputs[2].betensor = input_data[2]

    self.inputs[0].betensor = input_data[0]
    return input_data


def fp_where(self):
    condition = self.inputs[0].betensor.long()
    if len(self.inputs) == 1:
        indexes = torch.where(condition)
        output = torch.tensor([t.cpu().numpy() for t in indexes])
        output = output.permute(1, 0).flatten()
        dim = len(list(condition.shape))

        invalid_data_num = condition.numel() * dim - output.numel()
        max_value_num = 0 if invalid_data_num == 0 else condition.ndim
        invalid_data = torch.ones(max_value_num, dtype=output.dtype) * np.iinfo(self.outputs[0].dtype.np).max
        invalid_data_zero = torch.zeros(max(0, invalid_data_num - condition.ndim), dtype=self.outputs[0].betensor.dtype)
        output = torch.cat((output, invalid_data, invalid_data_zero))

    else:
        input_data1 = self.inputs[1].betensor
        input_data2 = self.inputs[2].betensor

        output = torch.where(condition != 0, input_data1, input_data2)

    output_shape = self.attrs['_output_shapes'][0] if '_output_shapes' in self.attrs else list(self.outputs[0].shape)
    output = output.reshape(output_shape)
    self.outputs[0].betensor = output
    return self.outputs[0].betensor
