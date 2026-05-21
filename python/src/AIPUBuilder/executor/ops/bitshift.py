import torch
import numpy as np
from ..engine import register_gt, register_gen_input
from AIPUBuilder.core import OpType, Dtype
from .gt_util.quant_utils import clamp


@register_gt(OpType.BitShift)
def bitshift(self, *args, **kwargs):
    input_tensor = self.inputs[0].betensor.cpu().numpy()
    input_tensor = input_tensor.astype(str(self.inputs[0].dtype))
    shifts = self.inputs[1].betensor.cpu().numpy().astype('uint8')
    input_tensor = input_tensor + self.inputs[0].quantization.offset
    output_type = str(self.outputs[0].dtype)
    if self.params['direction'] == 'RIGHT':
        output = input_tensor >> shifts
    else:
        output = input_tensor << shifts
        if '8' in output_type:
            output = output & 0xff
        elif '16' in output_type:
            output = output & 0xffff
        elif '32' in output_type:
            output = output & 0xffffffff
        else:
            print("Invalid output type")
    output = output.astype(self.outputs[0].dtype.np)
    output = output.astype(str(self.outputs[0].dtype))
    output -= self.outputs[0].quantization.offset
    output = np.clip(output, np.iinfo(output_type).min, np.iinfo(output_type).max)
    output = output.astype(output_type)
    self.outputs[0].betensor = output
    return self.outputs[0].betensor


@register_gen_input(OpType.BitShift)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)

    input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)
    shifts = np.random.randint(0, 127 + 1, size=in_shape1, dtype=in_type1)  # lib only support shift <= 127

    input_data = [input0, shifts]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    return input_data
