import torch
import numpy as np
from AIPUBuilder.core import OpType
from AIPUBuilder.logger import ERROR
from ..engine import register_gt, register_gen_input
from .gt_util.quant_utils import linear_requantize


@register_gt(OpType.SlotUpdate)
def SlotUpdate(self, *args, **kwargs):
    inputs = self.inputs
    if len(inputs) - 1 != self.inputs[0].shape[1]:
        ERROR(
            f"for slotupdate, the len(inputs) - 1(={len(inputs) - 1}) should be equal to inputs[0].shape[1]={self.inputs[0].shape[1]}")

    # input is dynamic, output is static
    out_shape = self.attrs['_output_shapes'][0] if '_output_shapes' in self.attrs else self.outputs[0].shape
    output = torch.zeros(list(out_shape)).float()
    inp0_betensor = self.inputs[0].betensor.float() + self.inputs[0].quantization.offset

    for i, inp_t in enumerate(self.inputs[1:]):
        idx = inp_t.betensor.long().cpu().numpy().tolist()
        for k, id in enumerate(idx):
            output[:, id, :] += inp0_betensor[:, i, :inp_t.betensor.shape[0]][:, k]

    if self.inputs[0].dtype.is_int:
        shift = self.params["shift_value"]
        scale = self.params["scale_value"]
        output = linear_requantize(output, scale, shift, self.outputs[0].dtype, self.outputs[0].quantization.offset)

    self.outputs[0].betensor = output
    return [o.betensor for o in self.outputs]


@register_gen_input(OpType.SlotUpdate)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)
    idx_len = in_shape0[2]
    idx_len = max(1, idx_len // 2)  # for ds test

    if self.inputs[0].dtype.is_float:
        input0 = np.random.uniform(-1000, 1000, size=in_shape0).astype(in_type0)
    else:
        input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)
    input_data = [input0]
    for i in range(1, len(self.inputs)):
        input_i = np.random.randint(0, idx_len, size=self.inputs[i].shape, dtype=in_type1)
        input_data.append(input_i)
        self.inputs[i].betensor = input_i
    self.inputs[0].betensor = input_data[0]
    return input_data
