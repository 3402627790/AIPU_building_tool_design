from ..engine import register_gt
from AIPUBuilder.core import Target
from .gt_util.quant_utils import clamp, _tpc_lut
from AIPUBuilder.core import OpType
import torch
import numpy as np


@register_gt(OpType.Div)
def div(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float:
        return fp_div(self, *args, **kwargs)
    input_1 = self.inputs[0].betensor.cpu().numpy()
    input_2 = self.inputs[1].betensor.cpu().numpy()
    input_1 = input_1.astype("int64") + self.inputs[0].quantization.offset
    input_2 = input_2.astype("int64") + self.inputs[1].quantization.offset
    input_2_sign = np.sign(input_2)
    input_2_sign[input_2_sign == 0] = 1
    assert str(self.inputs[1].dtype).endswith("8"), "only support 8 bit input now"
    shift_value = self.params["shift_value"]
    # if self.inputs[0].quantization.offset != 0 or self.inputs[1].quantization.offset != 0:
    #     input_1 = np.round(input_1 * 0.5).astype('int64')
    #     input_2 = np.round(input_2 * 0.5).astype('int64')
    #     shift_value -= 1
    input2_abs = np.abs(input_2)
    input2_abs[input2_abs > 255] = 255
    input2_lut = _tpc_lut(torch.from_numpy(input2_abs), self.constants["lut"].betensor, 8, False).cpu().numpy()
    outputs = input_1 * input2_lut * input_2_sign
    outputs = np.clip(outputs, np.iinfo(np.int32).min, np.iinfo(np.int32).max)

    outputs = np.round(outputs * self.params["scale_value"] * (0.5**shift_value))
    outputs = np.clip(outputs, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    outputs -= self.outputs[0].quantization.offset
    out = np.clip(outputs, np.iinfo(str(self.outputs[0].dtype)).min, np.iinfo(str(self.outputs[0].dtype)).max)
    self.outputs[0].betensor = torch.from_numpy(out)
    return self.outputs[0].betensor


def fp_div(self, *args, **kwargs):
    inp0 = self.inputs[0].irtensor.to(torch.float32)
    inp1 = self.inputs[1].irtensor.to(torch.float32)
    is_aiff = False
    if "is_aiff" in kwargs.keys():
        is_aiff = kwargs["is_aiff"]
    if is_aiff:
        from .gt_util.math_utils import fp32_to_fp24
        from .gt_util.aiff import x3_aiff_fp32_2_fpx

        tmp = fp32_to_fp24(1.0 / inp1)
        outputs = fp32_to_fp24(inp0 * tmp)
        outputs = x3_aiff_fp32_2_fpx(outputs, is_aiff, fpx=self.outputs[0].dtype)
    else:
        outputs = torch.div(inp0, inp1)
        outputs = outputs.to(self.outputs[0].dtype.torch)
    self.outputs[0].irtensor = outputs
    return outputs
