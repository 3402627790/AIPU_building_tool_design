from .gt_util import *
from ..engine import register_gt
from AIPUBuilder.core import OpType
from .gt_util.aiff import x3_aiff_fp32_2_fpx


@register_gt(OpType.SquaredDifference)
def squareddifference(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float or self.inputs[1].dtype.is_float:
        input0 = self.inputs[0].irtensor.to(torch.float32)
        input1 = self.inputs[1].irtensor.to(torch.float32)
        out0 = fp32_to_fp24(input0 - input1, True)
        output = fp32_to_fp24(torch.square(out0))
        output = x3_aiff_fp32_2_fpx(output, fpx=self.outputs[0].dtype)
    else:
        scale_out, scale_in0, scale_in1 = self.params["scale_value"]
        shift_out, shift_in0, shift_in1 = self.params["shift_value"]
        output_type = str(self.outputs[0].dtype)
        input0 = self.inputs[0].irtensor.cpu().numpy().astype(np.int64)
        input1 = self.inputs[1].irtensor.cpu().numpy().astype(np.int64)
        input0 = input0 + self.inputs[0].quantization.offset
        input1 = input1 + self.inputs[1].quantization.offset

        diff = np.round(input0 * scale_in0 * (0.5**shift_in0)) - np.round(input1 * scale_in1 * (0.5**shift_in1))
        diff = np.clip(diff, np.iinfo(np.int32).min, np.iinfo(np.int32).max)  # int32 before square
        square_diff = diff * diff
        square_diff = np.clip(square_diff, np.iinfo(np.int32).min, np.iinfo(np.int32).max)  # int32 after square
        output = np.round(square_diff * scale_out * (0.5**shift_out))
        output -= self.outputs[0].quantization.offset
        output = np.clip(output, np.iinfo(output_type).min, np.iinfo(output_type).max)
    self.outputs[0].irtensor = output
    return output
