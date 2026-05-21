from ..engine import register_gt
from .gt_util.quant_utils import clamp, linear_requantize
from .gt_util.aiff import x3_aiff_fp32_2_fpx
from AIPUBuilder.core import OpType, ComputeTensor, Dtype
from AIPUBuilder.logger import FATAL, DEBUG
import torch
import numpy as np


def f16_to_int(i, s=1, shift=0):
    f = np.array(np.array(i, dtype=np.float64)*s*(2**shift))
    neg = -((1 << 32)-1)
    pos = (1 << 32) - 1
    if f.size == 1:
        if np.isnan(f):
            return 0
        if np.isneginf(f):
            return neg
        if np.isposinf(f):
            return pos
    f[np.isnan(f)] = 0  # remove nan
    f[np.isneginf(f)] = neg
    f[np.isposinf(f)] = pos
    v = np.round(f).astype(np.int64)
    # print(i,f,v)
    return v


@register_gt(OpType.Quantize)
def quantize(self, *args, **kwargs):
    is_aiff = kwargs.get("is_aiff", False)
    # fuse = kwargs.get("fuse", False)
    if is_aiff and self.outputs[0].dtype.is_int and self.inputs[0].dtype.is_float and self.inputs[0].dtype.size > 1:
        # not support fp8
        DEBUG('run quantize: aiff version !')
        input_tensor = self.inputs[0].irtensor.double().cpu().numpy()

        if "quantize_scale" in self.constants:
            layer_top_scale = self.constants["quantize_scale"]
        else:
            layer_top_scale = self.outputs[0].quantization.scale
        shift = np.floor(np.log2(layer_top_scale))
        scale = layer_top_scale / 2 ** shift
        scale = min(round(scale * 128), 255)
        shift -= 7
        output = f16_to_int(input_tensor, scale, shift)
        if self.outputs[0].dtype.bits == 8 and self.outputs[0].dtype.is_signed:
            output[output > 32767] = 32767
            output[output < -32768] = -32768
        output = output - self.outputs[0].quantization.offset
        output = clamp(output, self.outputs[0].dtype).astype(self.outputs[0].dtype.np)
        self.outputs[0].irtensor = output
        return self.outputs[0].betensor
    else:
        DEBUG('run quantize: tpc version !')
        # tpc version
        round_mode = self.params.get("round_mode", "ROUND_TO_EVEN")
        round_mode_func = {
            "ROUND_TO_EVEN": torch.round,
            "TRUNC": torch.trunc,
            "FLOOR": torch.floor,
            "CEIL": torch.ceil,
        }
        if round_mode not in round_mode_func:
            FATAL("aipu quantize cannot support round_mode=%s", round_mode)
        DEBUG("quantize round_mode=%s", round_mode)
        input_tensor = self.inputs[0].irtensor
        ##### fma #####
        # refer:
        # https://jira.armchina.com/browse/CP-8589
        # https://jira.armchina.com/browse/CP-8844
        # http://szc-gerritai02.armchina.com:8080/c/AIPU_building_tool_design/+/55768
        output = input_tensor.double() * torch.tensor(self.outputs[0].quantization.scale).double()
        output = output - torch.tensor(self.outputs[0].quantization.offset).double()
        output = output.to(torch.float)
        ###############
        if self.outputs[0].dtype.is_int:
            output = round_mode_func[round_mode](output)
        # output = output.to(self.outputs[0].dtype.torch)
        if str(self.outputs[0].dtype) in ['float8_e5m2', 'float8_e4m3fn']:
            output = x3_aiff_fp32_2_fpx(output, fpx=self.outputs[0].dtype)
        else:
            output = clamp(output.cpu().numpy(), self.outputs[0].dtype).astype(self.outputs[0].dtype.np)
        self.outputs[0].irtensor = output
        return self.outputs[0].betensor
