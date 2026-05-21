from ..engine import register_gt
from AIPUBuilder.core import OpType, ComputeTensor, Dtype
from AIPUBuilder.logger import FATAL, DEBUG
import numpy as np
import torch


def int16_to_fp32(i, shift=0):
    v = i/(2**shift)
    v = np.ascontiguousarray(v)
    a = ComputeTensor(v).astype(Dtype.FP32)
    # positive zero only
    ret = a.numpy()
    msk = ret == 0
    ret[msk] = 0
    return ret


@register_gt(OpType.DeQuantize)
def dequantize(self, *args, **kwargs):
    is_aiff = kwargs.get("is_aiff", False)
    fuse = kwargs.get("fuse", False)
    if is_aiff and self.inputs[0].dtype.is_int and self.outputs[0].dtype.is_float and self.outputs[0].dtype.size > 1:
        # not support fp8
        DEBUG('run dequantize: aiff version !')
        if not fuse:
            input_tensor = self.inputs[0].irtensor.cpu().numpy().astype("int32")

            input_tensor = input_tensor + self.inputs[0].quantization.offset
            if "quantize_scale" in self.constants:
                deqauntize_scale = self.constants["quantize_scale"]
            else:
                deqauntize_scale = self.inputs[0].quantization.scale
            shift_of_deqauntize_scale = np.floor(np.log2(deqauntize_scale))
            scale_of_deqauntize_scale = deqauntize_scale / 2**shift_of_deqauntize_scale
            new_scales = (1/2**0) / scale_of_deqauntize_scale

            shifts_of_new_scales = np.floor(np.log2(new_scales))
            scales_of_new_scales = new_scales / 2**shifts_of_new_scales
            scales_of_new_scales = np.clip(np.round(scales_of_new_scales * 128), 0, 255)
            shifts_of_new_scales -= 7

            input_tensor = input_tensor * scales_of_new_scales
            input_tensor = np.round(input_tensor * 2**shifts_of_new_scales)
            # input_tensor + zp may overflow input dtype
            if np.max(input_tensor) > np.iinfo(self.inputs[0].dtype.np).max or np.min(input_tensor) < np.iinfo(self.inputs[0].dtype.np).min:
                if self.inputs[0].dtype.is_signed or np.min(input_tensor) < 0:
                    input_tensor = input_tensor.astype(np.int16)
                else:
                    input_tensor = input_tensor.astype(np.uint16)
            else:
                input_tensor = input_tensor.astype(self.inputs[0].dtype.np)
            output = int16_to_fp32(input_tensor, shift_of_deqauntize_scale).astype(self.outputs[0].dtype.np)

            self.outputs[0].irtensor = output
            return self.outputs[0].betensor
        else:
            raise TypeError("not support aiff op fuse dequant yet")
    else:
        DEBUG('run dequantize: tpc version !')
        input_tensor = self.inputs[0].irtensor
        output = (input_tensor.double() + torch.tensor(self.inputs[0].quantization.offset).double()) * (1.0 / torch.tensor(self.inputs[0].quantization.scale))
        output = output.to(self.outputs[0].dtype.torch)
        self.outputs[0].irtensor = output
        return self.outputs[0].betensor
        # tpc
