from ..engine import register_gt
from .gt_util.quant_utils import clamp
from .gt_util.dtype_utils import torch_to_numpy
from AIPUBuilder.core import OpType
import torch
import numpy as np


@register_gt(OpType.Cast)
def cast(self, *args, **kwargs):
    if self.inputs[0].dtype.is_float:
        inp = torch_to_numpy(self.inputs[0].betensor, self.inputs[0].dtype.np).astype('float64')
        if self.outputs[0].dtype.is_int:
            clamp_min, clamp_max = np.iinfo(self.outputs[0].dtype.np).min, np.iinfo(self.outputs[0].dtype.np).max
            output = np.clip(inp, clamp_min, clamp_max).astype(self.outputs[0].dtype.np)
        else:
            output = inp.astype(self.outputs[0].dtype.np)
        self.outputs[0].betensor = output
        return self.outputs[0].betensor

    else:
        input_tensor = self.inputs[0].betensor.long()
        if self.outputs[0].dtype.is_int:
            scale_value = self.params.get('scale_value', default=None)
            shift_value = self.params.get('shift_value', default=None)
            if scale_value is not None and str(scale_value).lower() != 'false' \
                    and shift_value is not None and str(shift_value) != 'false':
                input_tensor = input_tensor + self.inputs[0].quantization.offset
                output = torch.round(input_tensor * scale_value * (0.5 ** shift_value))
                output -= self.outputs[0].quantization.offset
            else:
                ignore_scale_zp = self.params.get('ignore_scale_zp', default=None)
                clip_mode = self.params.get('clip_mode', default='SATURATION')
                if clip_mode.upper() == 'TRUNCATION':
                    # trancation does not need consider in_zp/out_zp
                    # torch do not support uint16/uint32, so use numpy
                    output = self.inputs[0].betensor.cpu().numpy().astype(self.outputs[0].dtype.np)
                    output = torch.from_numpy(output.astype('int64')) if str(self.outputs[0].dtype.np).endswith(
                        ('int32', 'int64')) else torch.from_numpy(output.astype('float64'))
                    output = output.to(input_tensor.device)
                else:
                    in_zp = 0 if ignore_scale_zp else self.inputs[0].quantization.offset
                    out_zp = 0 if ignore_scale_zp else self.outputs[0].quantization.offset
                    output = input_tensor + in_zp - out_zp
            output = clamp(output, self.outputs[0].dtype)
            self.outputs[0].betensor = output.to(self.outputs[0].dtype.torch)
        else:
            self.outputs[0].betensor = input_tensor.to(self.outputs[0].dtype.torch)
        return self.outputs[0].betensor
