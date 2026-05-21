import torch
import numpy as np
from ..engine import register_gt
from .gt_util.quant_utils import clamp
from .gt_util.layout_transfer import check_out_datalayout
from AIPUBuilder.core import OpType


@register_gt(OpType.InternalConstant)
@register_gt(OpType.Constant)
def constant(self, *args, **kwargs):
    out_type = str(self.outputs[0].dtype)
    if self.outputs[0].dtype.is_float:
        output = self.constants['weights'].numpy()
    else:
        input_data = self.constants['weights'].betensor.long()
        # origin_layout = self.constants['weights'].shape.layout.name

        scale_value = self.params.get('scale_value', 1)
        shift_value = self.params.get('shift_value', 0)
        output = torch.round(input_data * scale_value * (0.5 ** shift_value))
        output = torch.clip(output, np.iinfo(out_type).min, np.iinfo(out_type).max)
        if torch.Size(self.outputs[0].shape) != input_data.shape:
            output = output.reshape(tuple(self.outputs[0].shape))
    self.outputs[0].betensor = output
