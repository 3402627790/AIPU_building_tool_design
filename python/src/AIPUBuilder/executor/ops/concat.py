import torch
from ..engine import register_gt
from .gt_util.quant_utils import clamp
from AIPUBuilder.core import OpType, DataLayout


@register_gt(OpType.Concat)
@register_gt(OpType.InternalConcatenated)
def concat(self, *args, **kwargs):
    scale_value = self.params.get('scale_value', [256] * len(self.inputs))
    shift_value = self.params.get('shift_value', [8] * len(self.inputs))

    input_data = []
    for i in range(len(self.inputs)):
        if self.outputs[0].dtype.is_float:
            input_data.append(self.inputs[i].irtensor.float())
        else:
            input = self.inputs[i].irtensor.clone().double()
            if self.type.name == 'Concat':
                input = input + self.inputs[i].quantization.offset
                input = torch.round(input * scale_value[i] * 0.5 ** shift_value[i])
            input_data.append(input)
    output = torch.cat(tuple(input_data), dim=self.params['axis'])

    if self.type.name == 'Concat':
        output -= self.outputs[0].quantization.offset
    output = clamp(output, self.outputs[0].dtype)
    self.outputs[0].irtensor = output.to(self.outputs[0].dtype.torch)

    return self.outputs[0].betensor


def concat_float(self, *args, **kwargs):
    pass
