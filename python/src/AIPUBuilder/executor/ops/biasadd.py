from ..engine import register_gt, register_gen_weight
from AIPUBuilder.core import Target
from .gt_util.quant_utils import clamp, _tpc_lut
from AIPUBuilder.core import OpType
import numpy as np
import torch


@register_gt(OpType.BiasAdd)
def bias_add(self, *args, **kwargs):
    if str(self.inputs[0].dtype) in ['float16', 'bfloat16']:
        outputs = biasadd_float(self)
        self.outputs[0].betensor = outputs
    else:
        biases = self.constants["biases"].betensor.cpu().numpy().astype("int64")
        output_type = str(self.outputs[0].dtype)
        shift_num = 16 if self.inputs[0].dtype.bits == 16 else 24
        shift = self.params["shift_value"]
        scale = self.params["scale_value"]
        input = self.inputs[0].betensor.cpu().numpy().astype(np.int64)
        biases = biases.astype(np.int64)
        # 8 bit input need << 24, which actually represents a 32-bit number
        input_shift = np.left_shift(input, shift_num)
        bias_add = input_shift + biases
        bias_add = bias_add.astype(np.float32)
        bias_add_shift = bias_add / (2 ** shift_num)
        bias_add_shift = np.round(bias_add_shift)
        outputs = bias_add_shift
        outputs = clamp(outputs, self.outputs[0].dtype)
        self.outputs[0].betensor = torch.from_numpy(outputs)


def biasadd_float(self):
    inp0 = self.inputs[0].betensor.to(torch.float32)
    biases = self.constants['biases'].betensor
    outputs = inp0 + biases
    outputs = outputs.to(self.outputs[0].dtype.torch)
    return outputs


@register_gen_weight(OpType.BiasAdd)
def gen_random_weight(self):
    biases_type = str(self.constants['biases'].dtype)
    biases_shape = list(self.constants['biases'].shape)
    if 'float' in biases_type:
        biases = np.float32(np.random.randn(*biases_shape) * 100000)
    else:
        biases = np.random.randint(
            0, 255, size=self.constants['biases'].shape, dtype=str(self.constants['biases'].dtype))
    self.constants['biases'].betensor = biases
