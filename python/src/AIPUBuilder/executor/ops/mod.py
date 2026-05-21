import torch
from .gt_util import *
from ..engine import register_gt, register_gen_input
from AIPUBuilder.core import OpType
from AIPUBuilder.logger import WARN


@register_gt(OpType.Mod)
def Mod(self, *args, **kwargs):
    """
    if fmod=true, The positive and negative moduli are the same as the dividend, independent of the divisor.
    np.fmod(-7, 4)=-3，np.mod(-7, 4)=1
    """
    if self.inputs[0].dtype.is_float:
        output = mod_float(self)
    else:
        input1 = self.inputs[0].betensor.cpu().numpy().astype(np.int64)
        input2 = self.inputs[1].betensor.cpu().numpy().astype(np.int64)
        fmod = self.params['fmod'] if 'fmod' in self.params.keys() else 'false'
        scale_value = self.params['scale_value']
        shift_value = self.params['shift_value']

        dividend_tensor = clamp(np.round(input1 * scale_value[0] * (0.5 ** shift_value[0])), dtype=self.inputs[0].dtype)
        divisor_tensor = clamp(np.round(input2 * scale_value[1] * (0.5 ** shift_value[1])), dtype=self.inputs[0].dtype)

        if 'fmod' in self.params.keys() and str(fmod).lower() in ['true', '1']:
            output = np.fmod(dividend_tensor, divisor_tensor)
        else:
            output = np.mod(dividend_tensor, divisor_tensor)
        output = clamp(output, self.outputs[0].dtype)

    self.outputs[0].betensor = output
    return [o.betensor for o in self.outputs]


def mod_float(self):
    fmod = self.params.get('fmod', 'false')
    dividend = self.inputs[0].betensor.float()
    divisor = self.inputs[1].betensor.float()
    out = _func_map[fmod](dividend, divisor)
    return out


def torch_mod(a, b):
    # return a - a.div(b, rounding_mode='floor') * b
    try:
        ret = a - a.div(b, rounding_mode='floor') * b
    except Exception as e:
        WARN(f"{e} in mod op.")
        b[b == 0] = 1
        ret = a - a.div(b, rounding_mode='floor') * b
    return ret


def torch_fmod(a, b):
    try:
        ret = torch.fmod(a, b)
    except Exception as e:
        WARN(f"{e} in mod op.")
        b[b == 0] = 1
        ret = torch.fmod(a, b)
    return ret


_func_map = {
    True: torch_fmod,
    False: torch_mod,
}


@register_gen_input(OpType.Mod)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)

    if self.inputs[0].dtype.is_float:
        input0 = (np.random.uniform(-10000, 10000, size=in_shape0)).astype(in_type0)
        input1 = (np.random.uniform(-1000, 1000, size=in_shape1)).astype(in_type1)
        index = np.abs(input1) < 1  # avoid div error with lib
        input1[index] = np.float16(np.random.uniform(1, 1000)).astype(in_type1)
    else:
        input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)
        input1 = np.random.randint(np.iinfo(in_type1).min, np.iinfo(in_type1).max, size=in_shape1, dtype=in_type1)

    input_data = [input0, input1]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    return input_data
