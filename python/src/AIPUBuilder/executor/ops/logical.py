from ..engine import register_gt
from AIPUBuilder.core import OpType
import torch
import numpy as np


@register_gt(OpType.Logical)
def logical(self, *args, **kwargs):
    """
    1. no ref in IR & input_shape=[[a,b,c,d],[1,1]]: compare input_0 with input_1(a number)
    2. no ref in IR & input_shape=[[a,b,c,d],[a,b,c,d]]: compare input_0 with input_1(an array)
    3. ref in IR: only 1 input, input_shape=[a,b,c,d]: compare input_0 with ref
    4. method = NOT, only 1 input, input_shape=[a,b,c,d]: out = ! input
    """
    input0 = self.inputs[0].irtensor
    input_shape = input0.shape
    method = self.params["method"]
    if method == "NOT":
        input1 = torch.tensor(0, dtype=torch.int32)
    else:
        input1 = self.inputs[1].irtensor if len(self.inputs) == 2 else self.params["ref"].irtensor
    if isinstance(input1, int):
        input1 = torch.tensor(input1, dtype=torch.int32)
        input1 = torch.tile(input1, input_shape)

    input1_dtype = self.inputs[1].dtype if len(self.inputs) == 2 else self.inputs[0].dtype
    if (not self.inputs[0].dtype.is_float) and (not input1_dtype.is_float):
        if "scale_value" in self.params.keys():
            scale = self.params["scale_value"]
            if isinstance(scale, list):
                scale0 = scale[0]
                scale1 = scale[1] if len(scale) == 2 else scale[0]
            else:
                scale0 = scale
                scale1 = scale
        else:
            scale0 = 1
            scale1 = 1

        # scale/shift only work on input2 from 202106: input1 （>,<,==...） input2 * scale >> shift
        # scale/shift work both on input1 & input2 from 202107
        input0 = input0.int()
        input0 = input0 + self.inputs[0].quantization.offset
        input0 = input0 * scale0
        if not method == "NOT":
            input1 = input1.int()
            input1 = input1 + self.inputs[1].quantization.offset
            input1 = input1 * scale1

    method = method
    if method.upper() == "EQUAL":
        output = input0 == input1
    elif method.upper() == "NOT_EQUAL":
        output = input0 != input1
    elif method.upper() == "GREATER":
        output = input0 > input1
    elif method.upper() == "GREATER_EQUAL":
        output = input0 >= input1
    elif method.upper() == "LESS":
        output = input0 < input1
    elif method.upper() == "LESS_EQUAL":
        output = input0 <= input1
    elif method.upper() == "OR":
        output = torch.logical_or(input0, input1)
    elif method.upper() == "AND":
        output = torch.logical_and(input0, input1)
    elif method.upper() == "XOR":
        output = torch.logical_xor(input0, input1)
    elif method.upper() == "NOT":
        output = torch.logical_not(input0)
    else:
        raise TypeError("do not support method %s" % method)

    output = output + 0  # bool -> 0,1
    output[output != 0] = 1
    self.outputs[0].irtensor = output

    return self.outputs[0].betensor
