from ..engine import register_gt, register_gen_input, _GEN_INPUTS
from AIPUBuilder.core import OpType
import numpy as np


@register_gt(OpType.Input)
@register_gt(OpType.SubGraphInput)
def input_node(self, *args, **kwargs):
    return self.outputs[0].betensor


# generate input for general ops
@register_gen_input(OpType.Input)
def gen_input_data(self):
    input_datas = []
    for index, op in enumerate(self.inputs):
        if op._attrs.get('is_assigned', False):
            continue
        in_dtype = str(op.dtype)
        in_shape = list(op.shape)
        if 'int' in in_dtype:
            input_data = np.random.randint(np.iinfo(in_dtype).min,
                                           np.iinfo(in_dtype).max,
                                           size=in_shape,
                                           dtype=in_dtype)
        else:
            # input_data = np.random.randn(*in_shape) * 1000
            input_data = np.random.uniform(-100, 100, size=in_shape).astype(op.dtype.np)
            # input_data = np.float16(input_data)
            # if 'bfloat16' in in_dtype:
            #     from ml_dtypes import bfloat16
            #     input_data = input_data.astype(bfloat16)
            # elif in_dtype.endswith('e5m2'):
            #     from ml_dtypes import float8_e5m2
            #     input_data = input_data.astype(float8_e5m2)
            # elif in_dtype.endswith('e4m3fn'):
            #     from ml_dtypes import float8_e4m3fn
            #     input_data = input_data.astype(float8_e4m3fn)
        input_datas.append(input_data)
        self.inputs[index].betensor = input_data
    return input_datas
