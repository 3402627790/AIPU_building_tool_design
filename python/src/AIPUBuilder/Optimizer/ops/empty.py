# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import *
import torch

register_optype('Empty')

@op_register(OpType.Empty)
def empty__forward(self, *args):
    ds = self.outputs[0].ir_shape
    ds_list = [t for t in ds]
    ds_list[0] = 0
    ds = TensorShape(ds_list)
    if 'ds_output_shape' in self.params:
        ds = self.params['ds_output_shape'][0]
    self.outputs[0].betensor = torch.zeros(ds)
    return self.outputs[0].betensor

@quant_register(OpType.Empty)
def empty__quantize(self, *args):
    q_bits_activation = self.attrs["q_bits_activation"]
    q_mode_activation = self.attrs["q_mode_activation"]
    dtype = bits2dtype(q_bits_activation, True)
    self.outputs[0].dtype = dtype
    self.outputs[0].scale = 1.0
    self.outputs[0].zerop = 0
    self.outputs[0].qbits = q_bits_activation
    self.outputs[0].qinvariant = True
    self.outputs[0].qmin = dtype2range(dtype)[0]
    self.outputs[0].qmax = dtype2range(dtype)[1]
