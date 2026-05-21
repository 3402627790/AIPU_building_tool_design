# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.utils import *

register_optype('NonZero')


@op_register(OpType.NonZero)
def nonzero(self, *args):
    self.outputs[0].betensor = torch.nonzero(self.inputs[0].betensor)
    self.outputs[0].betensor = self.outputs[0].betensor.permute(1, 0)
    self.outputs[1].betensor = torch_tensor(self.outputs[0].betensor.shape[1], self.outputs[0].device).reshape([1])
    return self.outputs[0].betensor, self.outputs[1].betensor


@quant_register(OpType.NonZero)
def nonzero_quantize(self, *args):
    inp = self.inputs[0]
    q_bits_activation = self.attrs["q_bits_activation"]
    max_len = max(list(inp.ir_shape))
    max_qbits = torch.log2(torch.tensor(max_len)).ceil()
    signed_list = [False, self.force_dtype_int]
    #outputs[1].dtype only support int32 or uint32
    qbits_list = [max(q_bits_activation, max_qbits), 32]

    for idx, out in enumerate(self.outputs):
        out.scale = 1.0
        out.zerop = 0
        out.qbits = qbits_list[idx]
        out.dtype = bits2dtype(out.qbits, is_signed=signed_list[idx])
        out.qinvariant = True
        out.qmin, out.qmax = dtype2range(out.dtype)