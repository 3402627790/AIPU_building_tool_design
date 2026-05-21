import torch
from ..engine import register_gt, register_gen_input, _GEN_INPUTS
import numpy as np
from ..engine import _subgraph, _ALL_ENGINES
from AIPUBuilder.core import OpType, SubGraphNode


@register_gt(OpType.Loop)
def loop(self, *args, **kwargs):
    node = SubGraphNode.from_node(self)
    header_node = [x for x in node.subgraph.nodes if x.type == OpType.ForRangeHeader][0]
    ender_node = [x for x in node.subgraph.nodes if x.type == OpType.ForRangeEnder][0]
    depend_input_num = self.params["dependency_inputs_num"]
    scan_output_num = self.params["scan_outputs_num"]
    ender_node.attrs["dependency_inputs_num"] = depend_input_num
    ender_node.attrs["scan_outputs_num"] = scan_output_num

    max_count = header_node.inputs[0]
    cond_in = header_node.inputs[1]
    scan_len = header_node.outputs[2]
    iter_count = header_node.outputs[4]
    cond_out = ender_node.inputs[6 + depend_input_num]

    # init loop params and outputs
    scan_len.betensor = torch.zeros_like(scan_len.betensor)
    iter_count.betensor.zero_()
    cond_out.betensor = cond_in.betensor
    for i in range(depend_input_num):
        ender_node.outputs[i].betensor = header_node.inputs[i + 2].betensor

    while cond_out.betensor.item() != 0 \
            and iter_count.betensor.item() < max_count.betensor.item():
        _subgraph("_GT")(node, *args, **kwargs)
        for i in range(depend_input_num):
            header_node.inputs[i + 2].betensor = ender_node.outputs[i].betensor


@register_gt(OpType.ForRangeHeader)
def for_range_header(self, *args, **kwargs):
    self.outputs[3].betensor = self.inputs[0].betensor
    for it, ot in zip(self.inputs[1:], self.outputs[5:]):
        ot.betensor = it.betensor


@register_gt(OpType.ForRangeEnder)
def for_range_ender(self, *args, **kwargs):
    depend_input_num = self.attrs["dependency_inputs_num"]
    # scan_output_num = self.attrs["scan_outputs_num"]

    scan_len = self.inputs[2]
    iter_count = self.inputs[4]

    # map depend outputs
    for i in range(depend_input_num):
        self.outputs[i].betensor = self.inputs[i + 7 + depend_input_num].betensor

    # update scan outputs
    for i in range(depend_input_num, len(self.outputs)):
        out = self.outputs[i]
        inp = self.inputs[i + 7 + depend_input_num]
        val = inp.betensor
        if len(inp.betensor.shape) < len(out.betensor.shape):
            val = val.reshape(1, *val.shape)
        if iter_count.betensor.item() == 0:
            out.betensor = val
        else:
            out.betensor = torch.cat((out.betensor, val), axis=0)
        scan_len.betensor[i - depend_input_num] += val.shape[0]

    # update iter count
    iter_count.betensor += 1
