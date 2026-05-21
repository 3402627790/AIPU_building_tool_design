# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import torch
from torch.fx import symbolic_trace, Graph, Node, GraphModule
import copy
from typing import Dict, Callable
import operator


class SimpleTest(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x + 3.0)


def test_fx():
    st = SimpleTest()
    traced = symbolic_trace(st)
    traced.graph.lint()

    def transform(traced):
        new_graph = torch.fx.Graph()
        val_map: Dict[Node, Node] = {}
        output_value = new_graph.graph_copy(traced.graph, val_map)
        relu_out = new_graph.create_node(
            op='call_function', target=torch.reshape, args=(output_value, (-1,)), kwargs={})
        new_graph.output(relu_out)
        return GraphModule(traced, new_graph)
    traced.graph.print_tabular()
    transformed = transform(traced)
    transformed.graph.lint()
    transformed.graph.print_tabular()
    copied = copy.deepcopy(transformed)
    print(id(type(transformed)), id(type(copied)))
    x = torch.randn(3, 4)
    print(copied(x), transformed(x))
    kv = dict(transformed.named_modules())
    for k, v in kv.items():
        print(k, v)
    for n in transformed.graph.nodes:
        print(n.op, n.target, n.name)


# test_fx()
# print(test_fx)

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        # [1, 4]
        outs = torch.split(inputs, [2, 2], dim=-1)
        or0 = outs[0].reshape([1, 1, -1])
        or1 = or0[:, :, 0:1]
        or1 = outs[1]
        or1 = or1[0, :]
        # or1 = outs[1].reshape([-1])
        # return or0, or1
        return outs, or0


m = M()
traced = symbolic_trace(m)

print(traced.graph.print_tabular())
print(traced.code)

for node in traced.graph.nodes:
    if node.op == 'call_function' and node.target == operator.getitem:
        print(node)
        for arg in node.args:
            if isinstance(arg, tuple):
                for ar in arg:
                    if isinstance(ar, slice):
