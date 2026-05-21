# cython: language_level=3
#!/usr/bin/python
# -*- coding: UTF-8 -*-

from AIPUBuilder.core import Graph, OpType, Node, Tensor
from AIPUBuilder.executor import *


def fold_constant(g: Graph) -> bool:
    exe = Executor("_GT")
    ret = False
    while True:
        full_fold = True
        remove_nodes = []

        for n in g.nodes[:]:
            if len(n.parents) and all([p.type == OpType.Constant for p in n.parents]):
                temp_nodes = []
                if n.is_dynamic():
                    continue
                try:
                    for p in n.parents:
                        if len(p.children) <= 1:
                            temp_nodes.append(p)
                        exe.forward(p)
                    exe.forward(n)
                except Exception as e:
                    print(e)
                    continue
                remove_nodes += temp_nodes
                full_fold = False
                ret = True
                for p in n.parents[:]:
                    n.remove_input(p.outputs[0])
                remove_nodes.append(n)
                for i, out in enumerate(list(n.outputs)):
                    c = Node(g.get_valid_node_name("const"), OpType.Constant)
                    out.betensor.cpu().to(out.dtype.torch)
                    out.sync_betensor()
                    c.constants["weights"] = Tensor(out)
                    n.remove_output(out)
                    c.add_output(out)
                    g.add_node(c)
        for node in set(remove_nodes):
            g.remove_node(node)
        if full_fold:
            break
    return ret
