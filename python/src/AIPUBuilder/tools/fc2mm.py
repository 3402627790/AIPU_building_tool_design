
import sys
import os
import argparse
from AIPUBuilder import ops
from AIPUBuilder.logger import *
from AIPUBuilder.core import *
from AIPUBuilder.simplifier.main import *


def fc2mm_(g, fc):
    with g:
        w = fc.constants["weights"]
        b = fc.constants["biases"]
        wi = ops.input(w.shape, w.dtype)
        out = fc.outputs[0]
        inp = fc.inputs[0]
        mm = ops.matmul(inp, wi)
        if not all(b.numpy() == 0):
            b = ops.constant(b)
            mm = mm + b
        fc.remove_output(out)
        n = mm.op
        n.replace_output_by(mm, out)
        g.remove_node(fc)


def criteria(n):
    ret = n.type == OpType.FullyConnected
    ret = ret and n.inputs[0].dtype in [
        Dtype.FP32, Dtype.FP16, Dtype.BFP16]  # only support fp32/fp16/bfp16
    ret = ret and n.constants["weights"].dtype in [
        Dtype.FP32, Dtype.FP16, Dtype.BFP16]  # only support fp32/fp16/bfp16
    return ret


def fc2mm(g, nodes=[], out_name="new_graph"):
    is_changed = False
    for n in g.nodes:
        if n.name in nodes:
            if criteria(n):
                INFO(f"transforming FC {n.name} to matmul")
                fc2mm_(g, n)
                is_changed = True
    if is_changed:
        simplify(g)
        g.topological_sort()
    else:
        INFO(f"Nothing changed!!!")
    INFO(f"serialized {out_name}.txt and {out_name}.bin")
    g.serialize(f"{out_name}.txt", f"{out_name}.bin")
    return g


def main():
    args = argparse.ArgumentParser()
    args.add_argument('-v', '--version', action='version',
                      version='%(prog)s version ' + AIPUBuilder.__VERSION__)
    args.add_argument("graph", metavar='<graph.def>',
                      type=str, help='graph definition file.')
    args.add_argument('-w', '--weight', type=str, default="",
                      help='the weight file for the graph.')
    args.add_argument('-n', '--lora_nodes', type=str, default="",
                      help='the names of the lora FullyConnected node to convert to matmul, use comma "," to split if you have several nodes.')
    args.add_argument('-o', '--output', type=str, default="new_out",
                      help='the output IR file name for .txt and .bin. (default: "new_out")')
    options = args.parse_args(sys.argv[1:])
    g = Graph.parse(options.graph, options.weight)
    if options.lora_nodes:
        nodes = lora_nodes.split(',')
        fc2mm(g, nodes, options.output)
    else:
        INFO("no lora node to convert, nothing to do.")


if __name__ == "__main__":
    main()
