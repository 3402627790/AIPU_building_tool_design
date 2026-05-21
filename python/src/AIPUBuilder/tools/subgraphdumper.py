import sys
import argparse
import AIPUBuilder
from AIPUBuilder.core import *


def legalize_name(name):
    return name.replace(" ", "_").replace("-", "_").replace(":", "_").replace("/", "_")


def dump_subgraph_ir(g, recursive=True, prefix="dump_sg_", selected_nodes=None):
    for n in g.nodes:
        if n.type in [OpType.Loop, OpType.If]:
            if selected_nodes is not None and n.name not in selected_nodes:
                continue
            sg = SubGraphNode(n).extract_subgraph(n.name)
            dump_subgraph_ir(sg, recursive)
            ir, bin = sg.serialize()
            with open(f"{prefix}{legalize_name(n.name)}.txt", "w") as f:
                f.write(ir)
            with open(f"{prefix}{legalize_name(n.name)}.bin", "wb") as f:
                f.write(bin)


def main():
    args = argparse.ArgumentParser()
    args.add_argument('-v', '--version', action='version',
                      version='%(prog)s version ' + AIPUBuilder.__VERSION__)
    args.add_argument("graph", metavar='<graph.def>',
                      type=str, help='graph definition file.')
    args.add_argument('-w', '--weight', type=str, default="",
                      help='the weight file for the graph.')
    args.add_argument('-r', '--recursive', action="store_true",
                      help="recursive dump all subgraphs of selected subgraphs.")
    args.add_argument('-sg', '--subgraph', type=str, default="",
                      help='the name of the subgraph node to dump, use comma "," to split if need dump more than one subgraphs. If not specified, all subgraphs will be dumped.')
    args.add_argument('--prefix', type=str, default="dump_sg_",
                      help='the prefix of the dump IR file name. (default: "dump_sg_")')
    options = args.parse_args(sys.argv[1:])
    g = Graph.parse(options.graph, options.weight)
    subgraphs = None
    if options.subgraph:
        subgraphs = options.subgraph.split(',')

    recursive = False
    if options.recursive:
        recursive = True
    dump_subgraph_ir(g, subgraphs=subgraphs, prefix=options.prefix, recursive=recursive)


if __name__ == "__main__":
    main()
