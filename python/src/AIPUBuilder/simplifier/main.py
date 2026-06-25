# cython: language_level=3
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import sys
import time
import AIPUBuilder.plugin_loader
from AIPUBuilder.core import Graph, Dtype, check_ir, check_graph, BuilderConfig
from AIPUBuilder.core.utils import *
from AIPUBuilder.utils import *
from AIPUBuilder.core.transform import *
from AIPUBuilder._C._tools import gsim
from AIPUBuilder.logger import DEBUG, INFO, print_trace_stack, ERROR
from AIPUBuilder.simplifier.constant_folding import fold_constant
from AIPUBuilder.simplifier.plugin import get_pre_pass, get_post_pass

FULL_SIM_TIMEOUT = 300.0  # 300s


def simplify(g: Graph, debug=False) -> bool:
    pre_pass = get_pre_pass()
    post_pass = get_post_pass()
    ret = False
    for p in pre_pass:
        try:
            ret |= p(g)
        except Exception as e:
            ERROR(str(e))
    ret |= gsim(g, debug)
    try:
        while fold_constant(g):
            ret = True
    except Exception as e:
        ERROR(str(e))
        raise e
    for p in post_pass:
        try:
            ret |= p(g)
        except Exception as e:
            ERROR(str(e))
    return ret


def full_simplify(g: Graph, debug=False):
    try:
        t0 = time.time()
        while simplify(g, debug):
            t1 = time.time()
            if t1 - t0 > FULL_SIM_TIMEOUT:
                DEBUG(
                    "full_simplify is timeout(run more than %d seconds)!"
                    % FULL_SIM_TIMEOUT
                )
                return
    except KeyboardInterrupt:
        simplify(g, debug)
        return


def statistic_nodes_num(g: Graph):
    ret = dict()
    for n in g.nodes:
        ret[n.type] = ret.get(n.type, 0) + 1
    return ret


def has_weights(g: Graph):
    for n in g.nodes:
        if len(n.constants):
            return True
    return False


def main():
    from AIPUBuilder import simplifier

    args = argparse.ArgumentParser()
    args.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s version " + simplifier.__VERSION__,
    )
    args.add_argument(
        "graph", metavar="<graph.def>", type=str, help="graph definition file."
    )
    args.add_argument(
        "-w", "--weight", type=str, default="", help="the weight file for the graph."
    )
    args.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.def",
        help="the optimized graph definition file.",
    )
    args.add_argument(
        "-b",
        "--bin",
        type=str,
        default="output.cbin",
        help="the optimized graph weight file.",
    )
    args.add_argument(
        "-f",
        "--full",
        action="store_true",
        help="try to full optimize the graph by trying to find the fixed point with the known optimization passes. Please note that it may take an incalculable amount of time!!.",
    )
    args.add_argument(
        "--disable_pass",
        type=str,
        default="",
        help="Disable pass(es) by name. If you have several passes, use ',' to separate them",
    )
    args.add_argument(
        "--remove_batch",
        action="store_true",
        help="remove the batch if the model batch!=1 and the batch can be removed.",
    )
    args.add_argument(
        "-d", "--debug", action="store_true", help="show more debug info on output ir."
    )
    options = args.parse_args(sys.argv[1:])
    from AIPUBuilder.core import Graph

    g = Graph.parse(options.graph, options.weight)
    debug = False
    if g.attrs.get("compat_quantized_model", False):
        DEBUG(
            "compat quantized model is True, will not perform gsim, please use gsim after quant."
        )
        g.serialize_scale_zp = True
        g.serialize(options.output, options.bin)
        INFO("Simplify Done.")
        return
    # when no weight, we disbale the CSE, due to node compare depend weights data
    if len(options.weight) == 0 and has_weights(g):
        DEBUG("CSE optimization is disabled, due to missing weights data.")
        disable_pass("CSE_subnode")
    for p in options.disable_pass.split(","):
        disable_pass(p)
    if options.debug is not None and options.debug:
        BuilderConfig.set("show_location", True)
        debug = True

    if options.remove_batch is not None and options.remove_batch:
        remove_batch(g)
    before = statistic_nodes_num(g)
    INFO("Start to simplify the graph...")
    simplify(g, debug)
    if options.full:
        INFO("Using fixed-point full optimization, it may take long long time ....")
        full_simplify(g, debug)
    g.topological_sort()
    g.reset_layer_id()

    after = statistic_nodes_num(g)
    if options.remove_batch is not None and options.remove_batch:
        remove_batch(g)
    header = "GSim simplified result:\n" + "-" * 72 + "\n"
    s = ""
    for c in after:
        d = 0
        if c in before:
            d = before.pop(c)
        d = after[c] - d
        if d != 0:
            s += "\t%20s: %+4d\n" % (str(c), d)
    for c in before:
        s += "\t%20s: %+4d\n" % (str(c), 0 - before.get(c, 0))
    if len(s):
        s += "-" * 72
        INFO(header + s)
    with RaiseCRuntimeError():
        if not check_graph(g):
            g.serialize_scale_zp = True
            BuilderConfig.set("show_location", True)
            g.serialize(options.output + ".failed", options.bin + ".failed")
            raise Exception(
                "GSim failed to simplify the graph due to pass or graph issue. The failed IR is saved:%s and %s"
                % (options.output + ".failed", options.bin + ".failed")
            )
    INFO("Simplify Done.")
    g.serialize_scale_zp = True
    g.serialize(options.output, options.bin)


if __name__ == "__main__":
    main()
