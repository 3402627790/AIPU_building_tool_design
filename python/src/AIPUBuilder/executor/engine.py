# -*- coding: UTF-8 -*-
# cython: language_level=3
from AIPUBuilder.logger import DEBUG


# a engine is a node/op forward set
_ALL_ENGINES = dict()
_GEN_INPUTS = dict()
_GEN_WEIGHTS = dict()
_OP_METRICS = dict()


def _subgraph(engine, *args, **kwargs):

    def f(node, *args, **kwargs):
        from AIPUBuilder.core import OpType, Target, SubGraphNode
        target_name = Target().name
        if not SubGraphNode.is_subgraph_node(node):
            return
        is_plugin = "_plugin" in node.attrs and node.attrs["_plugin"] is not None
        in_tensors = {t.name: t for t in node.inputs}
        if len(in_tensors) == 0:
            in_tensors = {t.name: t for t in node.graph.input_tensors}
        else:
            for inpnode in node.subgraph.input_nodes:
                for inp in inpnode.inputs:
                    if inp.name in in_tensors:
                        inp.betensor = in_tensors[inp.name].betensor
                        in_tensors.pop(inp.name)
            # if len(in_tensors) != 0:
            #     raise Exception("Can not find input tensor of sub node in Subgraph node.")
        plugin_name = ''
        if is_plugin:
            plugin_name = node.attrs["_plugin"].name

        tensor_map = dict()
        sgattrs = node.attrs  # we can use {if 'sgattrs' in kwargs: sgattrs=kwargs['sgattrs']} to capture in each obs
        is_if = node.type == OpType.If
        dummy_index = 0
        cond = False
        if is_if:
            header = [op for op in node.subgraph.nodes if op.type == OpType.IfHeader][0]
            ender = [op for op in node.subgraph.nodes if op.type == OpType.IfEnder][0]
            dummy = ender.inputs[2].op
            dummy_index = [n for n in node.subgraph.nodes].index(dummy)
            cond = bool(header.inputs[0].betensor.item())

        for n in node.subgraph.nodes:
            if is_if:
                index = list(node.subgraph.nodes).index(n)
                if cond and index > dummy_index and n.type != OpType.IfEnder:
                    continue
                elif not cond and index < dummy_index and n.type != OpType.IfHeader:
                    continue
            if n.type == OpType.If:
                n = SubGraphNode.from_node(n)
                f(n)
                for ot in n.outputs:
                    tensor_map[ot.name] = ot
                    in_tensors[ot.name] = ot
                continue

            for i in n.inputs:
                if i.name in in_tensors and (i.betensor.double().cpu().numpy() != in_tensors[i.name].betensor.double().cpu().numpy()).any():
                    i.betensor = in_tensors[i.name].betensor
            fuse = False
            is_aiff = False
            if is_plugin and (len(n.children) > 0 or len(n.parents) > 0):
                fuse = True
            if plugin_name.startswith('Hwa'):
                is_aiff = True
            if 'layer_id' in n.attrs:
                DEBUG(f"[{n.attrs['layer_id']}] forward layer type {str(n.type).replace('OpType.', '')}: {n.name}")
            else:
                DEBUG(f"forward layer type {str(n.type).replace('OpType.', '')}: {n.name}")
            if target_name in _ALL_ENGINES[engine][n.type]:
                _ALL_ENGINES[engine][n.type][target_name](
                    n, fuse=fuse, is_aiff=is_aiff, plugin_name=plugin_name, sgattrs=sgattrs)
            else:
                _ALL_ENGINES[engine][n.type]["_all"](n, fuse=fuse, is_aiff=is_aiff,
                                                     plugin_name=plugin_name, sgattrs=sgattrs)
                # assert [list(o.shape) for o in n.outputs] == [list(o.betensor.shape) for o in n.outputs], str(n.name) + '   ' +  str(n.type)
            for o in n.outputs:
                tensor_map[o.name] = o
                in_tensors[o.name] = o
        for o in node.outputs:
            if o.name not in tensor_map:
                raise Exception("An output tensor %s is not in tensor map" % (o.name))
            o.betensor = tensor_map[o.name].betensor

    return f


def get_engine(name: str):
    if name in _ALL_ENGINES:
        return _ALL_ENGINES[name]


def register_engine(name: str):
    if name not in _ALL_ENGINES:
        from AIPUBuilder.core import OpType

        _ALL_ENGINES[name] = dict()
        _ALL_ENGINES[name][OpType.SubGraphOp] = dict()
        _ALL_ENGINES[name][OpType.SubGraphOp]["_all"] = _subgraph(name)


def register_executor(engine_name: str, optypes, target: str = "", version=1.):
    register_engine(engine_name)
    if not isinstance(optypes, (tuple, list)):
        optypes = [optypes]
    if len(target) == 0:
        target = "_all"

    def dec(func):
        for t in optypes:
            if t not in _ALL_ENGINES[engine_name]:
                _ALL_ENGINES[engine_name][t] = dict()
            _ALL_ENGINES[engine_name][t][target] = func
        return func

    return dec


def register_gt(optypes, target="", version=1.):
    return register_executor("_GT", optypes, target, version)


def register_gen_input(optypes):
    def dec(func):
        if optypes not in _GEN_INPUTS:
            _GEN_INPUTS[optypes] = func  # func is the gen_input_data() of op
        return func
    return dec


def register_gen_weight(optypes):
    def dec(func):
        if optypes not in _GEN_WEIGHTS:
            _GEN_WEIGHTS[optypes] = func
        return func
    return dec


def register_op_metric(optypes):
    def dec(func):
        if optypes not in _OP_METRICS:
            _OP_METRICS[optypes] = func
        return func
    return dec
