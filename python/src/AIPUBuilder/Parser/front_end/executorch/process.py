# SPDX-License-Identifier: Apache-2.0
# Copyright © 2025-2026 Arm Technology (China) Co. Ltd.

from .load import convert_executorch_to_graph
from ..onnx.passes.common_passes import apply_subgraph_plugin, record_output_tensors, remove_useless_op, fuse_const
from ...graph.graph_algo import infer, clear_redundant_nodes
from ..onnx.passes.front_passes import fuse_weights_const, convert_deconv
from .passes.front_passes import convert_to_onnx, convert_fc, remove_useless_atenop, convert_rsqrt, \
    convert_local_scalar_dense, convert_custom_sdpa, convert_aten_add_sub, convert_aten_max
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


def front_process_executorch(graph, params):
    record_output_tensors(graph)
    if graph is not None and len(graph) > 0:
        apply_subgraph_plugin(graph)
        infer(graph, partial=True)
        fuse_weights_const(graph)

        infer(graph)
        remove_useless_atenop(
            graph, ['_to_copy', 'alias_copy'])

        convert_aten_add_sub(graph)
        convert_aten_max(graph)
        convert_custom_sdpa(graph)
        convert_local_scalar_dense(graph)
        convert_fc(graph)
        convert_rsqrt(graph)
        convert_to_onnx(graph)
        infer(graph)
    else:
        WARN('[Parser]: Got empty graph in front_process_executorch!')


def process_executorch(graph, model_path, params):
    '''Do some preprocessing on the graph under the executorch framework.'''
    graph = convert_executorch_to_graph(graph, model_path, params)
    front_process_executorch(graph, params)

    return graph
