# SPDX-License-Identifier: Apache-2.0
# Copyright © 2025-2026 Arm Technology (China) Co. Ltd.

import numpy as np
import re
import copy
from ....common.defs import Tensor, FLOAT_EQUAL
from ....common.utils import extend_lists
from ....ops.op import Op, ActivationOnlyOp, BaseActivationOp, OpHasWeights, \
    OpHasPaddingStrides, OpNeedBroadcast, ExecutorchOp
from ....graph.node_wrap import NodeWrap
from ....graph.graph_algo import get_valid_node_name, clear_redundant_nodes
from ....graph.pattern_match import matched_patterns, single_node_matcher, two_nodes_matcher
from ...onnx.passes.common_passes import insert_constant, remove_node_safely, insert_transpose, insert_reshape, \
    insert_gather, insert_reshape_after, insert_tile, insert_cast_after, insert_transpose_after, insert_dequant_quant
from ....logger import INFO, DEBUG, WARN, ERROR, FATAL


def convert_aten_add_sub(graph):
    matches = single_node_matcher(graph, ['Atenadd', 'Atensub'])
    for m in matches:
        node = m['target']
        node_obj = NodeWrap(graph, node)['object']
        if node_obj is not None:
            in_edges = graph.sorted_in_edges(node, data=True)
            x_edge = next((e for e in in_edges if e[2].get('dst_in_port') == 0), None)
            y_edge = next((e for e in in_edges if e[2].get('dst_in_port') == 1), None)
            alpha_edge = next((e for e in in_edges if e[2].get('dst_in_port') == 2), None)

            if x_edge is None or y_edge is None:
                ERROR('[Parser]: Meets invalid Atenadd/Atensub Node(%s) inputs in convert_aten_add_sub!' % node)
                continue

            add_attr = node_obj.copied_attr()
            add_attr['opset_version'] = 7

            # aten::add/sub computes x +/- y * alpha. If alpha exists and is not 1, rewrite to Add/Sub(x, Mul(y, alpha)).
            alpha = node_obj.alpha
            need_alpha_mul = alpha_edge is not None and (alpha is None or not FLOAT_EQUAL(float(alpha), 1.0))
            if need_alpha_mul:
                mul_node = get_valid_node_name(graph, node + '_alpha_mul')
                graph.add_node(mul_node)

                graph.remove_edge(y_edge[0], node)
                graph.remove_edge(alpha_edge[0], node)

                y_in_attr = copy.deepcopy(y_edge[2])
                y_in_attr['dst_in_port'] = 0
                graph.add_edge(y_edge[0], mul_node, **y_in_attr)

                alpha_in_attr = copy.deepcopy(alpha_edge[2])
                alpha_in_attr['dst_in_port'] = 1
                graph.add_edge(alpha_edge[0], mul_node, **alpha_in_attr)

                mul_out_attr = copy.deepcopy(y_edge[2])
                mul_out_attr['src_out_port'] = 0
                mul_out_attr['dst_in_port'] = 1
                graph.add_edge(mul_node, node, **mul_out_attr)

                mul_attr = {'name': mul_node, 'opset_version': 7}
                NodeWrap(graph, mul_node).replace_obj('Mul', mul_attr)
            elif alpha_edge is not None:
                graph.remove_edge(alpha_edge[0], node)

            onnx_type = 'Add' if node_obj.type == 'Atenadd' else 'Sub'
            NodeWrap(graph, node).replace_obj(onnx_type, add_attr)
        else:
            ERROR('[Parser]: Meets invalid Atenadd/Atensub Node(%s) in convert_aten_add_sub!' % node)


def convert_aten_max(graph):
    matches = single_node_matcher(graph, 'Atenmax')
    for m in matches:
        node = m['target']
        node_obj = NodeWrap(graph, node)['object']
        if node_obj is not None:
            in_edges = graph.sorted_in_edges(node, data=True)
            out_edges = graph.sorted_out_edges(node, data=True)
            x_edge = next((e for e in in_edges if e[2].get('dst_in_port') == 0), None)

            if x_edge is None:
                ERROR('[Parser]: Meets invalid Atenmax Node(%s) inputs in convert_aten_max!' % node)
                continue

            axis = node_obj.dim
            keepdims = node_obj.keepdim
            graph.remove_edges_from([e for e in in_edges if e[2].get('dst_in_port') != 0])

            out_port1_edges = [e for e in out_edges if e[2].get('src_out_port') == 1]
            if out_port1_edges:
                argmax_node = get_valid_node_name(graph, node + '_argmax')
                graph.add_node(argmax_node)

                argmax_in_attr = copy.deepcopy(x_edge[2])
                argmax_in_attr['dst_in_port'] = 0
                graph.add_edge(x_edge[0], argmax_node, **argmax_in_attr)

                for _, dst, out_attr in out_port1_edges:
                    graph.remove_edge(node, dst)
                    new_out_attr = copy.deepcopy(out_attr)
                    new_out_attr['src_out_port'] = 0
                    graph.add_edge(argmax_node, dst, **new_out_attr)

                argmax_attr = {
                    'name': argmax_node,
                    'opset_version': 13,
                    'axis': axis,
                    'keepdims': int(keepdims),
                    'select_last_index': 0,
                }
                NodeWrap(graph, argmax_node).replace_obj('ArgMax', argmax_attr)

                if node in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(node)
                    graph._attr['output_names'].insert(index + 1, argmax_node)

            reduce_max_attr = {
                'name': node,
                'opset_version': 13,
                'axes': [axis],
                'keepdims': int(keepdims),
            }

            NodeWrap(graph, node).replace_obj('ReduceMax', reduce_max_attr)

            clear_redundant_nodes(graph)
        else:
            ERROR('[Parser]: Meets invalid Atenmax Node(%s) in convert_aten_max!' % node)


def convert_custom_sdpa(graph):
    matches = single_node_matcher(graph, 'Llamacustom_sdpa')
    # llama::custom_sdpa(Tensor query, Tensor key, Tensor value, SymInt start_pos, Tensor? attn_mask=None, float drpout_p=0.0, bool is_causal=False, float? scale=None)
    for m in matches:
        node = m['target']
        node_obj = NodeWrap(graph, node)['object']
        if node_obj is not None:
            in_edges = graph.sorted_in_edges(node, data=True)
            input_shapes = node_obj.get_input_shapes()
            q_dtype = node_obj.get_input_dtypes()[0]
            in_consts = node_obj.sorted_in_consts()
            is_causal = False
            scale = None
            for _, port, value in in_consts:
                if port == 6:
                    is_causal = bool(value)
                if port == 7:
                    scale = float(value)

            if is_causal:
                assert input_shapes[4] is None, 'is_causal and attn_mask cannot be set together.'

            graph.remove_edges_from(in_edges[3:])

            q_seq_len = input_shapes[0][1]
            kv_seq_len = input_shapes[1][1]
            if input_shapes[4] is None:
                # No Mask input
                trilu_node = get_valid_node_name(graph, node + '_trilu')
                graph.add_node(trilu_node)

                insert_constant(graph, node + '_init_mask',
                                np.ones([q_seq_len, kv_seq_len], dtype=q_dtype),
                                trilu_node, in_port=0)

                k, _, in_attr = in_edges[3]
                k_in_attr = copy.deepcopy(in_attr)
                k_in_attr['dst_in_port'] = 1
                graph.add_edge(k, trilu_node, **k_in_attr)
                trilu_attr = {
                    'name': trilu_node,
                    'opset_version': 14,
                    'upper': 0
                }
                NodeWrap(graph, trilu_node).replace_obj('Trilu', trilu_attr)

                sub_node = get_valid_node_name(graph, node + '_sub')
                graph.add_node(sub_node)

                insert_constant(graph, node + '_ones',
                                np.ones([], dtype=q_dtype),
                                sub_node, in_port=0)

                trilu_out_attr = copy.deepcopy(in_attr)
                trilu_out_attr['src_out_port'] = 0
                trilu_out_attr['dst_in_port'] = 1
                graph.add_edge(trilu_node, sub_node, **trilu_out_attr)

                sub_attr = {
                    'name': sub_node,
                    'opset_version': 6,
                }
                NodeWrap(graph, sub_node).replace_obj('Sub', sub_attr)

                mul_node = get_valid_node_name(graph, node + '_mul')
                graph.add_node(mul_node)

                insert_constant(graph, node + '_inf',
                                np.array(float('-inf'), dtype=q_dtype),
                                mul_node, in_port=1)

                sub_out_attr = copy.deepcopy(in_attr)
                sub_out_attr['src_out_port'] = 0
                sub_out_attr['dst_in_port'] = 0
                graph.add_edge(sub_node, mul_node, **sub_out_attr)

                mul_attr = {
                    'name': mul_node,
                    'opset_version': 6,
                }
                NodeWrap(graph, mul_node).replace_obj('Mul', mul_attr)
                mask_in_attr = copy.deepcopy(in_attr)
                mask_in_attr['dst_in_port'] = 3
                graph.add_edge(mul_node, node, **mask_in_attr)

                is_causal = False
            else:
                # Have Mask input
                mask_src = None
                mask_edge_attr = None
                for src, _, data in in_edges:
                    if data.get('dst_in_port') == 4:
                        mask_src = src
                        mask_edge_attr = copy.deepcopy(data)
                        break

                if mask_src is not None:
                    mask_edge_attr['dst_in_port'] = 3
                    graph.add_edge(mask_src, node, **mask_edge_attr)
                else:
                    WARN('[Parser]: attn_mask edge (port 4) not found for node(%s) in convert_custom_sdpa!' % node)

                is_causal = False

            if scale is None:
                q_head_size = input_shapes[0][-1]
                scale = (1 / np.sqrt(q_head_size)).astype(np.float32)
            sdpa_attr = node_obj.copied_attr()
            q_num_heads = input_shapes[0][2]
            k_num_heads = input_shapes[1][2]
            enable_gqa = (q_num_heads != k_num_heads
                          and q_num_heads % k_num_heads == 0
                          and k_num_heads == input_shapes[2][2])
            sdpa_attr['is_causal'] = is_causal
            sdpa_attr['scale'] = scale
            sdpa_attr['enable_gqa'] = enable_gqa
            NodeWrap(graph, node).replace_obj('ScaledDotProductAttention', sdpa_attr)
        else:
            ERROR('[Parser]: Meets invalid Llamacustom_sdpa Node(%s) in convert_custom_sdpa!' % node)


def convert_fc(graph):
    matches = single_node_matcher(graph, 'XNNFullyConnected')
    for m in matches:
        fc = m['target']
        fc_obj = NodeWrap(graph, fc)['object']
        fc_in_edges = graph.sorted_in_edges(fc, keys=True, data=True)
        if fc_obj is not None \
                and len(fc_in_edges) >= 1 \
                and len(fc_obj.get_input_shapes()) >= 1 \
                and len(fc_obj.get_input_shapes()[0]) >= 2:
            if graph._attr.get('quantize', False) \
                    and fc_obj.quantize:
                quantize = True
            else:
                quantize = False
            last_name = fc
            input_shapes = fc_obj.get_input_shapes()
            src, _, k, in_attr = fc_in_edges[0]
            fc_attr = fc_obj.copied_attr()
            need_add = False
            if fc_obj.weights is not None:
                fc_w = np.transpose(fc_obj.weights) if fc_obj.flag == 1 else fc_obj.weights
                if len(input_shapes[0]) > 2:
                    dim = [-1, fc_w.shape[-1]]
                    insert_reshape(graph, src, fc, in_attr, dim, key=k, quantize=quantize)
                biases = np.zeros((fc_w.shape[0],),
                                  dtype=np.int32 if np.issubdtype(
                                      fc_w.dtype, np.integer) else np.float32
                                  )
                if fc_obj.biases is not None:
                    bias_shape = list(fc_obj.biases.shape)
                    if bias_shape:
                        if bias_shape[-1] == int(np.prod(bias_shape).item()):
                            biases = np.reshape(fc_obj.biases, [bias_shape[-1]])
                        else:
                            # use Add Op
                            need_add = True
                    else:
                        biases.fill(fc_obj.biases.item())

                fc_attr.update({'weights': fc_w})
                fc_attr.update({'biases': biases})
                NodeWrap(graph, fc).replace_obj('FullyConnected', fc_attr)

                if need_add:
                    add = get_valid_node_name(graph, fc + '_bias_add')
                    out_edges = graph.sorted_out_edges(fc, data=True)
                    for _, dst, out_attr in out_edges:
                        graph.remove_edge(fc, dst)
                        graph.add_edge(add, dst, **out_attr)
                    graph.add_edge(fc, add)
                    insert_constant(graph, add + '_bias', fc_obj.biases,
                                    add, in_port=1, data_format='NHWC')
                    add_attr = {'name': add, 'opset_version': 7}
                    NodeWrap(graph, add).replace_obj('Add', add_attr)
                    last_name = add
                graph.remove_edges_from(fc_in_edges[1:])
            elif len(input_shapes) >= 2 and len(input_shapes[1]) == 2:
                if len(input_shapes[0]) > 2:
                    dim = [-1, input_shapes[1][-1]]
                    insert_reshape(graph, src, fc, in_attr, dim, key=k, quantize=quantize)
                    fc_in_edges = graph.sorted_in_edges(
                        fc, keys=True, data=True)
                second_input, _, k, in_attr2 = fc_in_edges[1]
                fc_attr.update({'opset_version': 9})
                NodeWrap(graph, fc).replace_obj('MatMul', fc_attr)
                if fc_obj.flag == 0:
                    insert_transpose(graph, second_input, fc,
                                     in_attr2, [1, 0], key=k, quantize=quantize)
                if fc_obj.biases is None \
                        and len(fc_in_edges) == 3 \
                        and fc_obj.get_input_tensors()[2] is not None:
                    fc_obj.biases = fc_obj.get_input_tensors()[2]
                if fc_obj.biases is not None:
                    add = get_valid_node_name(graph, fc + '_bias_add')
                    out_edges = graph.sorted_out_edges(fc, data=True)
                    for _, dst, out_attr in out_edges:
                        graph.remove_edge(fc, dst)
                        graph.add_edge(add, dst, **out_attr)
                    graph.add_edge(fc, add)
                    insert_constant(graph, add + '_bias', fc_obj.biases,
                                    add, in_port=1, data_format='NHWC')
                    add_attr = {'name': add, 'opset_version': 7}
                    NodeWrap(graph, add).replace_obj('Add', add_attr)
                    last_name = add
                graph.remove_edges_from(fc_in_edges[2:])
            else:
                last_name = None
                ERROR(
                    '[Parser]: Meets invalid pattern of XNNFullyConnected Node(%s) in convert_fc!' % fc)
                continue
            if fc in graph._attr['output_names'] \
                    and last_name \
                    and fc != last_name:
                index = graph._attr['output_names'].index(fc)
                graph._attr['output_names'][index] = last_name
        else:
            ERROR('[Parser]: Meets invalid XNNFullyConnected Node(%s) in convert_fc!' % fc)


def convert_local_scalar_dense(graph):
    matches = single_node_matcher(graph, 'Aten_local_scalar_dense')
    for m in matches:
        node = m['target']
        node_obj = NodeWrap(graph, node)['object']
        if node_obj is not None:
            input_shape = node_obj.get_input_shapes()[0]
            if len(input_shape) == 0:
                if node in graph._attr['output_names']:
                    index = graph._attr['output_names'].index(node)
                    pred_node_name = graph.sorted_in_edges(node)[0][0]
                    if pred_node_name not in graph._attr['output_names']:
                        graph._attr['output_names'][index] = pred_node_name
                    else:
                        graph._attr['output_names'].remove(node)
                remove_node_safely(graph, node)
            else:
                if len(input_shape) > 1:
                    # flatten the input to 1D
                    rs_shape = [int(np.prod(input_shape).item())]
                    src, _, in_attr = graph.sorted_in_edges(node, data=True)[0]
                    insert_reshape(graph, src, node, in_attr, rs_shape)

                insert_constant(graph, node + '_indices', np.array(0, np.int32), node, in_port=1)

                gather_attr = node_obj.copied_attr()
                gather_attr['opset_version'] = 11
                gather_attr['axis'] = 0
                NodeWrap(graph, node).replace_obj('Gather', gather_attr)
        else:
            ERROR('[Parser]: Meets invalid Aten_local_scalar_dense Node(%s) in convert_local_scalar_dense!' % node)


def convert_rsqrt(graph):
    matches = single_node_matcher(graph, 'XNNReciprocalSquareRoot')
    for m in matches:
        rsqrt = m['target']
        rsqrt_obj = NodeWrap(graph, rsqrt)['object']
        rsqrt_in_edges = graph.sorted_in_edges(rsqrt, data=True)
        if rsqrt_obj is not None:
            sqrt = get_valid_node_name(graph, rsqrt + '_sqrt')
            graph.add_node(sqrt)
            for src, _, in_attr in rsqrt_in_edges:
                graph.remove_edge(src, rsqrt)
                graph.add_edge(src, sqrt, **in_attr)
            graph.add_edge(sqrt, rsqrt)
            sqrt_attr = {'name': sqrt, 'opset_version': 6}
            NodeWrap(graph, sqrt).replace_obj('Sqrt', sqrt_attr)

            reci_attr = rsqrt_obj.copied_attr()
            reci_attr['opset_version'] = 6
            NodeWrap(graph, rsqrt).replace_obj('Reciprocal', reci_attr)
        else:
            ERROR('[Parser]: Meets invalid XNNReciprocalSquareRoot Node(%s) in convert_rsqrt!' % rsqrt)


def convert_to_onnx(graph):
    '''Convert the model to the onnx version.'''
    executorch_ops = ExecutorchOp.get_concrete_subclass_names()
    matches = extend_lists([single_node_matcher(graph, op_type)
                            for op_type in executorch_ops])
    for m in matches:
        node_name = m['target']
        node_obj = NodeWrap(graph, node_name)['object']
        if node_obj is not None:
            pure_type = re.sub(r'^(XNN|Aten|Llama|Executorch_prim|Dim_order_ops)', '', node_obj.type)
            if getattr(node_obj, 'correspond_onnx_op', None) is not None:
                if isinstance(node_obj, OpHasPaddingStrides):
                    xnn_padding = node_obj.padding
                    node_obj.pads = OpHasPaddingStrides.xnn_to_onnx(xnn_padding)
                new_node_attr = node_obj.copied_attr()
                if isinstance(node_obj, OpHasWeights):
                    if node_obj.weights is None:
                        ERROR('[Parser]: Node(%s) dosenot contain weights in convert_to_onnx!' %
                              node_name)
                        continue
                    new_weights = np.transpose(
                        node_obj.weights, axes=type(node_obj).perm_xnn_to_onnx())
                    new_node_attr.update({'weights': new_weights})
                if pure_type in ('ARG_MIN', 'ARG_MAX') and not node_obj.quantize:
                    post_cast = insert_cast_after(graph, node_name, 'int64', node_obj.to)
                    if node_name in graph._attr['output_names']:
                        index = graph._attr['output_names'].index(node_name)
                        graph._attr['output_names'][index] = post_cast
                elif pure_type == 'slice_copy':
                    in_edges = graph.sorted_in_edges(
                        node_name, keys=True, data=True)
                    input_tensors = node_obj.get_input_tensors()
                    axis = int(input_tensors[1])
                    start = int(input_tensors[2])
                    end = int(input_tensors[3])
                    step = int(input_tensors[4])
                    graph.remove_edges_from(in_edges[1:])

                    insert_constant(
                        graph, node_name + '_starts', np.array([start], np.int64), node_name, in_port=1)
                    insert_constant(
                        graph, node_name + '_ends', np.array([end], np.int64), node_name, in_port=2)
                    insert_constant(
                        graph, node_name + '_axes', np.array([axis], np.int64), node_name, in_port=3)
                    insert_constant(
                        graph, node_name + '_steps', np.array([step], np.int64), node_name, in_port=4)
                elif pure_type == 'unsqueeze_copy':
                    in_edges = graph.sorted_in_edges(
                        node_name, keys=True, data=True)
                    input_tensors = node_obj.get_input_tensors()
                    new_node_attr['axes'] = [int(input_tensors[1])]
                    graph.remove_edges_from(in_edges[1:])
                elif pure_type == 'RESHAPE':
                    in_edges = graph.sorted_in_edges(
                        node_name, keys=True, data=True)
                    input_tensors = node_obj.get_input_tensors()
                    if len(in_edges) == 2 and len(input_tensors) == 2 and np.ndim(input_tensors[1]) != 1:
                        shape_inp, _, k, in_attr = in_edges[1]
                        dim = [input_tensors[1].size]
                        insert_reshape(graph, shape_inp,
                                       node_name, in_attr, dim, key=k)
                elif pure_type == 'StaticResizeBilinear2D':
                    input_shape = node_obj.get_input_shapes()[0]
                    full_size = [input_shape[0], node_obj.new_height, node_obj.new_width, input_shape[-1]]
                    assert node_obj.correspond_onnx_op['version'] >= 11, \
                        '[Parser]: Only support Resize above 11 when converting from XNN to Onnx!'
                    insert_constant(graph, node_name + '_roi', np.array([],
                                                                        np.int64), node_name, in_port=1, data_format='NHWC')
                    insert_constant(
                        graph, node_name + '_scales', np.array([], np.float32), node_name, in_port=2, data_format='NHWC')
                    insert_constant(
                        graph, node_name + '_size', np.array(full_size, np.int32), node_name, in_port=3, data_format='NHWC')
                    new_node_attr.update({'mode': 'linear'})
                elif pure_type == 'SLICE':
                    starts = node_obj.sorted_in_consts()[0][2]
                    ends_name = node_obj.sorted_in_consts()[1][0]
                    NodeWrap(graph, ends_name)[
                        'object'].value = starts + node_obj.size
                elif pure_type == 'Square':
                    in_edges = graph.sorted_in_edges(node_name, data=True)
                    in0_edge = next((e for e in in_edges if e[2].get('dst_in_port') == 0), None)
                    in1_edge = next((e for e in in_edges if e[2].get('dst_in_port') == 1), None)
                    if in0_edge is None:
                        ERROR('[Parser]: Meets invalid XNNSquare Node(%s) missing input port(0)!' % node_name)
                    elif in1_edge is None:
                        dup_attr = copy.deepcopy(in0_edge[2])
                        dup_attr['dst_in_port'] = 1
                        dup_attr['src_out_port'] = dup_attr.get('src_out_port', 0)
                        graph.add_edge(in0_edge[0], node_name, **dup_attr)
                elif pure_type == 'BatchMatrixMultiply':
                    flags = int(getattr(node_obj, 'flag', 0) or 0)
                    if flags:
                        in_edges = graph.sorted_in_edges(
                            node_name, keys=True, data=True)
                        input_shapes = node_obj.get_input_shapes()
                        quantize = bool(graph._attr.get('quantize', False)
                                        and getattr(node_obj, 'quantize', False))

                        def _insert_transpose_for_port(dst_in_port):
                            edge = next((e for e in in_edges if e[3].get('dst_in_port') == dst_in_port), None)
                            if edge is None:
                                ERROR('[Parser]: Meets invalid XNNBatchMatrixMultiply Node(%s) missing input port(%d)!' %
                                      (node_name, dst_in_port))
                                return
                            src, _, key, in_attr = edge
                            input_rank = len(input_shapes[dst_in_port]) if len(input_shapes) > dst_in_port else 0
                            if input_rank < 2:
                                ERROR('[Parser]: Meets invalid XNNBatchMatrixMultiply Node(%s) input rank(%d) on port(%d)!' %
                                      (node_name, input_rank, dst_in_port))
                                return
                            perm = list(range(input_rank))
                            perm[-1], perm[-2] = perm[-2], perm[-1]
                            insert_transpose(graph, src, node_name, in_attr,
                                             perm, key=key, quantize=quantize)

                        # XNN flags: bit 1 -> transpose A(input 0), bit 0 -> transpose B(input 1)
                        if (flags & 0x2) != 0:
                            _insert_transpose_for_port(0)
                        if (flags & 0x1) != 0:
                            _insert_transpose_for_port(1)
                    new_node_attr.pop('flag', None)
                elif pure_type == 'Clamp':
                    if 'OutputMin' in new_node_attr:
                        new_node_attr['min'] = float(new_node_attr.pop('OutputMin'))
                    if 'OutputMax' in new_node_attr:
                        new_node_attr['max'] = float(new_node_attr.pop('OutputMax'))
                    new_node_attr.pop('flag', None)
                elif pure_type == 'Pow':
                    in_edges = graph.sorted_in_edges(node_name, data=True)
                    in0_edge = next((e for e in in_edges if e[2].get('dst_in_port') == 0), None)
                    in1_edge = next((e for e in in_edges if e[2].get('dst_in_port') == 1), None)
                    if in0_edge is None:
                        ERROR('[Parser]: Meets invalid XNNPow Node(%s) missing input port(0)!' % node_name)
                    elif in1_edge is None:
                        insert_constant(graph, node_name + '_pow_exp', np.array(2.0, dtype=np.float32),
                                        node_name, in_port=1)
                elif pure_type == 'StaticReshape':
                    in_edges = graph.sorted_in_edges(node_name, data=True)
                    in1_edge = next((e for e in in_edges if e[2].get('dst_in_port') == 1), None)
                    if in1_edge is None:
                        shape = np.array(getattr(node_obj, 'NewShapeAsNumpy', []), dtype=np.int64)
                        if shape.size == 0:
                            ERROR('[Parser]: Meets invalid XNNStaticReshape Node(%s) missing target shape!' % node_name)
                        else:
                            insert_constant(graph, node_name + '_shape', shape, node_name, in_port=1)
                    new_node_attr.pop('NewShapeAsNumpy', None)
                    new_node_attr.pop('NumDims', None)
                    new_node_attr.pop('flag', None)
                elif pure_type == 'StaticSlice':
                    in_edges = graph.sorted_in_edges(node_name, data=True)
                    starts = np.array(node_obj.offset, dtype=np.int64)
                    sizes = np.array(node_obj.size, dtype=np.int64)
                    if starts.size == 0 or sizes.size == 0 or starts.size != sizes.size:
                        ERROR('[Parser]: Meets invalid XNNStaticSlice Node(%s) with invalid offsets/sizes!' % node_name)
                    else:
                        ends = starts + sizes
                        ends = np.where(sizes == 0, np.int64(np.iinfo(np.int64).max), ends)
                        input_shape = node_obj.get_input_shapes()[0]
                        input_rank = len(input_shape) if input_shape is not None else None
                        axes = np.arange(starts.size, dtype=np.int64)
                        steps = np.ones(starts.size, dtype=np.int64)

                        in1_edge = next((e for e in in_edges if e[2].get('dst_in_port') == 1), None)
                        in2_edge = next((e for e in in_edges if e[2].get('dst_in_port') == 2), None)
                        in3_edge = next((e for e in in_edges if e[2].get('dst_in_port') == 3), None)
                        in4_edge = next((e for e in in_edges if e[2].get('dst_in_port') == 4), None)

                        if input_rank is not None and starts.size > input_rank:
                            ERROR('[Parser]: Meets invalid XNNStaticSlice Node(%s), slice dims(%d) exceeds input rank(%d)!' %
                                  (node_name, starts.size, input_rank))
                        else:
                            if in1_edge is None:
                                insert_constant(graph, node_name + '_starts', starts, node_name, in_port=1)
                            if in2_edge is None:
                                insert_constant(graph, node_name + '_ends', ends, node_name, in_port=2)
                            if in3_edge is None:
                                insert_constant(graph, node_name + '_axes', axes, node_name, in_port=3)
                            if in4_edge is None:
                                insert_constant(graph, node_name + '_steps', steps, node_name, in_port=4)
                elif pure_type == 'StaticConstantPad':
                    in_edges = graph.sorted_in_edges(node_name, data=True)
                    in1_edge = next((e for e in in_edges if e[2].get('dst_in_port') == 1), None)
                    in2_edge = next((e for e in in_edges if e[2].get('dst_in_port') == 2), None)
                    if in1_edge is None:
                        pads = np.array(node_obj.pads).astype(np.int64)
                        insert_constant(graph, node_name + '_pads', pads, node_name, in_port=1)
                    if in2_edge is None:
                        pad_val = np.array(node_obj.pad_value, dtype=np.float32)
                        insert_constant(graph, node_name + '_value', pad_val, node_name, in_port=2)
                    new_node_attr.update({'mode': 'constant'})
                elif pure_type == '_native_batch_norm_legit_no_training':
                    in_edges = graph.sorted_in_edges(node_name, data=True)
                    input_tensors = node_obj.get_input_tensors()
                    if len(input_tensors) > 5 and input_tensors[5] is not None:
                        new_node_attr['momentum'] = float(np.array(input_tensors[5]).item())
                    if len(input_tensors) > 6 and input_tensors[6] is not None:
                        new_node_attr['epsilon'] = float(np.array(input_tensors[6]).item())
                    # Onnx BatchNormalization takes only X, scale, B, mean, var as inputs.
                    graph.remove_edges_from([e for e in in_edges if e[2].get('dst_in_port') not in (0, 1, 2, 3, 4)])

                new_node_attr.update(
                    {'opset_version': node_obj.correspond_onnx_op['version']})
                NodeWrap(graph, node_name).replace_obj(
                    node_obj.correspond_onnx_op['type'], new_node_attr)
            else:
                ERROR('[Parser]: Executorch op %s cannot be converted to Onnx' % pure_type)
        else:
            ERROR(
                '[Parser]: Meets invalid Executorch op for Node(%s) in convert_to_onnx!' % node_name)

    clear_redundant_nodes(graph)


def remove_useless_atenop(graph, op_type_list):
    '''Remove some redundant unwanted OPs.'''
    for op_type in op_type_list:
        removing_nodes = []
        matched = single_node_matcher(graph, f'Aten{op_type}')
        for m in matched:
            node_name = m['target']
            node_obj = NodeWrap(graph, node_name)['object']
            if node_obj is None:
                continue
            if op_type in ('_to_copy', 'alias_copy'):
                removing_nodes.append(node_name)
            else:
                # TODO
                removing_nodes.append(node_name)

        for rn in removing_nodes:
            if rn in graph._attr['output_names']:
                index = graph._attr['output_names'].index(rn)
                pred_node_name = graph.sorted_in_edges(rn)[0][0]
                if pred_node_name not in graph._attr['output_names']:
                    graph._attr['output_names'][index] = pred_node_name
                else:
                    graph._attr['output_names'].remove(rn)
            remove_node_safely(graph, rn)
