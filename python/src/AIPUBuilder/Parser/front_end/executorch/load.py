# SPDX-License-Identifier: Apache-2.0
# Copyright © 2025-2026 Arm Technology (China) Co. Ltd.
import copy
import numpy as np
from sympy import Symbol
from .buffer import *
from .executorch_flatbuffer.TensorShapeDynamism import TensorShapeDynamism
from ...common.defs import Framework, Tensor, get_opset_version
from ...common.utils import get_version, extend_lists
from ...utils._optional_import import require_optional_import
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL
from ...graph.node_wrap import NodeWrap
from ...graph.graph_algo import get_valid_node_name, clear_redundant_nodes


def convert_attr_to_onnx(attr_dict):
    new_attr = copy.deepcopy(attr_dict)
    for k, v in attr_dict.items():
        if k == 'AdjustmentHeight':
            adj = new_attr.get('adj', [0, 0])
            adj[0] = v
            new_attr.update({'adj': adj})
            new_attr.pop(k)
        elif k == 'AdjustmentWidth':
            adj = new_attr.get('adj', [0, 0])
            adj[1] = v
            new_attr.update({'adj': adj})
            new_attr.pop(k)
        elif k == 'Axis':
            new_attr.update({'axis': v})
            new_attr.pop(k)
        elif k == 'DilationDepth':
            new_attr.update(
                {'dilations': [v, *new_attr.get('dilations', [1, 1, 1])[1:3]]})
            new_attr.pop(k)
        elif k == 'DilationHeight':
            if 'DilationDepth' in attr_dict:
                new_attr.update(
                    {'dilations': [new_attr.get('dilations', [1, 1, 1])[0], v, new_attr.get('dilations', [1, 1, 1])[2]]})
            else:
                new_attr.update(
                    {'dilations': [v, new_attr.get('dilations', [1, 1])[1]]})
            new_attr.pop(k)
        elif k == 'DilationWidth':
            if 'DilationDepth' in attr_dict:
                new_attr.update(
                    {'dilations': [*new_attr.get('dilations', [1, 1, 1])[0:2], v]})
            else:
                new_attr.update(
                    {'dilations': [new_attr.get('dilations', [1, 1])[0], v]})
            new_attr.pop(k)
        elif k == 'GroupInputChannels':
            new_attr.update({'group_ic': v})
            new_attr.pop(k)
        elif k == 'GroupOutputChannels':
            new_attr.update({'group_oc': v})
            new_attr.pop(k)
        elif k == 'Groups':
            new_attr.update({'group': v})
            new_attr.pop(k)
        elif k in ('KernelHeight', 'PoolingHeight'):
            new_attr.update(
                {'kernel_shape': [v, new_attr.get('kernel_shape', [1, 1])[1]]})
            new_attr.pop(k)
        elif k in ('KernelWidth', 'PoolingWidth'):
            new_attr.update(
                {'kernel_shape': [new_attr.get('kernel_shape', [1, 1])[0], v]})
            new_attr.pop(k)
        elif k == 'NewHeight':
            new_attr.update({'new_height': v})
            new_attr.pop(k)
        elif k == 'NewWidth':
            new_attr.update({'new_width': v})
            new_attr.pop(k)
        elif k == 'PaddingTop':
            padding = new_attr.get('padding', [0, 0, 0, 0])
            padding[0] = v
            new_attr.update(
                {'padding': padding})
            new_attr.pop(k)
        elif k == 'PaddingBottom':
            padding = new_attr.get('padding', [0, 0, 0, 0])
            padding[1] = v
            new_attr.update(
                {'padding': padding})
            new_attr.pop(k)
        elif k == 'PaddingLeft':
            padding = new_attr.get('padding', [0, 0, 0, 0])
            padding[2] = v
            new_attr.update(
                {'padding': padding})
            new_attr.pop(k)
        elif k == 'PaddingRight':
            padding = new_attr.get('padding', [0, 0, 0, 0])
            padding[3] = v
            new_attr.update(
                {'padding': padding})
            new_attr.pop(k)
        elif k == 'PermAsNumpy':
            new_attr.update({'perm': v.tolist()})
            new_attr.pop(k)
        elif k in ('SubsamplingHeight', 'StrideHeight'):
            strides = new_attr.get('strides', [1, 1])
            strides[0] = v
            new_attr.update({'strides': strides})
            new_attr.pop(k)
        elif k in ('SubsamplingWidth', 'StrideWidth'):
            strides = new_attr.get('strides', [1, 1])
            strides[1] = v
            new_attr.update({'strides': strides})
            new_attr.pop(k)
    return new_attr


def get_last_domain(domain, fwd_ep):
    # domain: 0_1
    chain_id, inst_id = [int(i) for i in domain.split('_')]

    if inst_id > 0:
        last_inst_id = inst_id - 1
        last_chain_id = chain_id
    else:
        assert chain_id > 0, 'Must have multi-chains.'
        last_chain_id = chain_id - 1
        chain = fwd_ep.Chains(last_chain_id)
        last_inst_id = chain.InstructionsLength() - 1

    return f'{last_chain_id}_{last_inst_id}'


def convert_executorch_to_graph(graph, model_path, params):
    from ...plugin_loader import PARSER_OP_DICT
    try:
        onnx = require_optional_import('onnx', 'executorch parser path')
    except ImportError as e:
        ERROR('[Parser]: %s' % str(e))
        return graph
    graph._attr['framework'] = Framework.EXECUTORCH
    graph._attr['output_tensor_names'] = params.get('output_tensor_names', [])
    force_float = True \
        if ((params.get('force_float_ir', 'false').lower() == 'true')
            or (params.get('compat_quantized_model', 'true').lower() == 'false')) \
        else False

    try:
        model = read_executorch_model(model_path)
        if model:
            consumer_ver = get_version(onnx)
            if consumer_ver >= 1.04:
                opset_version = get_opset_version(consumer_ver)
                graph._attr['opset_version'] = opset_version
                net_inputs = model.fwd_ep.InputsAsNumpy().tolist()
                net_outputs = model.fwd_ep.OutputsAsNumpy().tolist()
                nodes = model.parse_chain()

                custom_input_infos = {}
                for inp_name in params.get('input_names', []):
                    if 'input_shapes' in params:
                        inp_shape = params['input_shapes'][inp_name]
                        if inp_shape:
                            if 'input_dtypes' in params:
                                inp_dtype = params['input_dtypes'][inp_name]
                                if inp_dtype.find('int') != -1:
                                    net_in_tensor = np.zeros(inp_shape, inp_dtype)
                                else:
                                    net_in_tensor = (np.random.ranf(
                                        size=inp_shape) * 100).astype(inp_dtype)
                            else:
                                net_in_tensor = (np.random.ranf(
                                    size=inp_shape) * 100).astype(np.float32)
                            custom_input_infos[inp_name] = Tensor(name=inp_name, value=net_in_tensor)
                        else:
                            custom_input_infos[inp_name] = None
                    else:
                        custom_input_infos[inp_name] = None

                input_tensors = OrderedDict()
                for i, single_input in enumerate(net_inputs):
                    inp_attr = {}
                    value_info = model.parse_value(single_input)
                    tensor_name = str(single_input)
                    if tensor_name in custom_input_infos:
                        del custom_input_infos[tensor_name]
                    if tensor_name in params['input_layouts'] and params['input_layouts'][tensor_name]:
                        inp_attr.update({'layout': params['input_layouts'][tensor_name]})
                    outputs = [value_info]

                    shape = value_info['shape']
                    if tensor_name in params['input_shapes'] \
                            and params['input_shapes'][tensor_name] is not None \
                            and shape != params['input_shapes'][tensor_name]:
                        new_shape = params['input_shapes'][tensor_name]
                        WARN('Original model expects input shape of input %s to be %s. '
                             'Now reset it to %s basing on config file!' % (tensor_name, str(shape), str(new_shape)))
                        shape = new_shape

                    net_in_dtype = value_info['dtype']
                    if net_in_dtype.find('int') != -1:
                        net_in_tensor = np.zeros(shape, net_in_dtype)
                    else:
                        net_in_tensor = (np.random.ranf(
                            size=shape) * 100).astype(net_in_dtype)

                    inp_tensor = Tensor(name=tensor_name, value=net_in_tensor)

                    global_symbol_dict = {}
                    if value_info['dynamic_type'] != TensorShapeDynamism.STATIC:
                        # dynamic shape, set symbol in tensor
                        shape_symbol = []
                        for axis in range(len(shape)):
                            sym = Symbol(f'd{i}{axis}', integer=True)
                            shape_symbol.append(sym)
                            if sym not in global_symbol_dict:
                                global_symbol_dict[sym] = shape[axis]
                        inp_tensor.shape_symbol = shape_symbol
                        inp_tensor.is_dynamic = True
                        graph._attr['ds_mode'] = True
                        inp_attr['ds_mode'] = True
                    else:   # TODO, ds_mode attr pass method during graph fwd
                        inp_attr['ds_mode'] = False

                    if global_symbol_dict:
                        graph._attr['global_symbols'].update(global_symbol_dict)

                    nodes.insert(
                        0, {'type': 'Input', 'name': tensor_name, 'op_attr': inp_attr,
                            'inputs': [], 'outputs': outputs})

                    input_tensors.update(
                        {tensor_name: inp_tensor})

                # handle update_cache
                need_insert_input_tensors = []
                for op_info in nodes:
                    op_type = op_info['type']
                    if op_type == 'Llamaupdate_cache':
                        for in_port, in_tensor in enumerate(op_info['inputs']):
                            if in_port == 1:
                                need_insert_input_tensors.append(in_tensor)

                for in_tensor in need_insert_input_tensors:
                    outputs = [in_tensor]
                    tensor_name = in_tensor['name']
                    nodes.insert(0,
                                 {'type': 'Input', 'name': tensor_name,
                                  'op_attr': {}, 'inputs': [], 'outputs': outputs}
                                 )

                    shape = in_tensor['shape']
                    net_in_dtype = in_tensor['dtype']

                    net_in_tensor = np.zeros(shape, net_in_dtype)
                    input_tensors.update(
                        {tensor_name: Tensor(name=tensor_name, value=net_in_tensor)})

                graph._attr['input_tensors'] = copy.deepcopy(input_tensors)

                graph_is_quantized = False
                # if force_float:
                #     graph_is_quantized = False
                # else:
                #     if all(not nodes[out_id].get('quantize', False)
                #            for op in nodes
                #            for out_id in op.get('outputs', []) if op['type'] not in ['Input', 'CAST']):
                #         graph_is_quantized = False
                #     else:
                #         graph_is_quantized = True

                out_tensor_operator_map = {k['name']: oi for oi, op_info in enumerate(nodes)
                                           for k in op_info['outputs']}
                out_tensor_op_name_map = OrderedDict()

                for op_id, op_info in enumerate(nodes):
                    node_name = op_info['name']
                    node_name = get_valid_node_name(graph, node_name)
                    graph.add_node(node_name)
                    detect_quantize = False
                    for o in op_info.get('outputs', []):
                        out_tensor_op_name_map[o['name']] = node_name
                        detect_quantize = detect_quantize or o['quantize']
                    op_type = op_info['type']
                    if op_info['type'] in ('DEQUANTIZE', 'QUANTIZE'):
                        detect_quantize = True
                    quantize = False if not graph_is_quantized else detect_quantize
                    attr_dict = copy.deepcopy(op_info['op_attr'])
                    attr_dict.update(
                        {'name': node_name, 'data_format': 'NHWC', 'quantize': quantize})
                    if op_type not in PARSER_OP_DICT:
                        attr_dict = convert_attr_to_onnx(attr_dict)
                    node = NodeWrap(graph, node_name)
                    node.replace_obj(op_type, attr_dict)

                    for in_port, in_tensor in enumerate(op_info['inputs']):
                        in_tensor_name = in_tensor['name']
                        if in_tensor['is_const']:
                            pre_node_name = get_valid_node_name(graph, in_tensor_name)
                            assert not graph.has_node(
                                pre_node_name), ('The const node(%s) already exist, cannot add node into graph in convert_executorch_to_graph.' % pre_node_name)
                            graph.add_node(pre_node_name)
                            src_out_port = 0
                            const_node = NodeWrap(graph, pre_node_name)
                            const_node.replace_obj('Constant',
                                                   {
                                                       'name': pre_node_name,
                                                       'value': in_tensor['data'],
                                                       'data_format': 'NHWC',
                                                       'quantize': in_tensor['quantize'],
                                                   })
                        else:
                            if in_tensor_name not in out_tensor_operator_map:
                                if in_tensor['data'] is None and in_tensor['dtype'] is None:
                                    in_tensor_name = get_valid_node_name(
                                        graph, node_name + '_in_tensor_' + str(in_port))
                                    graph.add_node(in_tensor_name)
                                    blank_node = NodeWrap(graph, in_tensor_name)
                                    blank_attr = {'name': in_tensor_name}
                                    blank_node.replace_obj('Blank', blank_attr)
                                    src_out_port = 0
                                    pre_node_name = in_tensor_name
                                else:
                                    # Special case, Insert Reshape to connect graph
                                    # suppose in_tensor_id - 1 is reshape input
                                    current_domain = op_info['domain']
                                    last_domain = get_last_domain(current_domain, model.fwd_ep)
                                    assert last_domain in model.fwd_inst_info, 'FWD EP should have inst io infos.'
                                    assert len(model.fwd_inst_info[last_domain]['outputs']
                                               ) == 1, 'Suppose last inst only have 1 output.'
                                    rs_in_tensor_name = model.fwd_inst_info[last_domain]['outputs'][0]
                                    assert rs_in_tensor_name in out_tensor_operator_map, 'Cannot find output tensor id in nodes.'

                                    # check if can insert reshape
                                    rs_pre_node = nodes[out_tensor_operator_map[rs_in_tensor_name]]
                                    rs_pre_node_src_out_port = [o['name']
                                                                for o in rs_pre_node['outputs']].index(rs_in_tensor_name)
                                    rs_pre_node_out_tensor_size = int(
                                        np.prod(rs_pre_node['outputs'][rs_pre_node_src_out_port]['shape']))
                                    current_tensor_size = int(np.prod(in_tensor['shape']))
                                    assert rs_pre_node_out_tensor_size == current_tensor_size, 'Graph disconnected and cannot insert reshape.'

                                    dim = in_tensor['shape']
                                    reshape_name = get_valid_node_name(
                                        graph, f'{rs_in_tensor_name}_{in_tensor_name}_reshape')
                                    assert not graph.has_node(
                                        reshape_name), (
                                        'The reshape node(%s) already exist, cannot add node into graph in convert_executorch_to_graph.' % pre_node_name)
                                    graph.add_node(reshape_name)
                                    reshape_attr = {'name': reshape_name, 'opset_version': 5, 'quantize': in_tensor['quantize'],
                                                    'from_parser': True}

                                    reshape_dim = np.array(dim, np.int32)
                                    const = get_valid_node_name(graph, reshape_name + '_shape')
                                    graph.add_node(const)
                                    const_attr = {
                                        'name': const,
                                        'value': reshape_dim,
                                        'data_format': 'NHWC',
                                        'opset_version': 9
                                    }
                                    NodeWrap(graph, const).replace_obj('Constant', const_attr)
                                    const_edge_attr = {
                                        'src_out_port': 0,
                                        'dst_in_port': 1,
                                        'tensor': Tensor(value=reshape_dim, is_const=True)
                                    }
                                    graph.add_edge(const, reshape_name, **const_edge_attr)
                                    reshape_node = NodeWrap(graph, reshape_name)
                                    reshape_node.replace_obj('Reshape', reshape_attr)

                                    reshape_in_attr = {
                                        'src_out_port': rs_pre_node_src_out_port,
                                        'dst_in_port': 0,
                                        'tensor': Tensor(value=rs_pre_node['outputs'][rs_pre_node_src_out_port]['data'],
                                                         shape=rs_pre_node['outputs'][rs_pre_node_src_out_port]['shape'],
                                                         dtype=rs_pre_node['outputs'][rs_pre_node_src_out_port]['dtype'],
                                                         is_const=rs_pre_node['outputs'][rs_pre_node_src_out_port]['is_const'])
                                    }
                                    graph.add_edge(rs_pre_node['name'], reshape_name, **reshape_in_attr)
                                    src_out_port = 0
                                    pre_node_name = reshape_name
                            else:
                                pre_op_id = out_tensor_operator_map[in_tensor_name]
                                pre_op_info = nodes[pre_op_id]
                                pre_node_name = pre_op_info['name']
                                src_out_port = [o['name'] for o in pre_op_info['outputs']].index(in_tensor_name)
                        edge_tensor = Tensor(name=in_tensor['name'],
                                             dtype=in_tensor['dtype'],
                                             is_const=in_tensor['is_const'])
                        edge_tensor.value = in_tensor['data']
                        if 'shape' in in_tensor:
                            edge_tensor.shape = in_tensor['shape']
                        if 'quant_info' in in_tensor:
                            if 'Max' in in_tensor['quant_info'] \
                                    and 'Min' in in_tensor['quant_info']:
                                edge_tensor.min_max = (
                                    in_tensor['quant_info']['Min'], in_tensor['quant_info']['Max'])
                            if 'Scale' in in_tensor['quant_info'] \
                                    and 'ZeroPoint' in in_tensor['quant_info']:
                                edge_tensor.scale_zp = (in_tensor['quant_info']['Scale'],
                                                        in_tensor['quant_info']['ZeroPoint'])

                        edge_attr = {'src_out_port': src_out_port,
                                     'dst_in_port': in_port, 'tensor': edge_tensor}
                        if pre_node_name in custom_input_infos:
                            custom_input_infos[pre_node_name] = copy.deepcopy(edge_tensor)
                        assert graph.has_node(
                            pre_node_name), f'The node({pre_node_name}) does not exist, cannot add edge into graph in convert_executorch_to_graph.'
                        graph.add_edge(
                            pre_node_name, node_name, **edge_attr)

                # add output node into graph
                output_names = []
                for output_index, output_tensor_id in enumerate(net_outputs):
                    out_tensor_name = str(output_tensor_id)
                    if output_tensor_id < 0 \
                            or out_tensor_name not in out_tensor_operator_map:
                        DEBUG('[Parser]: Meets invalid output tensor id(%d) in convert_executorch_to_graph!' %
                              output_tensor_id)
                        continue
                    value_info = model.parse_value(output_tensor_id)
                    out_op_id = out_tensor_operator_map[out_tensor_name]
                    out_op_info = nodes[out_op_id]
                    out_node_name = out_op_info['name']
                    noop_node_name = out_tensor_name + '_noop_' + str(
                        output_index)
                    assert graph.has_node(out_node_name) and not graph.has_node(
                        noop_node_name), 'The output node does not exist, cannot add output node into graph in convert_tflite_to_graph.'
                    graph.add_node(noop_node_name)
                    noop_node = NodeWrap(graph, noop_node_name)
                    noop_node.replace_obj('Out', {'name': noop_node_name})
                    if out_tensor_name in params.get('output_tensor_map', {}):
                        params['output_tensor_map'][out_tensor_name] = [noop_node_name]

                    src_out_port = [o['name'] for o in out_op_info['outputs']].index(
                        out_tensor_name)
                    out_edge_tensor = Tensor(name=out_tensor_name, shape=value_info['shape'])
                    if 'quant_info' in value_info:
                        if 'Max' in value_info['quant_info'] \
                                and 'Min' in value_info['quant_info']:
                            out_edge_tensor.min_max = (
                                value_info['quant_info']['Min'], value_info['quant_info']['Max'])
                        if 'Scale' in value_info['quant_info'] \
                                and 'ZeroPoint' in value_info['quant_info']:
                            out_edge_tensor.scale_zp = (
                                value_info['quant_info']['Scale'], value_info['quant_info']['ZeroPoint'])
                    if 'dtype' in value_info:
                        out_edge_tensor.dtype = value_info['dtype']
                    if value_info.get('data', None) is not None:
                        out_edge_tensor.value = value_info['data']
                        out_edge_tensor.shape = value_info['data'].shape
                    out_edge_attr = {'src_out_port': src_out_port,
                                     'dst_in_port': 0, 'tensor': out_edge_tensor}
                    graph.add_edge(
                        out_node_name, noop_node_name, **out_edge_attr)

                    if out_node_name not in output_names:
                        output_names.append(out_node_name)

                # Check outputs set in config
                if params.get('output_names', []):
                    names = []
                    for n in params['output_names']:
                        if graph.has_node(n):
                            if n not in names:
                                names.append(n)
                        elif n in out_tensor_op_name_map:
                            if out_tensor_op_name_map[n] not in names:
                                names.append(out_tensor_op_name_map[n])
                        else:
                            WARN(
                                '[Parser]: Output name (%s) is neither a node name nor a tensor name! Please check config file!' % n)
                    if not names:
                        WARN(
                            '[Parser]: Cannot get valid output names from config file! Will use parsed node names instead!')
                        graph._attr['output_names'] = copy.deepcopy(
                            output_names)
                    else:
                        graph._attr['output_names'] = copy.deepcopy(names)
                        # convert node name to tensor name
                        if any([t not in out_tensor_op_name_map for t in graph._attr['output_tensor_names']]) \
                                and all([graph.has_node(n) for n in graph._attr['output_tensor_names']]):
                            out_tensors = []
                            for n in graph._attr['output_tensor_names']:
                                out_edges = graph.sorted_out_edges(
                                    n, data=True)
                                for u, v, d in out_edges:
                                    t = d.get('tensor', None)
                                    if t is not None:
                                        out_tensors.append(t.name)
                                    else:
                                        pass
                else:
                    graph._attr['output_names'] = copy.deepcopy(
                        output_names)

                for name in graph._attr['output_names']:
                    if graph.has_node(name):
                        node_successors = graph.children(name)
                        if not node_successors \
                                or all([graph.nodes[succ]['op'] != 'Out' for succ in node_successors]):
                            out_edges = graph.sorted_out_edges(
                                name, data=True)
                            try:
                                out_ports = graph.nodes[name]['object'].get_out_ports()
                            except:
                                out_ports = []
                            for p in out_ports:
                                tensor = None
                                for _, _, out_attr in out_edges:
                                    if out_attr['src_out_port'] == p:
                                        tensor = out_attr['tensor']
                                        break
                                out_name = get_valid_node_name(
                                    graph, name + '_out_' + str(p))
                                graph.add_edge(
                                    name, out_name, **{'src_out_port': p, 'dst_in_port': 0, 'tensor': tensor})
                                NodeWrap(graph, out_name).replace_obj(
                                    'Out', {'name': out_name})
                                if tensor is not None and tensor.name in params.get('output_tensor_map', {}):
                                    params['output_tensor_map'][tensor.name] = [out_name]

                if not graph._attr['output_names']:
                    ERROR('[Parser]: Got no output names for graph, cannot proceed!')
                else:
                    if any(in_name not in graph._attr['input_tensors'] for in_name in params['input_names']):
                        for in_name in params['input_names']:
                            if graph.has_node(in_name) \
                                    and in_name not in graph._attr['input_tensors']:
                                out_edges = graph.sorted_out_edges(in_name, data=True)
                                if len(out_edges) > 0:
                                    in_edges = graph.sorted_in_edges(in_name)
                                    graph.remove_edges_from(in_edges)
                                    obj = NodeWrap(graph, in_name)['object']
                                    if obj is not None:
                                        NodeWrap(graph, in_name).replace_obj('Input', obj.copied_attr())
                                    else:
                                        NodeWrap(graph, in_name).replace_obj('Input', {'name': in_name})
                                    if out_edges[0][2]['tensor'] is not None and out_edges[0][2][
                                            'tensor'].value is not None:
                                        graph._attr['input_tensors'].update(
                                            {in_name: out_edges[0][2]['tensor']})
                                    else:
                                        try:
                                            inp_tensor = custom_input_infos[in_name]
                                        except:
                                            ERROR(
                                                '[Parser]: Shape of Input(%s) is unknown! Please provide input_shape in cfg file!' % in_name)
                                            continue
                                        for out_edge in out_edges:
                                            out_edge[2]['tensor'] = inp_tensor
                                        graph._attr['input_tensors'].update(
                                            {in_name: inp_tensor})
                        graph._attr['input_tensors'] = OrderedDict(
                            {k: v for k, v in graph._attr['input_tensors'].items() if graph.has_node(k)})

                clear_redundant_nodes(graph)
            else:
                ERROR('Onnx version is too low, needs updating!')
    except Exception as e:
        ERROR('[Parser]: Read Executorch %s meets error in convert_executorch_to_graph! %s' % (
            model_path, str(e)))
    return graph
