# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.logger import OPT_INFO
from AIPUBuilder.Optimizer.framework import PyNode, PyTensor, OpType, Dtype
from AIPUBuilder.Optimizer.utils import QuantMode, Target, OP_ONLY_CHANGE_SHAPE
from .deduce_key_axis import deduce_key_axis, ACT_PER_CHANNEL_DEDUCE_KEY_AXIS
import torch

__all__ = ["enable_activation_perchannel", ]


def before_activation_perchannel(graph, config):
    for n in graph.nodes:
        act_mode = n.attrs['q_mode_activation']
        pmode_status = all([QuantMode.is_per_tensor(pn.attrs['q_mode_activation']) for pn in n.parents])
        min_compatible_zhouyi_target = n.attrs["min_compatible_zhouyi_target"].upper()
        if QuantMode.is_per_channel(act_mode) and (n.type not in ACT_PER_CHANNEL_DEDUCE_KEY_AXIS or (n.type in OP_ONLY_CHANGE_SHAPE and pmode_status) or (Target.optimized_target_level(min_compatible_zhouyi_target) < 2)):
            n.attrs['q_mode_activation'] = QuantMode.to_per_tensor(act_mode)
        else:
            # take FC as an example: A (197, 768) x W (768, 3072) + B (1, 3072)
            # if A.zp.shape = (197, 1) then after B absorb A.zp will change to b.shape = (197, 3072) which is too wasteful
            # if A.zp.shape = (1, 768) then W has to absorb A.scale first and then push sum(A.zp * w) to bias, although we will got new bias.shape = (1, 3072), but W likely become too trickly to be quantized, and for conv operator we do not support per-channel zp padding on AIFF
            n.attrs['q_mode_activation'] = QuantMode.to_symmetric(act_mode)
        is_regularize_scale = n.attrs.get('regularize_activation_perchannel_scales', 'disable').lower()
        is_per_channel = QuantMode.is_per_channel(n.attrs['q_mode_activation'])
        for ot in n.outputs:
            ot.attrs['regularize_activation_perchannel_scales'] = is_regularize_scale == 'enable' and is_per_channel


def after_activation_perchannel(graph, config):
    for n in graph.nodes:
        act_mode = n.attrs['q_mode_activation']
        is_perchannel = QuantMode.is_per_channel(act_mode)
        for ot in n.outputs:
            ot.key_axis = ot.key_axis if is_perchannel else None


def deal_with_key_axis_conflict(graph, config):
    for n in graph.nodes:
        for t in n.outputs:
            flag = False
            for nc in n.children:
                if t in nc.inputs and not QuantMode.is_per_tensor(nc.attrs["q_mode_activation"]):
                    flag = True
                    break
            if not flag:
                t.key_axis = None

    def condition_func(node, parent_node, edge_tensor):
        return edge_tensor.key_axis is not None and edge_tensor.key_axis != edge_tensor.attrs['key_axis_request'][node]
    stm_nodes = graph.insert_dummy_node_ahead(OpType.BiasAdd, condition_func)
    for n in stm_nodes:
        edge_tensor = n.inputs[0]
        node = n.children[0]
        n.outputs[0].key_axis = edge_tensor.attrs['key_axis_request'][node]
        # atensor = edge_tensor.clone(graph.get_valid_tensor_name(edge_tensor.name + "_constant"))
        # atensor.key_axis = None
        # cnode = PyNode(graph.get_valid_node_name(n.name + "_constant"), OpType.Constant)
        # cnode.additional = True
        # cnode.constants['weights'] = PyTensor(
        #     cnode.name + "_weights", torch.tensor(0.0, device=edge_tensor.betensor.device), dtype=Dtype.FP32)
        # cnode.constants['weights'].ir_dtype = Dtype.FP32
        # cnode.attrs.update(n.attrs.clone())
        # cnode.attrs['q_mode_activation'] = QuantMode.to_per_tensor(cnode.attrs['q_mode_activation'])
        # cnode.attrs['q_mode_weight'] = QuantMode.to_per_tensor(cnode.attrs['q_mode_weight'])
        # atensor.ir_shape = cnode.constants['weights'].ir_shape
        # atensor.ir_dtype = cnode.constants['weights'].ir_dtype
        # cnode.add_output(atensor)
        # n.add_input(atensor)
        # graph.add_node(cnode)
        n.constants['biases'] = PyTensor(
            n.name + "_biases", torch.tensor(0.0, device=edge_tensor.betensor.device), dtype=Dtype.FP32)


def enable_activation_perchannel(graph, config):
    """
    because now support key_axis in ops, so we should set output.key_axis=None, so when using OptForward
    also should do this init.
    """
    enable_activation_perchannel = False

    for _, sg in graph.subgraph_map.items():
        for sn in sg.nodes:
            for ot in sn.outputs:
                ot.key_axis = None

    for node in graph.nodes:
        for ot in node.outputs:
            ot.key_axis = None
        if not enable_activation_perchannel:
            enable_activation_perchannel = QuantMode.is_per_channel(node.attrs['q_mode_activation'])

    if enable_activation_perchannel:
        OPT_INFO(f"Optimizer enables perchannel quantization for activation tensors.")
        before_activation_perchannel(graph, config)
        deduce_key_axis(graph, config)
        deal_with_key_axis_conflict(graph, config)
        after_activation_perchannel(graph, config)
        # name = f"{name}"
        # graph.serialize(f"{name}.txt", f"{name}.bin")
