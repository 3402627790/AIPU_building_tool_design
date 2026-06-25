"""
//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2022 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
"""

# cython: language_level=3
# !/usr/bin/python
# -*- coding: UTF-8 -*-

from ...logger import *
import numpy as np
from AIPUBuilder.core import Target, Profiler
from .profiler_config import *


def get_actuals(node):
    actual_nodes = []

    def get_actuals_internal(internal_node):
        if internal_node["op_type"] == "SubGraph":
            if "plugin" in internal_node and internal_node["plugin"] != "" and internal_node["plugin"] != 0:
                actual_nodes.append(internal_node)
            else:
                for n in internal_node["internal_nodes"]:
                    get_actuals_internal(n)
        else:
            actual_nodes.append(internal_node)
    get_actuals_internal(node)
    return actual_nodes


def is_x1():
    target_name = Target().name.upper()
    if target_name.startswith("X1") or target_name == "Z3_1204":
        return True
    return False


def fetch_info_from_binary(graph_json, binary):
    profiler = Profiler()
    profiler.unregister_all()
    info = graph_json["profile"]

    events = info["profile events"]
    if events == "default":
        unit = info["prof_unit"]
        profiler.register_default_events(unit)
    else:
        if isinstance(events, str):
            events = events.strip().split(",")
        events = [int(e) for e in events]
        events.sort()
        for event in events:
            profiler.register_event(event)

    current_events = profiler.registered_events()
    normal_events = [eid for eid in current_events if eid < 256]
    special_events = [eid for eid in current_events if eid >= 256]

    normal_slots = 4
    if is_x1():
        normal_slots = 8

    perf_nodes = []
    for node in graph_json["nodes"]:
        if "perf_node" in node and node["perf_node"] == True:
            actual_node = get_actuals(node)
            perf_nodes.extend([n for n in actual_node if "perf_node" in n and n["perf_node"] == True])

    perf_nodes_num = len(perf_nodes)

    ret = list()
    head_size = 24
    header = binary[0:head_size]
    binary = binary[head_size:]
    node_bin_size = normal_slots * 8 + len(special_events) * 4

    current_offset = header[4:8]
    current_offset.dtype = np.int32
    current_offset = int(current_offset)
    runtime_perf_num = (current_offset - head_size) // node_bin_size
    if perf_nodes_num == 0:
        FATAL("[Profiler] no runable node detected, this model may do nothing.")

    if runtime_perf_num != perf_nodes_num:
        FATAL("[Profiler] incompatiable perf node number in graph.json and binary")
    ops_count = 0
    for node_id in range(perf_nodes_num):
        ret_dict = dict()
        node = perf_nodes[node_id]
        ret_dict["name"] = node["layer node name"]
        node["ops_count"] = int(node["ops_count"]) if "ops_count" in node else 0
        ret_dict["ops_count"] = node["ops_count"]
        ops_count += ret_dict["ops_count"]

        bin = binary[node_id*node_bin_size:(node_id+1)*node_bin_size]
        for idx, event in enumerate(normal_events):
            num = bin[8 * idx:8 * (idx + 1)]
            num.dtype = np.int64
            num = int(num)
            ret_dict[event] = num
        bin = bin[normal_slots * 8:]
        for idx, event in enumerate(special_events):
            num = bin[4 * idx:4 * (idx + 1)]
            num.dtype = np.uint32
            num = int(num)
            ret_dict[event] = num
        ret.append(ret_dict)

    for node_id in range(perf_nodes_num - 1, 0, -1):
        for id, event in enumerate(normal_events):
            ret[node_id][event] = ret[node_id][event] - ret[node_id - 1][event]

    for node_id in range(perf_nodes_num):
        ret[node_id]["total_ops"] = ops_count
    return ret


def fetch_custom_record_from_binary(graph_json, binary):
    if "auxiliary info" not in graph_json["profile"]:
        return dict()
    aux = graph_json["profile"]["auxiliary info"]
    custom_buffer_offset = aux["custom_buffer_offset"]
    max_perf_offset = aux["max_perf_offset"]

    custom_buffer = binary[custom_buffer_offset:max_perf_offset]
    header = binary[:24]
    header.dtype = np.uint32
    custom_buffer.dtype = np.uint32

    if header[2] < custom_buffer_offset:
        FATAL(
            f"[Profiler] the default profile buffer maybe overflow(uint32) {custom_buffer_offset} -> {header[2]}, please close some events and run buildtool again")

    record_num = (header[2] - custom_buffer_offset) // 12

    if header[2] >= max_perf_offset:
        FATAL(
            f"[Profiler] the default profile buffer is full, please set max_sample_point_num to {1 + record_num} and run buildtool again")

    custom_exec = dict()
    for exec_id in range(record_num):
        value = custom_buffer[exec_id * 3: (exec_id + 1) * 3]
        custom_eid = int(value[0])
        tick_low = int(value[1])
        tick_high = int(value[2])
        tick = tick_low + (tick_high << 32)
        eid = custom_eid // 2
        if eid not in custom_exec:
            custom_exec[eid] = list()

        if custom_eid % 2 == 0:
            custom_exec[eid].append(("begin", tick))
        else:
            custom_exec[eid].append(("end", tick))
    return custom_exec


def get_perf_data_x1(graph_json, binary, args):
    graph_name = graph_json["graph_name"]
    verbose = int(args.verbose)
    verbose = np.clip(verbose, 0, 2)
    freq = args.frequency if args.frequency else graph_json["profile"]["frequency"]
    set_frequency(freq)
    nodes_info = fetch_info_from_binary(graph_json, binary)
    nodes = dict()
    tensors = graph_json["tensors"]

    if verbose >= 2:
        fetch_custom_record_from_binary(graph_json, binary)

    all_actuals = []
    for node in graph_json["nodes"]:
        actuals = get_actuals(node)
        all_actuals.extend(actuals)

    graph_json["actual_nodes"] = all_actuals
    for node in all_actuals:
        nodes[node["layer node name"]] = node
    input_tensors = []

    for node in all_actuals:
        if node["op_type"] == "Input":
            input_tensors.append(node["outputs"][0])

    input_tensor_shape = [tensors[t]["shape"] for t in input_tensors]
    node_dict = dict()
    for node in all_actuals:
        node_dict[node["layer node name"]] = node

    for node in nodes_info:
        name = node["name"]
        input_names = nodes[name]["inputs"]
        output_names = nodes[name]["outputs"]
        node["inputs"] = dict()
        node["outputs"] = dict()
        if input_names != None and output_names != "null":
            for inp in input_names:
                node["inputs"][inp] = tensors[inp]
        if output_names != None and output_names != "null":
            for out in output_names:
                node["outputs"][out] = tensors[out]
        node["model input shapes"] = ",".join(input_tensor_shape)
        node["operator_id"] = node_dict[name]["operator_id"]
        node["plugin"] = node_dict[name]["plugin"] if "plugin" in node_dict[name] else ""
    return nodes_info
