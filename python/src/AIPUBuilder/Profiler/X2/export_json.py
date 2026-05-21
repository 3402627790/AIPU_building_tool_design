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

import json
import copy

from AIPUBuilder.core import Target, Profiler
from AIPUBuilder.logger import *

from .load_perf_data import load_perf_data, get_perf_nodes_from_json


def export_json(graph_json, perf_data, frequency):

    ret = dict()
    ret["graph_name"] = graph_json["graph_name"]
    ret["frequency"] = frequency  # MHz

    perf_nodes = get_perf_nodes_from_json(graph_json)

    graph = dict()
    graph["nodes"] = dict()
    graph["tensors"] = graph_json["tensors"]

    # copy
    for node_id, node in perf_nodes.items():
        node_dict = dict()
        name = node["layer node name"]
        node_dict["layer node name"] = name
        node_dict["op_type"] = node["op_type"]
        node_dict["plugin"] = node["plugin"]
        node_dict["ops_count"] = node["ops_count"]
        node_dict["inputs"] = node["inputs"]
        node_dict["outputs"] = node["outputs"]
        # this constants after apply plugin / wdc
        node_dict["constants"] = node["constants"]
        node_dict["constant_size"] = node["constant_size"]
        if "params" in node:
            node_dict["params"] = node["params"]
        if "internal_nodes" in node:
            node_dict["internal_nodes"] = []
            for inode in node.get("internal_nodes", []):
                # avoid death loop
                if inode.get("internal_nodes", []) == []:
                    node_dict["internal_nodes"].append(inode)
        graph["nodes"][node_id] = node_dict

    ret["graph"] = graph
    profile_data = copy.deepcopy(perf_data)
    for subgraph in profile_data["subgraphs"]:
        for node in subgraph["nodes"]:
            stamp = node["stamp"]
            update_stamp = {}
            info_name = ["aiff", "dma", "tec", "custom"]
            for idx, info in enumerate(stamp):
                for name in info_name:
                    update_stamp[f"{name}{idx}"] = info[name]
            node["stamp"] = update_stamp
    if "subgraph switch cycles" in profile_data:
        profile_data.pop("subgraph switch cycles")
    profile_data.pop("items")
    profile_data.pop("summaries")
    if "emu" in profile_data:
        profile_data.pop("emu")
    ret["profile_data"] = profile_data
    return ret
