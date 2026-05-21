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
from AIPUBuilder.core import Target, Profiler
from .profiler_config import *
from .prof_x1 import fetch_info_from_binary, get_actuals




def x1_get_data(graph_json, binary, **kwargs):
    graph_name = graph_json["graph_name"]
    freq = kwargs["frequency"] if "frequency" in kwargs else graph_json["profile"]["frequency"]
    set_frequency(freq)
    nodes_info = fetch_info_from_binary(graph_json, binary)
    ret = dict()
    profile_data = list()
    profiler = Profiler()
    current_events = profiler.registered_events()
    rename_item = dict()
    rename_byte_item = dict()
    for new_name in BASE_TITLE_ITEM_CONFIG:
        item = BASE_TITLE_ITEM_CONFIG[new_name]
        if isinstance(item, NormalEventItem):
            rename_item[item.eid] = new_name
        if isinstance(item, BytesBaseItem):
            rename_byte_item[item.eid] = new_name

    for node in nodes_info:
        update_node = dict()
        update_node["name"] = node["name"]

        rt_info = dict()
        for eid in current_events:
            if eid in rename_item:
                rt_info[rename_item[eid]] = node[eid]
            elif eid in rename_byte_item:
                rt_info[rename_byte_item[eid]] = node[eid] * Target().axi_bandwidth // 8
            else:
                rt_info[profiler.event_description(eid)] = node[eid]
        update_node["runtime_information"] = rt_info
        profile_data.append(update_node)

    all_actuals = dict()
    for node in graph_json["nodes"]:
        actuals = get_actuals(node)
        for n in actuals:
            all_actuals[n["layer node name"]] = n

    ret["profile_data"] = profile_data
    ret["graph_name"] = graph_name
    ret["frequency"] = freq
    graph = dict()
    graph["nodes"] = all_actuals
    graph["tensors"] = graph_json["tensors"]
    ret["graph"] = graph

    return ret