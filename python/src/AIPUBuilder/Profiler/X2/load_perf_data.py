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
import math
import copy
import numpy as np
from enum import IntEnum
from AIPUBuilder.logger import *
from AIPUBuilder.core import Target, Profiler
from AIPUBuilder._C._builder import ISAType


def get_event_description(eid):
    profiler = Profiler()
    profiler.unregister_all()
    return profiler.event_description(eid)


def get_perf_nodes_from_json(graph_json):
    # flat tree to list
    # group by perf_node_id
    perf_nodes = {}

    def perf_nodes_dfs(node):
        # check
        # NOTE
        # Not all plugin != ""
        # Not all perf node subgraph == Subgraph
        # e.g. LayerConvetor
        if node.get("perf_node", False) and node.get("perf_node_id", 0) > 0:
            perf_nodes[node["perf_node_id"]] = node

        # dfs nodes
        for n in node.get("internal_nodes", []):
            perf_nodes_dfs(n)
        # dfs root
        for n in node.get("nodes", []):
            perf_nodes_dfs(n)

    # search
    perf_nodes_dfs(graph_json)

    return perf_nodes


def get_mac_weight_dtype(node):
    """
    Search the dtype of a node, return first result of internal nodes
    Only check the weight of conv/fc
    """
    if 'internal_nodes' in node:
        for n in node['internal_nodes']:
            plugin = n['plugin']
            dtype = get_mac_weight_dtype(n)
            if dtype:
                return dtype

    op_type = node["op_type"]
    if op_type not in ["Convolution", "DepthwiseConv", "ConvTranspose", "FullyConnected", "Convolution3D", "ConvTranspose3D"]:
        return None

    if "constants" in node and node["constants"]:
        if "weights" in node["constants"]:
            return node["constants"]["weights"]["dtype"]
    return None


def fetch_subgraph_info(graph_json, subgraph_sec):
    aux_info = graph_json["profile"]["auxiliary info"]
    max_subgraph_node_count = aux_info.get(
        "max_subgraph_node_count", np.iinfo(np.int32).max)
    core_num = aux_info["core_num"]
    target = graph_json["platform target"]

    core_id = subgraph_sec[:4]
    core_id.dtype = np.uint32
    core_id = int(core_id)
    start_tick = subgraph_sec[4:12]
    start_tick.dtype = np.uint64
    start_tick = int(start_tick)
    nodes_count = subgraph_sec[12:16]
    nodes_count.dtype = np.uint32
    nodes_count = int(nodes_count)

    if nodes_count > max_subgraph_node_count:
        WARN(
            f"[Profiler] {nodes_count} node in data exceeded the max subgraph node count limit {max_subgraph_node_count}")
        nodes_count = max_subgraph_node_count

    # if len(actuals_perf) != nodes_count:
    #     FATAL(f"[Profiler] Incompatible perf node number in graph.json:{len(actuals_perf)} and binary:{nodes_count}")

    profile_data = dict()
    profile_data["start_tick"] = start_tick
    perf_node_ids = []
    end_ticks = []
    end_read_data = []
    end_write_data = []
    tec_events_end_dict = dict()

    aux_info = graph_json["profile"]["auxiliary info"]
    node_dump_bytes = aux_info["node_dump_bytes"]
    tec_events = aux_info["tec_events"]
    if tec_events is None:
        tec_events = list()
        aux_info["tec_events"] = tec_events
    noc_events = list()
    if "noc_events" in aux_info and aux_info["noc_events"]:
        noc_events = aux_info["noc_events"]
    tec_num = aux_info["tec_num"]
    asid_perf_items = aux_info["asid_perf_items"] if "asid_perf_items" in aux_info else [
    ]
    end_asid_read = {}
    end_asid_write = {}
    for item in asid_perf_items:
        end_asid_read[item] = []
        end_asid_write[item] = []

    # max_subgraph_node_count
    for node_idx in range(nodes_count):
        buf = subgraph_sec[16 + node_dump_bytes *
                           node_idx: 20 + node_dump_bytes * (node_idx + 1)]
        # Same as 'layer_id' of dump IR
        perf_node_id = buf[:4]
        perf_node_id.dtype = np.uint32
        perf_node_id = int(perf_node_id)
        end_tick = buf[4:12]
        end_tick.dtype = np.uint64
        end_tick = int(end_tick)

        if end_tick <= start_tick:
            FATAL(
                f"[Profiler] End tick {end_tick} before start tick {start_tick} in binary")

        end_read = buf[12:16]
        end_read.dtype = np.uint32
        end_read = int(end_read)
        end_write = buf[16:20]
        end_write.dtype = np.uint32
        end_write = int(end_write)
        base = 20
        for item in asid_perf_items:
            asid_read = buf[base:base+4]
            asid_write = buf[base+4:base+8]
            asid_read.dtype = np.uint32
            asid_write.dtype = np.uint32
            asid_read = int(asid_read)
            asid_write = int(asid_write)
            base += 8
            end_asid_read[item].append(asid_read)
            end_asid_write[item].append(asid_write)

        for eid in noc_events:
            end_value = buf[base:base+4]
            end_value.dtype = np.uint32
            end_value = int(end_value)
            # all tecs share the same noc events
            for tec_id in range(tec_num):
                name = f"Tec{tec_id} {get_event_description(eid)}"
                if name not in tec_events_end_dict:
                    tec_events_end_dict[name] = list()
                tec_events_end_dict[name].append(end_value)
            base = base + 4

        for tec_id in range(tec_num):
            for eid in tec_events:
                name = f"Tec{tec_id} {get_event_description(eid)}"
                if name not in tec_events_end_dict:
                    tec_events_end_dict[name] = list()
                val = buf[base:base+4]
                val.dtype = np.uint32
                val = int(val)
                tec_events_end_dict[name].append(val)
                base = base + 4

        perf_node_ids.append(perf_node_id)
        end_ticks.append(end_tick)
        end_read_data.append(end_read)
        end_write_data.append(end_write)

    profile_data["perf_node_ids"] = perf_node_ids
    profile_data["end_ticks"] = end_ticks
    profile_data["end_reads"] = end_read_data
    profile_data["end_writes"] = end_write_data
    profile_data["core_id"] = core_id
    profile_data["tec_events_ends"] = tec_events_end_dict
    if asid_perf_items:
        profile_data["asid_perf_read"] = end_asid_read
        profile_data["asid_perf_write"] = end_asid_write
    return profile_data


def fetch_tec_record_info(graph_json, tec_sec, tecs_dict):
    aux_info = graph_json["profile"]["auxiliary info"]
    tec_num = aux_info["tec_num"]
    tec_area_size = aux_info["tec_area_size"]
    tec_dump_bytes = aux_info["tec_dump_bytes"]
    tec_events = aux_info["tec_events"]

    if tec_events is None:
        tec_events = list()

    is_full = False
    max_perf_tec_count = 0
    max_count_from_aux_info = aux_info["max_perf_tec_count"]
    is_fast_perf = "enable_fast_perf" in aux_info and aux_info["enable_fast_perf"]
    for tec_id in range(tec_num):
        tec_record = tec_sec[tec_area_size *
                             tec_id: tec_area_size * (tec_id + 1)]
        tec_record.dtype = np.uint32
        # full = tec_record[0]
        tec_record_num = tec_record[1]
        full = (tec_record_num > max_count_from_aux_info)
        if is_fast_perf:
            tec_record_num = 0
        elif full:
            is_full = True
            max_perf_tec_count = max(max_perf_tec_count, tec_record_num)
            continue
        if is_full and not is_fast_perf:
            continue
        tecs_dict[tec_id]["tec_record_num"] = tec_record_num
        tecs_dict[tec_id]["tec_full"] = full
        record = []
        for exec_id in range(tec_record_num):
            tec_exec = dict()
            offset = 2 + exec_id * tec_dump_bytes // 4
            tec_data = tec_record[offset:offset+tec_dump_bytes//4]
            tec_stamp = tec_data[:2]
            tec_stamp.dtype = np.uint64
            tec_stamp = int(tec_stamp)
            if tec_events:
                for idx, eid in enumerate(tec_events):
                    e_val = tec_data[2+idx]
                    tec_exec[eid] = int(e_val)
            tec_exec["stamp"] = tec_stamp
            record.append(tec_exec)
        tecs_dict[tec_id]["tec_record"] = record
    return is_full, max_perf_tec_count


def fetch_aiff_record_info(graph_json, aiff_sec, tecs_dict):
    aux_info = graph_json["profile"]["auxiliary info"]
    tec_num = aux_info["tec_num"]
    aiff_dump_bytes = aux_info["aiff_dump_bytes"]
    aiff_area_size = aux_info["aiff_area_size"]
    aiff_events = aux_info["aiff_events"] if "aiff_events" in aux_info else list(
    )

    is_x3 = (Target().isa >= ISAType.ZYv6)
    if is_x3:
        AIFF_Index = IntEnum('AIFF_Index', ['START', 'BUSY', 'MTP0', 'MTP1', 'ITP0', 'ITP1', 'PTP',
                                            'RDMA_WEIGHT_RAED', 'RDMA_ACT_READ', 'UDMA_PARAM_READ', 'UDMA_ACT_WRITE',
                                            'WDMA_ACT_WRITE', 'BIF_READ', 'BIF_WRITE', 'COM0', 'COM1', 'ID'])
    else:
        AIFF_Index = IntEnum('AIFF_Index', ['START', 'BUSY', 'MTP0', 'MTP1', 'ITP', 'PTP', 'WRB',
                                            'RDMA_WEIGHT_RAED', 'RDMA_ACT_READ', 'UDMA_PARAM_READ', 'UDMA_ACT_WRITE',
                                            'WDMA_ACT_WRITE', 'BIF_READ', 'BIF_WRITE', 'COM0', 'COM1'])

    is_full = False
    max_perf_aiff_count = 0  # changing value, used to save max count of current graph
    # fixed value, gotten from auxiliary info
    max_count_from_aux_info = aux_info["max_perf_aiff_count"]
    is_fast_perf = "enable_fast_perf" in aux_info and aux_info["enable_fast_perf"]
    for tec_id in range(tec_num):
        aiff_record = aiff_sec[aiff_area_size *
                               tec_id: aiff_area_size * (tec_id + 1)]

        aiff_record.dtype = np.uint32
        # full = aiff_record[0]
        aiff_execute_num = aiff_record[1]
        full = (aiff_execute_num > max_count_from_aux_info)

        if is_fast_perf:
            aiff_execute_num = 0
        elif full:
            is_full = True
            max_perf_aiff_count = max(max_perf_aiff_count, aiff_execute_num)
            continue
        if is_full and not is_fast_perf:  # no need to parse info if there is at least one full tec
            continue
        tecs_dict[tec_id]["aiff_record_num"] = aiff_execute_num
        tecs_dict[tec_id]["aiff_full"] = full
        record = []
        for exec_id in range(aiff_execute_num):
            aiff_exec = dict()
            offset = 16 + exec_id * aiff_dump_bytes // 4
            aiff_data = aiff_record[offset:offset+aiff_dump_bytes//4]
            aiff_start = aiff_data[0:2]
            aiff_start.dtype = np.uint64
            aiff_start = int(aiff_start)
            busy_cycle = int(aiff_data[AIFF_Index.BUSY])
            mtp0_busy_cycle = int(aiff_data[AIFF_Index.MTP0])
            mtp1_busy_cycle = int(aiff_data[AIFF_Index.MTP1])
            ptp_busy_cycle = int(aiff_data[AIFF_Index.PTP])
            rdma_weight_rd_cnt = int(aiff_data[AIFF_Index.RDMA_WEIGHT_RAED])
            rdma_act_rd_cnt = int(aiff_data[AIFF_Index.RDMA_ACT_READ])
            udma_param_rd_cnt = int(aiff_data[AIFF_Index.UDMA_PARAM_READ])
            udma_act_wr_cnt = int(aiff_data[AIFF_Index.UDMA_ACT_WRITE])
            wdma_act_wr_cnt = int(aiff_data[AIFF_Index.WDMA_ACT_WRITE])
            bif_rd_cnt = int(aiff_data[AIFF_Index.BIF_READ])
            bif_wr_cnt = int(aiff_data[AIFF_Index.BIF_WRITE])

            aiff_exec["start"] = aiff_start
            aiff_exec["duration"] = busy_cycle

            aiff_exec["mtp0_busy_cycle"] = mtp0_busy_cycle
            aiff_exec["mtp1_busy_cycle"] = mtp1_busy_cycle
            aiff_exec["ptp_busy_cycle"] = ptp_busy_cycle
            aiff_exec["rdma_weight_read_count"] = rdma_weight_rd_cnt
            aiff_exec["rdma_act_read_count"] = rdma_act_rd_cnt
            aiff_exec["udma_param_read_count"] = udma_param_rd_cnt
            aiff_exec["udma_act_write_count"] = udma_act_wr_cnt
            aiff_exec["wdma_act_write_count"] = wdma_act_wr_cnt
            aiff_exec["bif_read_count"] = bif_rd_cnt
            aiff_exec["bif_write_count"] = bif_wr_cnt
            if is_x3:
                itp0_busy_cycle = int(aiff_data[AIFF_Index.ITP0])
                aiff_exec["itp0_busy_cycle"] = itp0_busy_cycle
                itp1_busy_cycle = int(aiff_data[AIFF_Index.ITP1])
                aiff_exec["itp1_busy_cycle"] = itp1_busy_cycle
                aiff_exec["physical_aiff_id"] = int(aiff_data[AIFF_Index.ID])
            else:
                itp_busy_cycle = int(aiff_data[AIFF_Index.ITP])
                aiff_exec["itp_busy_cycle"] = itp_busy_cycle
                wrb_busy_cycle = int(aiff_data[AIFF_Index.WRB])
                aiff_exec["wrb_busy_cycle"] = wrb_busy_cycle
            if len(aiff_events) >= 1:
                aiff_exec["com0"] = int(aiff_data[AIFF_Index.COM0])
            if len(aiff_events) >= 2:
                aiff_exec["com1"] = int(aiff_data[AIFF_Index.COM1])
            record.append(aiff_exec)
        tecs_dict[tec_id]["aiff_record"] = record
    return is_full, max_perf_aiff_count


def fetch_custom_record_info(graph_json, custom_sec, tecs_dict):
    aux_info = graph_json["profile"]["auxiliary info"]
    tec_num = aux_info["tec_num"]
    custom_dump_bytes = aux_info["custom_dump_bytes"]
    custom_area_size = aux_info["custom_area_size"]
    is_full = False
    max_perf_custom_count = 0
    max_count_from_aux_info = aux_info["max_perf_custom_count"]

    is_fast_perf = "enable_fast_perf" in aux_info and aux_info["enable_fast_perf"]
    for tec_id in range(tec_num):
        custom_record = custom_sec[custom_area_size *
                                   tec_id: custom_area_size * (tec_id + 1)]
        custom_record.dtype = np.uint32
        # full = custom_record[0]
        custom_record_num = custom_record[1]
        full = (custom_record_num > max_count_from_aux_info)
        if is_fast_perf:
            custom_record_num = 0
        elif full:
            is_full = True
            max_perf_custom_count = max(
                max_perf_custom_count, custom_record_num)
            continue
        if is_full and not is_fast_perf:
            continue
        tecs_dict[tec_id]["custom_record_num"] = custom_record_num
        tecs_dict[tec_id]["custom_full"] = full
        record = []
        custom_exec = dict()
        for exec_id in range(custom_record_num):
            offset = 2 + exec_id * custom_dump_bytes // 4
            custom_data = custom_record[offset:offset+custom_dump_bytes//4]
            custom_eid = int(custom_data[0])
            eid = custom_eid // 2
            if eid not in custom_exec:
                custom_exec[eid] = list()
            custom_tick = custom_data[1:3]
            custom_tick.dtype = np.uint64
            custom_tick = int(custom_tick)
            if custom_eid % 2 == 0:
                custom_exec[eid].append(("begin", custom_tick))
            else:
                custom_exec[eid].append(("end", custom_tick))

        custom_exec_ = dict()
        for eid in custom_exec:
            custom_exec_[eid] = list()
            begin_stack = list()
            for event in custom_exec[eid]:
                if event[0] == "begin":
                    begin_stack.append(event[1])
                else:
                    if len(begin_stack) != 0:
                        begin_tick = begin_stack.pop()
                        end_tick = event[1]
                        duration = end_tick - begin_tick
                        record = dict()
                        record["start"] = begin_tick
                        record["duration"] = duration
                        custom_exec_[eid].append(record)
                    else:
                        continue

        tecs_dict[tec_id]["custom_record"] = custom_exec_
    return is_full, max_perf_custom_count


def fetch_dma_record_info_v1(graph_json, dma_sec, tecs_dict):
    aux_info = graph_json["profile"]["auxiliary info"]

    tec_num = aux_info["tec_num"]
    dma_dump_bytes = aux_info["dma_dump_bytes"]
    dma_area_size = aux_info["dma_area_size"]
    is_full = False
    max_perf_dma_count = 0
    max_count_from_aux_info = aux_info["max_perf_dma_count"]
    axi_bandwidth = Target().axi_bandwidth // 8
    is_fast_perf = "enable_fast_perf" in aux_info and aux_info["enable_fast_perf"]
    for tec_id in range(tec_num):
        dma_record = dma_sec[dma_area_size *
                             tec_id: dma_area_size * (tec_id + 1)]
        dma_record.dtype = np.uint32
        # full = dma_record[0]
        dma_record_num = dma_record[1]
        full = (dma_record_num > max_count_from_aux_info)
        if is_fast_perf:
            dma_record_num = 0
        elif full:
            is_full = True
            max_perf_dma_count = max(max_perf_dma_count, dma_record_num)
            continue
        if is_full and not is_fast_perf:
            continue
        tecs_dict[tec_id]["dma_record_num"] = dma_record_num
        tecs_dict[tec_id]["dma_full"] = full
        record = []
        for exec_id in range(dma_record_num):
            dma_exec = dict()
            offset = 16 + exec_id * dma_dump_bytes // 4
            dma_data = dma_record[offset:offset+dma_dump_bytes//4]
            dma_start = dma_data[:2]
            dma_start.dtype = np.uint64
            dma_start = int(dma_start)

            busy_cycle = int(dma_data[2])
            dma_exec["start"] = dma_start
            dma_exec["duration"] = busy_cycle
            read_num = int(dma_data[4])
            write_num = int(dma_data[6])
            dma_exec["read_bytes"] = read_num * axi_bandwidth
            dma_exec["write_bytes"] = write_num * axi_bandwidth
            record.append(dma_exec)
        tecs_dict[tec_id]["dma_record"] = record
    return is_full, max_perf_dma_count


def fetch_dma_block(graph_json, binary):
    dma_blocks = []
    bin_offset = 0

    def _check_dma_block(offset):
        cur_head = binary[offset:offset+4].tolist()
        # DMA
        head = [0x44, 0x4D, 0x41, 0x0]
        return head == cur_head

    while(_check_dma_block(bin_offset)):
        size = binary[bin_offset+4:bin_offset+8]
        size.dtype = np.uint32
        size = int(size)
        block = binary[bin_offset:bin_offset+size]
        dma_blocks.append(block)
        bin_offset = bin_offset + size

    aux_info = graph_json["profile"]["auxiliary info"]
    dma_dump_bytes = aux_info["dma_dump_bytes"]
    tec_num = aux_info["tec_num"]
    core_num = aux_info["core_num"]

    record_data = list()
    for core_id in range(core_num):
        record_data.append(list())
        for tec_id in range(tec_num):
            record_data[-1].append(list())

    axi_bandwidth = Target().axi_bandwidth // 8
    for block in dma_blocks:
        size = block[4:8]
        size.dtype = np.uint32
        size = int(size)

        tec_id = block[8:10]
        core_id = block[10:12]
        tec_id.dtype = np.uint16
        core_id.dtype = np.uint16
        tec_id = int(tec_id)
        core_id = int(core_id)
        head_size = 64
        rec_num = (size - head_size) // dma_dump_bytes
        offset = head_size
        for _ in range(rec_num):
            dma_exec = dict()
            dma_data = block[offset:offset + dma_dump_bytes]
            dma_data.dtype = np.uint32

            dma_start = dma_data[:2]
            dma_start.dtype = np.uint64
            dma_start = int(dma_start)

            busy_cycle = int(dma_data[2])
            dma_exec["start"] = dma_start
            dma_exec["duration"] = busy_cycle
            read_num = int(dma_data[4])
            write_num = int(dma_data[6])
            dma_exec["read_bytes"] = read_num * axi_bandwidth
            dma_exec["write_bytes"] = write_num * axi_bandwidth
            offset = offset + dma_dump_bytes
            record_data[core_id][tec_id].append(dma_exec)
    return record_data, binary[bin_offset:]


def fetch_info_from_binary(graph_json, binary):
    record_data, binary = fetch_dma_block(graph_json, binary)
    # perf_subgraphs = get_perf_subgraphs_from_json(graph_json)
    perf_nodes = get_perf_nodes_from_json(graph_json)

    if len(perf_nodes) < 1:
        FATAL(f"[Profiler] no runable node detected")

    flat_input_shape = []
    flat_output_symbol = []

    aux_info = graph_json["profile"]["auxiliary info"]
    subgraph_section_offset = aux_info["subgraph_sect_offset"]
    tec_section_offset = aux_info["tec_sect_offset"]
    aiff_section_offset = aux_info["aiff_sect_offset"]
    dma_section_offset = aux_info["dma_sect_offset"]
    custom_section_offset = aux_info["custom_sect_offset"]
    size = aux_info["subgraph_record_size"]

    input_shape_offset = aux_info.get("input_shape_offset", -1)
    input_shape_count = aux_info.get("input_shape_count", 0)
    output_symbol_offset = aux_info.get("output_symbol_offset", -1)
    output_symbol_count = aux_info.get("output_symbol_count", 0)
    output_shape_count = aux_info.get("output_shape_count", 0)
    magic_number_offset = aux_info.get("magic_number_offset", -1)

    # deprecated, instead: layer_id <-> perf_node_id
    # subgraph_order = aux_info["subgraph_order"]
    subgraph_levels = aux_info["subgraph_levels"]

    version = aux_info["graph_json_version"]

    if "disable_dma_perf" in aux_info and aux_info["disable_dma_perf"]:
        WARN(
            "[Profiler] GBuilder has disabled DMA perf as it may cost time and memory,"
            " all data about DMA in the report would be zero."
            " If need DMA information, add --enable_dma_perf when using GBuilder."
        )

    tec_num = aux_info["tec_num"]
    core_num = int(aux_info["core_num"])

    max_perf_aiff_count = 0
    max_perf_dma_count = 0
    max_perf_custom_count = 0
    dma_full = False
    aiff_full = False
    custom_full = False

    idx = 0
    profile_data_per_subgraph = []
    while (idx + 1) * size <= len(binary):

        if idx >= graph_json['perf nodes number']:
            WARN(
                f"skip extraneous perf node data[{idx}] in binary {size * idx}:{len(binary)}")
            break

        buffer = binary[size * idx:(idx + 1) * size]

        subgraph_sec = buffer[subgraph_section_offset:tec_section_offset]
        profile_data = fetch_subgraph_info(graph_json, subgraph_sec)

        tecs = []
        for _ in range(tec_num):
            tecs.append(dict())

        tec_sec = buffer[tec_section_offset:aiff_section_offset]
        fetch_tec_record_info(graph_json, tec_sec, tecs)

        aiff_sec = buffer[aiff_section_offset:dma_section_offset]
        _aiff_full, _max_perf_aiff_count = fetch_aiff_record_info(
            graph_json, aiff_sec, tecs)
        aiff_full = aiff_full or _aiff_full
        max_perf_aiff_count = max(max_perf_aiff_count, _max_perf_aiff_count)

        dma_sec = buffer[dma_section_offset:custom_section_offset]
        if "1.0.0" == version:
            _dma_full, _max_perf_dma_count = fetch_dma_record_info_v1(
                graph_json, dma_sec, tecs)
        else:
            FATAL(f"currently not support such graph json version {version}")
        dma_full = dma_full or _dma_full
        max_perf_dma_count = max(max_perf_dma_count, _max_perf_dma_count)

        custom_sec = buffer[custom_section_offset:]
        _custom_full, _max_perf_custom_count = fetch_custom_record_info(
            graph_json, custom_sec, tecs)
        custom_full = custom_full or _custom_full
        max_perf_custom_count = max(
            max_perf_custom_count, _max_perf_custom_count)

        if len(flat_input_shape) == 0 and input_shape_count > 0:
            magic_number_record = buffer[magic_number_offset:]
            magic_number_record.dtype = np.uint32
            magic_number = magic_number_record[0]
            if magic_number == 0x44595348:
                input_shape_record = buffer[input_shape_offset: input_shape_offset + (
                    input_shape_count * 4)]
                input_shape_record.dtype = np.uint32
                for i in range(input_shape_count):
                    flat_input_shape.append(input_shape_record[i])
                if output_symbol_count > 0:
                    output_symbol_record = buffer[output_symbol_offset: output_symbol_offset +
                                                  (output_symbol_count * 4)]
                    output_symbol_record.dtype = np.uint32
                    for i in range(output_symbol_count):
                        flat_output_symbol.append(output_symbol_record[i])
        profile_data["tecs"] = tecs
        if idx < len(subgraph_levels):
            profile_data["level"] = subgraph_levels[idx]
        else:
            profile_data["level"] = -1

        if len(profile_data["perf_node_ids"]) >= 1:
            profile_data_per_subgraph.append(profile_data)
        else:
            WARN(
                f"empty subgraph data in binary {size * idx}:{(idx + 1) * size}")

        idx += 1

    if len(profile_data_per_subgraph) == 0:
        FATAL(f"[Profiler] Not any avaliable data in binary")

    # cal idle time for each subgraph
    # the idle time of first subgraph on each core is 0
    group_by_core = {}
    for profile_data in profile_data_per_subgraph:
        if profile_data['core_id'] not in group_by_core:
            group_by_core[profile_data['core_id']] = []
        group_by_core[profile_data['core_id']].append(profile_data)
    for profile_data_list in group_by_core.values():
        sorted_list = sorted(profile_data_list, key=lambda x: x['start_tick'])
        i = 1
        while i < len(sorted_list):
            sorted_list[i]['pre_idle'] = sorted_list[i]['start_tick'] - \
                sorted_list[i - 1]['end_ticks'][-1]
            i += 1
            # post_idle

    err_msg = ""
    if aiff_full:
        err_msg += f"set max_aiff_sample_point_num to {max_perf_aiff_count},"
    if dma_full:
        err_msg += f"set max_dma_sample_point_num to {max_perf_dma_count},"
    if custom_full:
        err_msg += f"set max_custom_sample_point_num to {max_perf_custom_count},"

    is_fast_perf = "enable_fast_perf" in aux_info and aux_info["enable_fast_perf"]
    if not is_fast_perf and (aiff_full or dma_full or custom_full):
        FATAL(
            f"[Profiler] the default profile buffer is full, please {err_msg} and run buildtool again")

    vars_map = dict()
    symbols = graph_json.get("symbols", None)
    if symbols is not None:
        for k, v in symbols.items():
            t = v["type"]
            i = v["dims"][0]
            if t == 0:
                # is input symbol
                if i < 0 or i >= len(flat_input_shape):
                     FATAL(
                        f"Dim of input symbols:{k} must in range [0, {len(flat_input_shape)}], please check graph.json")
                vars_map[k] = flat_input_shape[i]
            elif t == 1:
                # is output symbol
                # NOTE: in graph it starts from output shape
                # e.g. [1, 2, 3] + [r1, r2]
                #  `output shape is [1, 2, 3]
                #  `dim of r1 is 3
                #  `dim of r2 is 4
                if i < output_shape_count or i >= output_shape_count + len(flat_output_symbol):
                    FATAL(
                        f"Dim of output symbols:{k} must in range [{output_shape_count}, {output_shape_count + len(flat_output_symbol) - 1}], please check graph.json")
                vars_map[k] = flat_output_symbol[i - output_shape_count]
    else:
        if len(flat_input_shape) > 0:
            for i, d in enumerate(flat_input_shape):
                vars_map["d" + str(i)] = d
        if len(flat_output_symbol) > 0:
            for i, d in enumerate(flat_output_symbol):
                vars_map["r" + str(output_shape_count + i)] = d

    return perf_nodes, profile_data_per_subgraph, vars_map


def eval_expression(expression, vars_map):
    '''Evaluate expression by using Expr and eval function from tongue and return a int value.
    '''
    from AIPUBuilder._C._tongue import Expr
    expr = Expr(expression)
    assert all(var.name in vars_map for var in expr.variables()
               ), "Meets unknown dim in expression %s" % expression
    vars_map = {var: vars_map[var.name] for var in expr.variables()}
    assert expr.eval(vars_map), "Fail to eval expression %s" % expression
    return int(expr.value)


def load_perf_data(graph_json, binary, freq, verbose):

    perf_nodes, profile_data_per_subgraph, vars_map = fetch_info_from_binary(
        graph_json, binary)

    # tcb_switch_cycle = 0
    # for idx in range(1, len(profile_data_per_subgraph), 1):
    #     end_pre = perf_subgraphs[idx - 1]["profile_data"]["end_ticks"][-1]
    #     start_cur = perf_subgraphs[idx]["profile_data"]["start_tick"]
    #     tcb_switch_cycle += (start_cur - end_pre)

    aux_info = graph_json["profile"]["auxiliary info"]
    tec_num = aux_info["tec_num"]
    core_num = aux_info["core_num"]
    max_core_ids = core_num - 1
    #tec_events = graph_json["profile"]["auxiliary info"]["tec_events"]
    aiff_events = aux_info["aiff_events"] if "aiff_events" in aux_info else list(
    )
    profiler = Profiler()
    aiff_event_names = [get_event_description(eid) for eid in aiff_events]

    target = Target()
    perf_actual_nodes = []
    axi_bandwidth = target.axi_bandwidth // 8
    aiff_num_per_core = target.aiff_num
    is_fast_perf = "enable_fast_perf" in aux_info and aux_info["enable_fast_perf"]

    tec_event_names = list()

    modify_l2c = (target.isa >= ISAType.ZYv6) and aux_info.get(
        "noc_events", None) is not None and 513 in aux_info["noc_events"]
    modify_l2c_by_layers = modify_l2c
    l2c_cacheline = 2048 // 8
    if modify_l2c_by_layers:
        WARN("correct footprint by each layer due to L2 cache")

    ops_groupby_dtype = {
        'i8': 0,
        'i8 x i16': 0,
        'i16': 0,
        'f16': 0,
        'f8' : 0,
        'wf4' : 0,
    }

    for profile_data in profile_data_per_subgraph:
        nodes = []

        start_tick = profile_data["start_tick"]
        perf_node_ids = profile_data["perf_node_ids"]
        end_ticks = profile_data["end_ticks"]
        end_reads = profile_data["end_reads"]
        end_writes = profile_data["end_writes"]
        tec_events_ends = profile_data["tec_events_ends"]
        tec_event_names = list(tec_events_ends.keys())
        for tec_perf_name in tec_events_ends:
            tec_events_ends[tec_perf_name] = [
                0] + tec_events_ends[tec_perf_name]

        perf_node_ids = perf_node_ids
        ticks = [start_tick] + end_ticks[:]
        reads = [0] + end_reads
        writes = [0] + end_writes

        for idx, perf_node_id in enumerate(perf_node_ids):

            if perf_node_id in perf_nodes:
                node = copy.deepcopy(perf_nodes[perf_node_id])
            else:
                FATAL(
                    f"[Profiler] cannot find perf_node_id:{perf_node_id} in graph.json")

            node["actual_start_tick"] = ticks[idx]
            node["actual_end_tick"] = ticks[idx + 1]
            node["read_data_bytes"] = (
                reads[idx + 1] - reads[idx]) * axi_bandwidth
            node["write_data_bytes"] = (
                writes[idx + 1] - writes[idx]) * axi_bandwidth
            for tec_perf_name in tec_events_ends:
                val = tec_events_ends[tec_perf_name][idx +
                                                     1] - tec_events_ends[tec_perf_name][idx]
                node[tec_perf_name] = val

            # log idle on first node
            if idx == 0 and "pre_idle" in profile_data:
                node["pre_idle"] = profile_data["pre_idle"]

            nodes.append(node)

        if "asid_perf_read" in profile_data and "asid_perf_write" in profile_data:
            asid_perf_read = profile_data["asid_perf_read"]
            asid_perf_write = profile_data["asid_perf_write"]

            for item in asid_perf_read:
                asid_read = [0] + asid_perf_read[item]
                asid_write = [0] + asid_perf_write[item]

                for idx, node in enumerate(nodes):
                    node[f"{item}_read_bytes"] = (
                        asid_read[idx + 1] - asid_read[idx]) * axi_bandwidth
                    node[f"{item}_write_bytes"] = (
                        asid_write[idx + 1] - asid_write[idx]) * axi_bandwidth

            asid = aux_info["asid_perf_items"]
            all_asid = [asid_id for asid_id in asid if "asid" in asid_id]
            all_asid = [set(id.split("_")[-1]) for id in all_asid]
            all_asids = set().union(*all_asid)
            if all_asids == set(["0", "1", "2", "3"]) and all_asid[0] & all_asid[1] == set():
                for idx, node in enumerate(nodes):
                    node[f"read_data_bytes"] = sum(
                        [node[f"{item}_read_bytes"] for item in asid_perf_read])
                    node[f"write_data_bytes"] = sum(
                        [node[f"{item}_write_bytes"] for item in asid_perf_read])

        node_tecs = []

        subgraph_dict = dict()
        subgraph_dict["Core ID"] = profile_data["core_id"]
        max_core_ids = max(max_core_ids, profile_data["core_id"])
        subgraph_dict["Level"] = profile_data["level"]
        for node in nodes:
            node_tecs = []
            node_start = node["actual_start_tick"]
            node_end = node["actual_end_tick"]

            for tec_id in range(tec_num):
                node_tec = dict()
                tec_info = profile_data["tecs"][tec_id]
                tec_records = tec_info.get("tec_record", [])
                aiff_records = tec_info.get("aiff_record", [])
                dma_records = tec_info.get("dma_record", [])
                custom_records = tec_info.get("custom_record", [])

                aiff = []
                for record in aiff_records:
                    start = record["start"]
                    if start > node_start and start < node_end:
                        aiff.append(record)
                    elif start > node_end:
                        break
                node_tec["aiff"] = aiff

                dma = []
                for record in dma_records:
                    start = record["start"]
                    if start > node_start and start < node_end:
                        dma.append(record)
                    elif start > node_end:
                        break
                node_tec["dma"] = dma

                tec = []
                for record in tec_records:
                    stamp = record["stamp"]
                    if stamp > node_start and stamp < node_end:
                        tec.append(record)
                    elif start > node_end:
                        break
                node_tec["tec"] = tec

                custom = dict()
                for eid in custom_records:
                    custom[eid] = list()
                    stamps = custom_records[eid]
                    for stamp in stamps:
                        start = stamp["start"]
                        if start > node_start and start < node_end:
                            custom[eid].append(stamp)
                        elif start > node_end:
                            break
                node_tec["custom"] = custom

                node_tecs.append(node_tec)
            node["runtime_info"] = node_tecs
        subgraph_dict["nodes"] = nodes
        perf_actual_nodes.append(subgraph_dict)

    # print(tcb_switch_cycle)
    perf_data = dict()

    total_dma_cycles = 0
    total_aiff_cycles = 0

    start = 0xFFFFFFFFFFFFFFFF
    end = 0
    subgraphs = []
    total_reads = 0
    total_writes = 0
    total_dma_reads = 0
    total_dma_writes = 0
    nodes_num = 0
    total_ops = 0
    total_ideal_cycles = 0
    total_constant_reads = 0
    total_asid_items = {}  # {item Read/Write Bytes: total}
    total_L2_hit = 0
    total_L2_miss = 0
    total_ddr_read = 0

    if "asid_perf_items" in aux_info:
        total_asid_items.update(
            {f"{item.upper()} Read Bytes": 0 for item in aux_info["asid_perf_items"]})
        total_asid_items.update(
            {f"{item.upper()} Write Bytes": 0 for item in aux_info["asid_perf_items"]})

    def find_total_cycles(periods):
        if len(periods) == 0:
            return 0
        periods.sort(key=lambda elem: elem["start"])
        start_tick = periods[0]["start"]
        end_tick = start_tick + periods[0]["duration"]
        cycles = 0
        for period in periods[1:]:
            if period["start"] <= end_tick and period["start"] + period["duration"] >= end_tick:
                end_tick = period["start"] + period["duration"]
            elif period["start"] > end_tick:
                cycles += (end_tick - start_tick)
                start_tick = period["start"]
                end_tick = period["start"] + period["duration"]

        cycles += end_tick - start_tick
        return cycles

    def eval_shapes(list_shapes, vars_map):

        def split_expression(s):
            """
            input: [max((batch*12),12),1,seq_len,128]
            output: [
                max((batch*12),12),  <------- need ',' inside
                1,
                seq_len,
                128
            ]
            """
            s = s.strip("[").rstrip("]")
            result = []
            current = ""
            parentheses_count = 0

            for char in s:
                if char == '(':
                    parentheses_count += 1
                    current += char
                elif char == ')':
                    parentheses_count -= 1
                    current += char
                elif char == ',' and parentheses_count == 0:
                    result.append(current.strip())
                    current = ""
                else:
                    current += char

            if current:
                result.append(current.strip())

            return result

        actual_shapes = []
        for inp in list_shapes:
            shape_str_list = split_expression(inp)
            shape_str_list = [shape.strip() for shape in shape_str_list]
            shape_val = [shape_str if shape_str.isdigit() else str(eval_expression(shape_str, vars_map))
                         for shape_str in shape_str_list]
            actual_shapes.append("[" + ",".join(shape_val) + "]")
        return actual_shapes

    def cal_estimate_l2c(subgraphs):
        """
        It estimate footprint for each subgraph
        For mulit core, all cores share ONE L2 cache counter
        Unable to determine which core 'hit' or 'miss' request
        """
        # stop time stamp,  core 0  core 1, core 2
        # 100:              [0,     100,    20    ]
        # 200:              [0,     100,    50    ]
        # 300:              [20,    100,    0     ]
        footprint_by_timestamp = {}

        # fill zero
        for sg in subgraphs:
            for info in sg["nodes"]:
                cycles = info["Cycles"]
                if cycles <= 0:
                    continue
                footprint_by_timestamp[info["start"]] = [0] * (max_core_ids + 1)
                footprint_by_timestamp[info["end"]] = [0] * (max_core_ids + 1)

        # sort keys
        timestamps = list(footprint_by_timestamp.keys())
        timestamps.sort()

        # split footprint each core by timestamp
        for sg in subgraphs:
            for info in sg["nodes"]:
                footprint_constant = info["Constant Read Bytes"]
                footprint_constant_rest = footprint_constant
                if footprint_constant <= 0:
                    continue
                core_id = info["core_id"]
                cycles = info["Cycles"]
                start = info["start"]
                end = info["end"]
                for timestamp in timestamps:
                    # Linear distribute footpint by time
                    if timestamp >= end:
                        footprint_this_slice = footprint_constant_rest
                    elif timestamp <= start:
                        footprint_this_slice = 0
                    else:
                        footprint_this_slice = int(
                            footprint_constant * (timestamp - start) / cycles)
                        start = timestamp

                    if footprint_this_slice > 0:
                        footprint_by_timestamp[timestamp][core_id] = footprint_this_slice
                        footprint_constant_rest -= footprint_this_slice

                    # stop when no footprint left
                    if footprint_constant_rest <= 0:
                        break

        # for debug
        # for timestamp in timestamps:
        #     print(f"{timestamp}:{footprint_by_timestamp[timestamp]}")

        for sg in subgraphs:
            for info in sg["nodes"]:
                cache_miss_size = info.get("L2 Cache Miss", 0) * 256

                # TODO: if all cache hit?
                if cache_miss_size <= 0:
                    info["L2 Cache Reduced Footprint"] = 0
                    info["AIPU DDR Read Bytes"] = info["AIPU Read Bytes"]
                    continue

                # start
                # 100: [0,  10,  0]
                # 200: [0,  10, 20]
                # 300: [10, 10,  0]
                # stop
                # ratio of core 1 is 30 / 60 =  0.5
                A = 0
                B = 0
                core_id = info["core_id"]
                start = info["start"]
                end = info["end"]
                cycles = end - start
                for timestamp in timestamps:
                    if start < timestamp and timestamp <= end:
                        A += footprint_by_timestamp[timestamp][core_id]
                        B += sum(footprint_by_timestamp[timestamp])
                        # print(f'{start}:{end} {A}:{B} {core_id} {footprint_by_timestamp[timestamp]}')

                # TODO: why
                if B == 0:
                    WARN(
                        f"{start} - {end} L2 Cache Missed but not OP reading constant")
                    continue

                ratio = A / B

                cache_miss_size_this_core = int(cache_miss_size * ratio)

                info["L2 Cache Miss Footprint"] = cache_miss_size_this_core

                if not info.get("L2 Cache Work On Featuremap", False):
                    # normal
                    info["L2 Cache Reduced Footprint"] = info["Constant Read Bytes"] - \
                        cache_miss_size_this_core
                else:
                    # matmul
                    info["L2 Cache Reduced Footprint"] = 0

                # add warning log
                if ratio != 1:
                    info["L2 Cache Reduced Footprint Estimate Ratio"] = f'{ratio} of {cache_miss_size_this_core}'

                info["AIPU DDR Read Bytes"] = info["AIPU Read Bytes"] - \
                    info["L2 Cache Reduced Footprint"]

    mac_num = Target().ff_mac_num
    is_x3 = (Target().isa >= ISAType.ZYv6)

    missing_constant_read_warning_printed = False
    matmul_warning_printed = False

    for subgrahp_nodes in perf_actual_nodes:
        infos = []
        subgraph_infos = dict()
        core_id = subgrahp_nodes["Core ID"]
        subgraph_infos["Core ID"] = core_id
        subgraph_infos["Level"] = subgrahp_nodes["Level"]
        for node in subgrahp_nodes["nodes"]:
            name = node["layer node name"]
            nodes_num = nodes_num + 1
            info = dict()
            info["perf_node_id"] = node["perf_node_id"]
            info["start"] = node["actual_start_tick"]
            info["end"] = node["actual_end_tick"]
            if "pre_idle" in node:
                info["Pre Idle"] = node["pre_idle"]
            for tec_perf_name in tec_event_names:
                info[tec_perf_name] = node[tec_perf_name]
            end = max(info["end"], end)
            start = min(info["start"], start)
            cycles = info["end"] - info["start"]
            if cycles == 0:
                cycles = -1
            info["Cycles"] = cycles
            node_tecs = node["runtime_info"]
            periods = []
            mtp0_busy_cycle = 0
            mtp1_busy_cycle = 0
            itp_busy_cycle, itp0_busy_cycle, itp1_busy_cycle = 0, 0, 0
            ptp_busy_cycle = 0
            com0 = 0
            com1 = 0
            for tec in node_tecs:
                aiff_e = tec["aiff"]
                for aiff in aiff_e:
                    mtp0_busy_cycle += aiff["mtp0_busy_cycle"]
                    mtp1_busy_cycle += aiff["mtp1_busy_cycle"]
                    itp_busy_cycle += aiff.get("itp_busy_cycle", 0)
                    itp0_busy_cycle += aiff.get("itp0_busy_cycle", 0)
                    itp1_busy_cycle += aiff.get("itp1_busy_cycle", 0)
                    ptp_busy_cycle += aiff["ptp_busy_cycle"]
                    com0 += aiff["com0"] if "com0" in aiff else 0
                    com1 += aiff["com1"] if "com1" in aiff else 0
                    periods.append(aiff)

            aiff_sum_busy_cycle = find_total_cycles(periods)
            info["AIFF Busy Cycles"] = aiff_sum_busy_cycle
            info["AIFF Utilization"] = (aiff_sum_busy_cycle / cycles)
            total_aiff_cycles += aiff_sum_busy_cycle

            periods = []
            for tec in node_tecs:
                dma_e = tec["dma"]
                for dma in dma_e:
                    periods.append(dma)

            dma_cycle = find_total_cycles(periods)
            dma_read_bytes = 0
            dma_write_bytes = 0
            for dma in periods:
                dma_read_bytes += dma["read_bytes"]
                dma_write_bytes += dma["write_bytes"]
            info["DMA Busy Cycles"] = dma_cycle
            info["DMA Read Bytes"] = dma_read_bytes
            info["DMA Write Bytes"] = dma_write_bytes
            if "asid_perf_items" in aux_info:
                asid = aux_info["asid_perf_items"]
                for item in asid:
                    info[f"{item.upper()} Read Bytes"] = node[f"{item}_read_bytes"]
                    info[f"{item.upper()} Write Bytes"] = node[f"{item}_write_bytes"]
                    total_asid_items[f"{item.upper()} Read Bytes"] += node[f"{item}_read_bytes"]
                    total_asid_items[f"{item.upper()} Write Bytes"] += node[f"{item}_write_bytes"]
                if "asid_0" in asid and "asid_123" in asid:
                    fm_read_item = "ASID_0 Read Bytes"
                    fm_write_item = "ASID_0 Write Bytes"
                    wt_read_item = "ASID_123 Read Bytes"
                    wt_write_item = "ASID_123 Write Bytes"
                    info["Constant Read Bytes"] = info[wt_read_item]
                    info["Feature Map Read Bytes"] = info.pop(fm_read_item)
                    info["Feature Map Write Bytes"] = info.pop(fm_write_item)

            total_dma_cycles += dma_cycle
            info["stamp"] = node_tecs
            info["core_id"] = core_id
            info["Task ID"] = node.get("task_id", "")
            subgraph_infos["Task ID"] = node.get("task_id", "")
            info["Level"] = subgrahp_nodes["Level"]
            info["Layer Name"] = name
            plugin_name = node["plugin"]
            if len(plugin_name) == 0:
                plugin_name = node["op_type"]
            inputs = node["inputs"]
            if inputs:
                inputs = [graph_json["tensors"][name] for name in inputs]
            else:
                inputs = []
            outputs = node["outputs"]
            if outputs:
                outputs = [graph_json["tensors"][name] for name in outputs]
            else:
                outputs = []
            input_shapes = [inp["shape"] for inp in inputs]
            input_mem_info = [inp.get("memory", "") for inp in inputs]
            output_shapes = [out["shape"] for out in outputs]
            output_mem_info = [out.get("memory", "") for out in outputs]
            if vars_map:
                input_shapes = eval_shapes(input_shapes, vars_map)
                output_shapes = eval_shapes(output_shapes, vars_map)
            info["Input Shapes"] = ",\n".join(input_shapes)
            info["Input Memory"] = ",\n".join(input_mem_info)
            info["Output Shapes"] = ",\n".join(output_shapes)
            info["Output Memory"] = ",\n".join(output_mem_info)
            info["Plugin Name"] = plugin_name
            read_bytes = node["read_data_bytes"]
            info["AIPU Read Bytes"] = read_bytes
            write_bytes = node["write_data_bytes"]
            info["AIPU Write Bytes"] = write_bytes
            total_writes += write_bytes
            total_dma_reads += dma_read_bytes
            total_dma_writes += dma_write_bytes

            if "Constant Read Bytes" not in info:
                # if not perf, read constant bytes from constant size
                if "weight_reload" in node:
                    info["Constant Read Bytes"] = int(node["weight_reload"])
                    if not missing_constant_read_warning_printed:
                        WARN(
                            f"Node ({name}) Using 'weight_reload' as 'Constant Read Bytes'")
                        missing_constant_read_warning_printed = True

                elif "constant_size" in node:
                    info["Constant Read Bytes"] = int(node["constant_size"])
                    if not missing_constant_read_warning_printed:
                        WARN(
                            f"Node ({name}) Using 'constant_size' as 'Constant Read Bytes'")
                        missing_constant_read_warning_printed = True

            if modify_l2c_by_layers:
                if "hwamatmul" == node["plugin"].lower():
                    # assume no reload
                    info["L2 Cache Work On Featuremap"] = True
                    if not matmul_warning_printed:
                        WARN(
                            f"({name}): The second featuremap of matmul would review as wegiht for X3 ")
                        matmul_warning_printed = True

            miss = node.get("Tec0 Count for weight request miss times", None)
            hit = node.get("Tec0 Count for weight request hit times", None)

            if miss != None:
                info["L2 Cache Miss"] = miss

            if hit != None:
                info["L2 Cache Hits"] = hit

            # When user enable 512 & 513 event, user need know L2 hit rate
            if miss != None and hit != None and hit + miss > 0:
                info["L2 Cache Hit Rate"] = hit / (hit + miss)
                # Only statistic ones
                if core_id == 0:
                    total_L2_hit += hit
                    total_L2_miss += miss

            total_constant_reads += info["Constant Read Bytes"] if "Constant Read Bytes" in info else 0
            info["MTP0 Busy Cycles"] = mtp0_busy_cycle
            info["MTP0 Utilization"] = mtp0_busy_cycle / \
                aiff_num_per_core / cycles
            info["MTP1 Busy Cycles"] = mtp1_busy_cycle
            info["MTP1 Utilization"] = mtp1_busy_cycle / \
                aiff_num_per_core / cycles
            if is_x3:
                info["ITP0 Busy Cycles"] = itp0_busy_cycle
                info["ITP0 Utilization"] = itp0_busy_cycle / \
                    aiff_num_per_core / cycles
                info["ITP1 Busy Cycles"] = itp1_busy_cycle
                info["ITP1 Utilization"] = itp1_busy_cycle / \
                    aiff_num_per_core / cycles
                # 2 aiff and 2 itp calc average
                info["ITP Busy Cycles"] = itp0_busy_cycle + itp1_busy_cycle
                info["ITP Average Busy Cycles"] = info["ITP Busy Cycles"] / \
                    2 / aiff_num_per_core
                info["ITP Utilization"] = info["ITP Average Busy Cycles"] / cycles
                # 2 aiff and 2 mpt calc average
                mtp_busy_cycle = info["MTP0 Busy Cycles"] + \
                    info["MTP1 Busy Cycles"]
                info["MTP Average Busy Cycles"] = mtp_busy_cycle / \
                    aiff_num_per_core / 2
                info["MTP Utilization"] = mtp_busy_cycle / \
                    aiff_num_per_core / cycles / 2
                info["PTP Busy Cycles"] = ptp_busy_cycle
                info["PTP Average Busy Cycles"] = info["PTP Busy Cycles"] / \
                    aiff_num_per_core
                info["PTP Utilization"] = info["PTP Average Busy Cycles"] / cycles
            else:
                info["ITP Busy Cycles"] = itp_busy_cycle
                info["ITP Utilization"] = itp_busy_cycle / \
                    aiff_num_per_core / cycles
                info["PTP Busy Cycles"] = ptp_busy_cycle
                info["PTP Utilization"] = ptp_busy_cycle / \
                    aiff_num_per_core / cycles

            ops = node["ops_count"] if "ops_count" in node else 0
            if isinstance(ops, str):
                ops = int(ops) if ops.isdigit(
                ) else eval_expression(ops, vars_map)

            # CP-18272: need reduce mac_num when inp and weight is 16bit
            # first input type of MAC
            mac_A_type = 'int8'
            if inputs:
                mac_A_type = inputs[0]['dtype']

            # second input type of MAC
            mac_B_type = 'int8'
            weight_dtype = get_mac_weight_dtype(node)
            if weight_dtype:
                mac_B_type = weight_dtype

            # print(mac_A_type, mac_B_type)
            # unigned int same as signed int
            if 'float16' in mac_A_type or 'float16' in mac_B_type:
                # As long as one is fp16/bfp16, it is considered fp16/bfp16
                mac_dtype = 'f16'
                mac_scale = 0.5
            elif 'float8' in mac_A_type or 'float8' in mac_B_type:
                # As long as one is fp8, it is considered fp8
                mac_dtype = 'f8'
                mac_scale = 1
            elif 'int8' in mac_A_type and 'int16' in mac_B_type:
                mac_dtype = "i8 x i16"
                mac_scale = 0.5
            elif 'int16' in mac_A_type and 'int8' in mac_B_type:
                mac_dtype = "i8 x i16"
                mac_scale = 0.5
            elif 'int16' in mac_A_type and 'int16' in mac_B_type:
                mac_dtype = "i16"
                mac_scale = 0.25
            else:
                # default: int8 or less
                mac_dtype = "i8"
                mac_scale = 1
            if 'float4' in mac_B_type:
                mac_dtype = 'wf4'
                mac_scale *= 2

            ideal_cycles = ops / (mac_num * mac_scale * 2)

            # statistic
            ops_groupby_dtype[mac_dtype] += ops
            total_ops = total_ops + ops
            total_ideal_cycles = total_ideal_cycles + ideal_cycles

            info["Algorithm Ops(Million)"] = (ops / 1000000)
            info["Cycles Ratio"] = cycles
            # As to fast perf, profiler cannot catch mtp info.
            # Assume all hwa plugin using mtp in this sitation.
            mac_utils = 0
            if mtp0_busy_cycle + mtp1_busy_cycle > 0 or (is_fast_perf and ("hwa" in node["plugin"].lower() or "btconv" in node["plugin"].lower())):
                mac_utils = ideal_cycles / cycles

            info["Data Type"] = mac_dtype
            info["MAC Scale"] = mac_scale
            info["MAC Utilization"] = mac_utils

            # kernelsize
            if verbose > 0:
                local_kernel = []
                if 'internal_nodes' in node:
                    for n in node['internal_nodes']:
                        params = n.get('params', None)
                        if params and 'kernel_y' in params and 'dilation_y' in params and 'stride_y' in params:
                            local_kernel.append('k%sd%ss%s' %
                                                (params['kernel_y'], params['dilation_y'], params['stride_y']))
                info["Kernel Size"] = ','.join(local_kernel)

            if len(aiff_event_names) >= 1:
                info[aiff_event_names[0]] = com0
            if len(aiff_event_names) >= 2:
                info[aiff_event_names[1]] = com1
            infos.append(info)
        subgraph_infos["nodes"] = infos
        subgraphs.append(subgraph_infos)

    # update Cycles Ratio
    duration = end - start
    if duration == 0:
        duration = -1
    for sg in subgraphs:
        for info in sg["nodes"]:
            ratio = info["Cycles Ratio"] / duration
            info["Cycles Ratio"] = ratio

    if modify_l2c_by_layers:
        cal_estimate_l2c(subgraphs)

    for sg in subgraphs:
        for info in sg["nodes"]:
            total_reads += info["AIPU Read Bytes"]
            if info.get("AIPU DDR Read Bytes", 0) > 0:
                read_bytes = info["AIPU DDR Read Bytes"]
            else:
                read_bytes = info["AIPU Read Bytes"]

            total_ddr_read += read_bytes
            write_bytes = info["AIPU Write Bytes"]

            cycles = info["Cycles"]
            info["BW_RW(GB/s)"] = ((read_bytes + write_bytes)
                                   * freq / (cycles * 1000))
            info["BW_R(GB/s)"] = ((read_bytes) * freq / (cycles * 1000))
            info["BW_W(GB/s)"] = ((write_bytes) * freq / (cycles * 1000))

    perf_data["subgraphs"] = subgraphs
    perf_data["Total Cycles"] = duration
    cost_time = (duration / (1000 * freq))
    perf_data[f"Total Time(ms@{freq}MHz)"] = cost_time
    perf_data[f"FPS@{freq}MHz"] = 1000 / cost_time

    perf_data["Total DMA Busy Cycles"] = total_dma_cycles
    perf_data["Total AIFF Busy Cycles"] = total_aiff_cycles

    perf_data["AIPU Read Bytes"] = total_reads
    perf_data["AIPU Write Bytes"] = total_writes
    perf_data["Constant Read Bytes"] = total_constant_reads

    if modify_l2c_by_layers:
        perf_data["AIPU DDR Read Bytes"] = total_ddr_read
        perf_data["AIPU L2 Cached Bytes"] = total_reads - total_ddr_read

    if total_L2_hit + total_L2_miss > 0:
        perf_data["L2 Cache hit"] = total_L2_hit
        perf_data["L2 Cache miss"] = total_L2_miss
        perf_data["L2 Cache Hit Rate"] = total_L2_hit / \
            (total_L2_hit + total_L2_miss)

    perf_data["DMA Read Bytes"] = total_dma_reads
    perf_data["DMA Write Bytes"] = total_dma_writes
    perf_data["Subgraphs Num"] = len(subgraphs)
    perf_data["Layers Num"] = nodes_num
    perf_data["Total Ops(Mops)"] = (total_ops / 1000000)

    ops_dtypes_str = []
    for key, val in ops_groupby_dtype.items():
        if val > 0:
            ops_dtypes_str.append(f"{key}: {val / total_ops * 100:.2f}%")

    perf_data["Ops Data Type"] = " | ".join(ops_dtypes_str)
    perf_data["BW_RW(GB/s)"] = ((total_ddr_read +
                                 total_writes) / 1e9) / (cost_time / 1e3)
    perf_data["BW_R(GB/s)"] = ((total_ddr_read) / 1e9) / (cost_time / 1e3)
    perf_data["BW_W(GB/s)"] = ((total_writes) / 1e9) / (cost_time / 1e3)

    avg_mac_utils = total_ideal_cycles / perf_data["Total Cycles"] / core_num
    perf_data["Ideal Cycles"] = total_ideal_cycles
    perf_data["Average MAC Utilization"] = avg_mac_utils

    mtp_busy_cycle = 0
    for sg in subgraphs:
        for info in sg["nodes"]:
            mtp_busy_cycle += info["MTP0 Busy Cycles"] + \
                info["MTP1 Busy Cycles"]
    # mac_num needs divide 2 as mac_num has already counts 2 MTP
    perf_data["Average MTP Utilization(when working)"] = total_ops / (mtp_busy_cycle *
                                                                      (mac_num / 2 / aiff_num_per_core) * 2) if mtp_busy_cycle != 0 else 0

    perf_data["tec_num"] = tec_num
    perf_data["core_num"] = core_num

    perf_data["Target"] = graph_json["platform target"]
    # items: the detail table header name
    perf_data["items"] = ["Layer Name", "Core ID", "Level", "Task ID", "Plugin Name", "Cycles", "AIFF Busy Cycles",
                          "DMA Busy Cycles", "AIFF Utilization", "AIPU Read Bytes", "AIPU Write Bytes", "DMA Read Bytes", "DMA Write Bytes"]
    perf_data["items"] += ["Constant Read Bytes"]
    asid_read_write_heads = []
    if "asid_perf_items" in aux_info:
        asid = aux_info["asid_perf_items"]
        items = []
        if "asid_0" in asid and "asid_123" in asid:
            # default situation
            # rename asid_0 to feature map
            fm_read_item = "ASID_0 Read Bytes"
            fm_write_item = "ASID_0 Write Bytes"
            wt_read_item = "ASID_123 Read Bytes"
            wt_write_item = "ASID_123 Write Bytes"
            items += ["Feature Map Read Bytes", "Feature Map Write Bytes"]
            asid_read_write_heads.extend(
                ["Feature Map Read Bytes", "Feature Map Write Bytes"])
            perf_data["Feature Map Read Bytes"] = total_asid_items[fm_read_item]
            perf_data["Feature Map Write Bytes"] = total_asid_items[fm_write_item]
        else:
            for item in asid:
                read_item = f"{item.upper()} Read Bytes"
                write_item = f"{item.upper()} Write Bytes"
                items += [read_item, write_item]
                asid_read_write_heads.extend([read_item, write_item])
                perf_data[read_item] = total_asid_items[read_item]
                perf_data[write_item] = total_asid_items[write_item]
        perf_data["items"] += items

    if verbose > 0:
        perf_data["items"] += ["L2 Cache Reduced Footprint"]

    if is_x3:
        perf_data["items"] += ["MTP Average Busy Cycles", "MTP Utilization"]
        perf_data["items"] += ["ITP Average Busy Cycles",
                               "ITP Utilization", "PTP Average Busy Cycles", "PTP Utilization"]
    else:
        perf_data["items"] += ["MTP0 Busy Cycles",
                               "MTP0 Utilization", "MTP1 Busy Cycles", "MTP1 Utilization"]
        perf_data["items"] += ["ITP Busy Cycles",
                               "ITP Utilization", "PTP Busy Cycles", "PTP Utilization"]

    if verbose > 0:
        perf_data["items"] += ["Kernel Size"]

    perf_data["items"] += ["Input Shapes", "Output Shapes",
                           "Data Type", "Algorithm Ops(Million)", "MAC Utilization"]
    aux_info = graph_json["profile"]["auxiliary info"]

    if total_L2_hit + total_L2_miss > 0:
        perf_data["items"] += ["L2 Cache Hit Rate"]

    # TODO: auto remove no exist item
    removed_items = list()
    if "disable_dma_perf" in aux_info and aux_info["disable_dma_perf"]:
        removed_items = ["DMA Busy Cycles",
                         "DMA Read Bytes", "DMA Write Bytes"]
    if "enable_fast_perf" in aux_info and aux_info["enable_fast_perf"]:
        removed_items = [
            "AIFF Busy Cycles", "DMA Busy Cycles", "AIFF Utilization", "DMA Read Bytes",
            "DMA Write Bytes", "MTP0 Busy Cycles", "MTP0 Utilization", "MTP1 Busy Cycles",
            "MTP1 Utilization", "ITP Busy Cycles", "ITP Utilization", "ITP0 Busy Cycles",
            "ITP0 Utilization", "ITP1 Busy Cycles", "ITP1 Utilization",
            "PTP Busy Cycles", "PTP Utilization", "MTP Utilization", "ITP Average Busy Cycles",
            "PTP Average Busy Cycles"
        ]
    for item_n in removed_items:
        if item_n in perf_data["items"]:
            perf_data["items"].remove(item_n)

    if len(tec_event_names) != 0:
        if is_x3 and aux_info.get("noc_events", []) and len(aux_info["noc_events"]) == 1 and len(aux_info["tec_events"]) == 0:
            pass
        else:
            perf_data["items"] += tec_event_names

    if len(aiff_event_names) != 0:
        perf_data["items"] += aiff_event_names

    perf_data["summaries"] = ["Target", "Total Cycles", f"Total Time(ms@{freq}MHz)", f"FPS@{freq}MHz", "Total Ops(Mops)",
                              "Ops Data Type",
                              "Total AIFF Busy Cycles",
                              "Total DMA Busy Cycles", "AIPU Read Bytes", "AIPU Write Bytes", "DMA Read Bytes", "DMA Write Bytes", "Constant Read Bytes", "Subgraphs Num", "Layers Num"]
    if modify_l2c_by_layers:
        perf_data["summaries"] += ["AIPU DDR Read Bytes",
                                   "AIPU L2 Cached Bytes"]

    if total_L2_hit + total_L2_miss > 0:
        perf_data["summaries"] += ["L2 Cache Hit Rate"]

    perf_data["summaries"] += ["Average MAC Utilization",
                               "Average MTP Utilization(when working)"]
    perf_data["summaries"] += asid_read_write_heads
    remove_summaries = list()
    if "disable_dma_perf" in aux_info and aux_info["disable_dma_perf"]:
        remove_summaries = ["DMA Read Bytes",
                            "DMA Write Bytes", "Total DMA Busy Cycles"]
    if "enable_fast_perf" in aux_info and aux_info["enable_fast_perf"]:
        remove_summaries = [
            "Total AIFF Busy Cycles", "DMA Read Bytes", "DMA Write Bytes", "Total DMA Busy Cycles",
            "Average MTP Utilization(when working)"
        ]
    for item_n in remove_summaries:
        perf_data["summaries"].remove(item_n)
        perf_data.pop(item_n)

    if "emu" in graph_json["profile"]:
        emu_info = graph_json["profile"]["emu"]
        for info in emu_info:
            perf_data[info] = emu_info[info]
            perf_data["summaries"].append(info)
        perf_data["emu"] = emu_info
    return perf_data
