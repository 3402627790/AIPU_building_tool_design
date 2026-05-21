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
import numpy as np


class gen_charts():
    def __init__(self, raw_datas, freq_GHz):

        self.exlcude_optypes = []
        self.freq_factor = freq_GHz * 1000 * 1000
        self.raw_datas = raw_datas

        self.start_cycles = None    # update at get_nodes_data()
        self.end_cycles = None      # update at get_nodes_data()

        self.core_num = int(self.raw_datas["core_num"])

        self.all_core_nodes = self.get_nodes_data()

    def get_optype(self, node):
        node_name = node["Layer Name"]
        node_type = node["Plugin Name"]

        if node_type:
            return node_type

        if node_name:
            return node_name

    def get_op_infos(self, core_id):
        nodes = self.all_core_nodes[core_id]
        if len(nodes) == 0:
            return None, None, None

        all_layer_names = []
        all_layer_optypes = []
        all_cycles = []
        all_time_costs = []  # ms
        for n in nodes:
            all_layer_names.append(n["Layer Name"])

            optype = self.get_optype(n)
            all_layer_optypes.append(optype)

            all_cycles.append(n["Cycles"])
            all_time_costs.append(all_cycles[-1] / self.freq_factor)

        used_op_types = []
        for op in list(set(all_layer_optypes)):
            if op not in self.exlcude_optypes:
                used_op_types.append(op)

        idxs_for_each_type = [None]*len(used_op_types)
        time_costs_for_each_type = [None]*len(used_op_types)

        for i, op in enumerate(all_layer_optypes):
            if op not in used_op_types:
                continue

            idx = used_op_types.index(op)

            if idxs_for_each_type[idx] == None:
                idxs_for_each_type[idx] = []
                time_costs_for_each_type[idx] = []

            idxs_for_each_type[idx].append(i)
            time_costs_for_each_type[idx].append(all_time_costs[i])

        # show part
        candidate = []
        for i in range(len(used_op_types)):
            op_type_name = used_op_types[i]
            time_cost = sum(time_costs_for_each_type[i])
            layer_nums = len(idxs_for_each_type[i])
            candidate.append(op_type_name + ",%d layers,%.3fms" %
                             (layer_nums, time_cost))

        votes = [sum(ts) for ts in time_costs_for_each_type]
        norm_votes = [v / sum(votes) for v in votes]

        max_pos = norm_votes.index(max(norm_votes))
        global selected_optype_pos
        selected_optype_pos = max_pos

        explode = [0]*len(candidate)  # (0.1, 0, 0)
        explode[max_pos] = 0.1

        return votes, candidate, explode

    def simplify_plugin_optype(self, optype):
        if len(optype.split("_")) > 1:
            return optype.split("_")[0]
        else:
            return optype

    def get_x_infos(self, core_id):
        nodes = self.all_core_nodes[core_id]
        if len(nodes) == 0:
            return []

        all_layer_names = []
        all_layer_optypes = []

        all_time_costs = []  # ms

        all_start_cycles = []
        all_end_cycles = []
        all_layer_cycles = []
        for n in nodes:
            all_layer_names.append(n["Layer Name"])

            optype = self.get_optype(n)
            all_layer_optypes.append(self.simplify_plugin_optype(optype))

            all_start_cycles.append(n["start"])
            all_end_cycles.append(n["end"])
            all_layer_cycles.append(n["Cycles"])
            all_time_costs.append(all_layer_cycles[-1] / self.freq_factor)

        total_time = (self.end_cycles - self.start_cycles) / self.freq_factor
        sync_frame_time = total_time - sum(all_time_costs)

        used_op_types = []
        for op in list(set(all_layer_optypes)):
            if op not in self.exlcude_optypes:
                used_op_types.append(op)

        idxs_for_each_type = [None]*len(used_op_types)
        time_costs_for_each_type = [None]*len(used_op_types)

        for i, op in enumerate(all_layer_optypes):
            if op not in used_op_types:
                continue

            idx = used_op_types.index(op)

            if idxs_for_each_type[idx] == None:
                idxs_for_each_type[idx] = []
                time_costs_for_each_type[idx] = []

            idxs_for_each_type[idx].append(i)
            time_costs_for_each_type[idx].append(all_time_costs[i])

        votes = [sum(ts) for ts in time_costs_for_each_type]
        norm_votes = [v / total_time for v in votes]

        candidate = []
        time_total = 0.0
        percent_total = 0.0
        for i in range(len(used_op_types)):
            op_type_name = used_op_types[i]
            time_cost = sum(time_costs_for_each_type[i])
            # time_total  += time_cost
            percent = norm_votes[i]
            # percent_total += percent
            # layer_nums   = len(idxs_for_each_type[i])
            table_item = [op_type_name, time_cost, percent]
            candidate.append(table_item)
        # candidate.append(["idel_time", "%.3fms"%(sync_frame_time), "%.2f%%"%(sync_frame_time/total_time)])

        candidate.sort(key=lambda x: float(x[1]), reverse=True)
        candidate.append(["Idle", sync_frame_time, sync_frame_time/total_time])
        # table_item_total = ["Sub_total", time_total, percent_total]
        # candidate.append(table_item_total)
        return candidate

    def get_nodes_data(self):

        # via PAC-47
        # No:  per core start & end
        # Yes: all cores start & end
        # find min & max in loop
        self.start_cycles = self.raw_datas["Total Cycles"]
        self.end_cycles = 0

        all_core_nodes = [[] for i in range(self.core_num)]
        for d in self.raw_datas["subgraphs"]:
            for n in d["nodes"]:
                core_id = n["core_id"]
                while len(all_core_nodes) <= core_id:
                    all_core_nodes.append([])
                all_core_nodes[core_id].append(n)
                self.start_cycles = min(n["start"], self.start_cycles)
                self.end_cycles = max(n["end"], self.end_cycles)

        all_nodes_num = 0
        for core_nodes in all_core_nodes:
            all_nodes_num += len(core_nodes)
        assert all_nodes_num == self.raw_datas["Layers Num"], "nodes num != layer nums."

        return all_core_nodes

    def get_tabel_data(self, verbose=1):
        # verbose=1 for profiler, verbose=2 for pyecharts.
        all_core_tables = []

        for core_id in range(self.core_num):
            tmp_list = self.get_x_infos(core_id)
            if verbose == 1:
                for i in range(len(tmp_list)):
                    if "ms" in tmp_list[i][1]:
                        tmp_list[i][1] = tmp_list[i][1].replace("ms", "")
                    if "%" in tmp_list[i][2]:
                        tmp_list[i][2] = tmp_list[i][2].replace("%", "")
            all_core_tables.append(tmp_list)

        # padding
        optye_len = max([len(x) for x in all_core_tables])
        for core_id in range(self.core_num):
            padding_len = optye_len - len(all_core_tables[core_id])
            for i in range(padding_len):
                all_core_tables[core_id].append(["", "", ""])

        rows = []
        for r in range(optye_len):
            tmp = []
            for i in range(self.core_num):
                tmp += all_core_tables[i][r]
            rows.append(tmp)

        headers = ["Plugin_0", "Time(ms)_0", "Ratio(%)_0", "Plugin_1",
                   "Time(ms)_1", "Ratio(%)_1", "Plugin_2", "Time(ms)_2", "Ratio(%)_2", ]

        return headers, rows

    def get_plugin_datas(self, verbose=1):
        all_plugin_info = {}
        # e.g. all_plugin_info = {
        #   'TransposePlugin':
        #       {
        #           "total": {'time': 12.827706399999997, 'ratio': 0.24897386957225343},
        #           'core_0': {'time': 12.774575199999997, 'ratio': 0.24791731276099133},
        #           'core_1': {'time': 0.053131199999999996, 'ratio': 0.0010565568112620995}
        #       }
        # }

        for core_id in range(self.core_num):
            tmp_list = self.get_x_infos(core_id)
            # per-plugin
            for plugin_name, time, ratio in tmp_list:
                # new row
                if plugin_name not in all_plugin_info:
                    all_plugin_info[plugin_name] = {
                        "total": {
                            "time": 0,
                            "ratio": 0,
                        }
                    }
                # add row ele by core
                all_plugin_info[plugin_name][f'core_{core_id}'] = {
                    "time": time, "ratio": ratio}
                # sum
                all_plugin_info[plugin_name]["total"]["time"] += time
                all_plugin_info[plugin_name]["total"]["ratio"] += ratio

        return all_plugin_info

    def get_merged_pref(self, verbose=1):
        # layername, plugins, cycle, read, readrate, write, writerate, read+write, (rw)/totalrw
        total_read, total_write, max_level = 0, 0, 0
        op_list, op_idx = {}, 0
        for sg in self.raw_datas['subgraphs']:
            for sn in sg['nodes']:
                total_read += sn['AIPU Read Bytes']
                total_write += sn['AIPU Write Bytes']
                local_name = sn['Layer Name'].split('/')[0]
                if local_name not in op_list.keys():
                    op_list[local_name] = [
                        op_idx, local_name, [], 0, 0, 0, 0, 0, 0, 0]
                    op_idx += 1

        # collect cycle across cores
        for optype in op_list.keys():
            max_cycle = 0
            for i in range(self.core_num):
                local_cycle = 0
                for sg in self.raw_datas['subgraphs']:
                    if sg['Core ID'] == i:
                        for sn in sg['nodes']:
                            local_name = sn['Layer Name'].split('/')[0]
                            if optype == local_name:
                                local_cycle += sn['Cycles']
                if local_cycle > max_cycle:
                    max_cycle = local_cycle
            local_idx = op_list[optype][0]
            op_list.update(
                {optype: [local_idx, optype, [], max_cycle, 0, 0, 0, 0, 0, 0]})

        # collect footprint together
        for sg in self.raw_datas['subgraphs']:
            for sn in sg['nodes']:
                local_name = sn['Layer Name'].split('/')[0]
                if local_name in op_list.keys():
                    local_read = op_list[local_name][4] + sn['AIPU Read Bytes']
                    local_write = op_list[local_name][6] + \
                        sn['AIPU Write Bytes']

                    # avoid duplicate plugin name
                    local_oplist = op_list[local_name][2]
                    # avoid duplicate plugin name
                    local_cycle = op_list[local_name][3]
                    if sn['Plugin Name'] not in local_oplist:
                        local_oplist.append(sn['Plugin Name'])

                    local_idx = op_list[local_name][0]
                    local_op = {local_name: [local_idx, local_name, local_oplist,
                                             local_cycle,
                                             local_read, (local_read /
                                                          total_read) if total_read != 0 else 0,
                                             local_write, (local_write /
                                                           total_write) if total_write != 0 else 0,
                                             local_read + local_write]}
                    op_list.update(local_op)

        op_list2 = list(op_list.values())
        layer_pf2 = sorted(op_list2, key=lambda rw: rw[0], reverse=False)
        cleand_pf = []
        for itm in layer_pf2:
            cleand_pf.append(itm[1:])

        # cleand_pf.append(['Total R+W', '', '', '', '', '', '', total_read + total_write])
        headers = ['LayerName', 'PluginName', 'Cycle', 'Read',
                   'Read Ratio', 'Write', 'Write Ratio', 'Read+Write']
        return headers, cleand_pf


def export_xlsx(report_name, perf_data, freq, verbose):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment
    from openpyxl.utils.dataframe import dataframe_to_rows
    from openpyxl.utils import get_column_letter
    import pandas as pd

    # format precision

    def format_col(wp, format, col_list):
        for col in wp.columns:
            if col[0].value not in col_list:
                continue
            for cell in col:
                cell.number_format = format

    # format width
    def format_width(wp, ignore_title=False):
        for col in wp.columns:
            max_len = 0
            col_name = get_column_letter(col[0].column)
            for idx, cell in enumerate(col):
                if idx == 0 and ignore_title:
                    continue

                if isinstance(cell.value, float):
                    data_len = len('%.4f' % (cell.value)) + 2
                elif isinstance(cell.value, int):
                    data_len = len(str(cell.value))
                    # optimize width caused by thousands format
                    data_len += math.ceil(data_len/3.0)
                elif isinstance(cell.value, str):
                    data_len = len(cell.value.split('\n')[0]) + 2
                else:
                    data_len = len(str(cell.value)) + 2

                if data_len > max_len:
                    max_len = data_len
            # print(col_name, max_len)
            wp.column_dimensions[col_name].width = max_len

    def format_height(wp):
        for row in wp.iter_rows():
            max_lines = 1
            for cell in row:
                if cell.value and "\n" in str(cell.value):
                    lines = str(cell.value).split('\n')
                    max_lines = max(max_lines, len(lines))

            # 11 : default font size of libreoffice
            wp.row_dimensions[row[0].row].height = max_lines * 11 + 2

    report_data = list()
    items = perf_data["items"].copy()
    items.insert(3, "Plugin Name")
    items.insert(5, "Cycles Ratio")
    items.insert(11, "BW_RW(GB/s)")
    items.insert(12, "BW_R(GB/s)")
    items.insert(13, "BW_W(GB/s)")

    items.append("Constant Read Bytes")

    if "AIPU Read Bytes Exclude L2 Cache" in perf_data and verbose > 0:
        # modify_l2c_by_layers
        items.append("L2 Cache Reduced Footprint")
        items.append("BW Exclude L2 Cache(GB/s)")

    items.append("MAC Utilization")

    subgraph_nodes_num = []
    for subgraph in perf_data["subgraphs"]:
        subgraph_nodes_num.append(len(subgraph["nodes"]))
        for node in subgraph["nodes"]:
            data = {
                "Core ID": subgraph["Core ID"],
            }
            for key in node:
                if key in items:
                    data[key] = node[key]

                    # ignore cal_estimate_l2c < 0 in report
                    if key == "L2 Cache Reduced Footprint":
                        data[key] = max(data[key], 0)

            order_data = dict()
            for item in items:
                order_data[item] = data.get(item, "")
            report_data.append(order_data)

    layer_data = pd.DataFrame(report_data)

    summary_data = ["Target", "Total Cycles", "Cycles Ratio", f"Total Time(ms@{freq}MHz)", "Total Ops(Mops)", f"FPS@{freq}MHz", "Total DMA Busy Cycles",
                    "Total AIFF Busy Cycles", "AIPU Read Bytes", "AIPU Write Bytes", "Constant Read Bytes", "DMA Read Bytes", "DMA Write Bytes", "Subgraphs Num", "Layers Num", "BW_RW(GB/s)"]
    summary_data += ["Average MAC Utilization",
                     "Average MTP Utilization(when working)"]
    asid_items = ['GM', 'ASID0', 'ASID1', 'ASID2', 'ASID3', 'Feature Map']
    if "AIPU DDR Read Bytes" in perf_data:
        summary_data.append("AIPU DDR Read Bytes")
    if "AIPU L2 Cached Bytes" in perf_data:
        summary_data.append("AIPU L2 Cached Bytes")
    summary_data.extend([(item + " Read Bytes")
                        for item in asid_items if (item + " Read Bytes") in perf_data])
    summary_data.extend([(item + " Write Bytes")
                        for item in asid_items if (item + " Write Bytes") in perf_data])
    summary = dict()
    for key in perf_data:
        if key in summary_data:
            summary[key] = perf_data[key]
    if "emu" in perf_data:
        for info in perf_data["emu"]:
            summary[info] = perf_data["emu"][info]

    summary_data = pd.DataFrame(summary, index=[0])

    charts = gen_charts(perf_data, freq / 1000)
    core_num = perf_data["core_num"]
    plugin_datas_raw = charts.get_plugin_datas()

    plugin_names = list(plugin_datas_raw.keys())
    if "Idle" in plugin_names:
        # rotate to tail
        plugin_names.remove("Idle")
        plugin_names.append("Idle")

    plugin_datas = {
        "plugin": plugin_names,            # 1st     col: name
    }

    for i in range(core_num):              # 4 ~ 7th col: per core value
        plugin_datas[f"core_{i}_time"] = []
        plugin_datas[f"core_{i}_ratio"] = []

    for plugin in plugin_names:
        d = plugin_datas_raw[plugin]
        for i in range(core_num):
            if f"core_{i}" in d:
                plugin_datas[f"core_{i}_time"].append(d[f"core_{i}"]["time"])
                plugin_datas[f"core_{i}_ratio"].append(d[f"core_{i}"]["ratio"])
            else:   # pad blank
                plugin_datas[f"core_{i}_time"].append("")
                plugin_datas[f"core_{i}_ratio"].append("")

    plugin_datas = pd.DataFrame(plugin_datas)
    # print(plugin_datas)

    # for merged perf
    charts_fp = gen_charts(perf_data, freq / 1000)
    heads_fp, rows_fp = charts_fp.get_merged_pref()
    footprint_data = []
    for row_fp in rows_fp:
        tmp = dict()
        for i in range(len(heads_fp)):
            local_pf = row_fp[i]
            if isinstance(local_pf, list):
                local_pf = ','.join(local_pf)
            tmp[heads_fp[i]] = local_pf
        footprint_data.append(tmp)
    footprint_data = pd.DataFrame(footprint_data)

    wb = Workbook()
    ws = wb.active
    ws.title = "Layer Data"

    for row in dataframe_to_rows(layer_data, index=False, header=True):
        ws.append(row)

    # cur_row = 2
    # for num in subgraph_nodes_num:
    #     ws.merge_cells(start_row=cur_row, end_row=cur_row + num - 1, start_column=2, end_column=2)
    #     ws.merge_cells(start_row=cur_row, end_row=cur_row + num - 1, start_column=3, end_column=3)
    #     cur_row = cur_row + num

    # 0.12345 -> 12.3%
    format_col(ws, '0.00%', ["Cycles Ratio", "AIFF Utilization",
                             "MTP Utilization", "MTP0 Utilization", "MTP1 Utilization",
                             "ITP Utilization", "PTP Utilization", "MAC Utilization",
                             "ITP0 Utilization", "ITP1 Utilization", "L2 Cache Hit Rate"])
    # 0.12345 -> 0.12
    format_col(ws, '0.00', ["BW_RW(GB/s)", "BW_R(GB/s)", "BW_W(GB/s)",
                            "Algorithm Ops(Million)", "BW Exclude L2 Cache(GB/s)"])

    # 12345 -> 12,345
    format_width(ws, True)
    format_height(ws)

    col_names_to_format = [
        "Cycles", "AIPU Read Bytes", "AIPU Write Bytes", "Constant Read Bytes",
        "Feature Map Read Bytes", "Feature Map Write Bytes",
        "L2 Cache Reduced Footprint", "AIFF Busy Cycles",
        "MTP Average Busy Cycles", "ITP Average Busy Cycles", "PTP Average Busy Cycles"
    ]
    format_col(ws, "#,##0", col_names_to_format)

    col_num = len(list(ws.columns))
    last_col = list(ws.columns)[-1]
    col_name = get_column_letter(last_col[0].column)
    col_len = len(last_col)
    ws.auto_filter.ref = f"A1:{col_name}{col_len}"

    wb.create_sheet("Summary Data")
    ws = wb["Summary Data"]
    for row in dataframe_to_rows(summary_data, index=False, header=True):
        ws.append(row)
    format_width(ws)
    format_col(ws, '0.00', [
               f"Total Time(ms@{freq}MHz)", f"FPS@{freq}MHz", "Total Ops(Mops)", "BW_RW(GB/s)"])
    format_col(ws, '0.00%', ["Average MAC Utilization"])
    format_col(ws, '0.00%', ["Average MTP Utilization(when working)"])
    format_col(ws, "#,##0", ["Total AIFF Busy Cycles", "AIPU Read Bytes", "AIPU Write Bytes",
                             "AIPU DDR Read Bytes", "AIPU L2 Cached Bytes", "Constant Read Bytes",
                             "Feature Map Read Bytes", "Feature Map Write Bytes"])

    if verbose > 0:
        wb.create_sheet("Muti-core Plugin Data")
        wmpd = wb["Muti-core Plugin Data"]
        # 1st row: thead
        # | plugin | core 0       | core 1        | core 2     |
        wmpd.cell(1, 1, "Plugin")
        for i in range(core_num):
            wmpd.cell(1, 2 + 2 * i, "Core%d" % (i))
            wmpd.merge_cells(start_row=1, end_row=1,
                             start_column=2 + 2 * i, end_column=3 + 2 * i)
        # 2nd row: thead
        # | xxx    | ratio | time | ratio | time | ratio | time | ratio |
        for i in range(core_num):
            wmpd.cell(2, 2 + i * 2, "Time(ms)")
            wmpd.cell(2, 3 + i * 2, "Ratio(%)")
        wmpd.merge_cells(start_row=1, end_row=2, start_column=1, end_column=1)
        # data: tbody
        # | xxplugin | x | x | x | ...
        for row in dataframe_to_rows(plugin_datas, index=False, header=False):
            wmpd.append(row)

        # format centered
        for i in range(2):
            row = wmpd[i + 1]
            for cell in row:
                cell.alignment = Alignment(
                    horizontal='center', vertical='center')
        for i in range(core_num):
            base_idx = i * 2 + 1
            col_time = wmpd[get_column_letter(base_idx + 1)]
            col_ratio = wmpd[get_column_letter(base_idx + 2)]
            for cell in col_time:
                cell.alignment = Alignment(
                    horizontal='center', vertical='center')
                cell.number_format = '0.00'
            for cell in col_ratio:
                cell.alignment = Alignment(
                    horizontal='center', vertical='center')
                cell.number_format = '0.00%'
        format_width(wmpd)

    if verbose > 0:
        wb.create_sheet("Merged Performance")
        wmp = wb["Merged Performance"]
        for row in dataframe_to_rows(footprint_data, index=False, header=True):
            wmp.append(row)
        format_width(wmp)
        format_col(wmp, '0.00%', ["Read Ratio", "Write Ratio"])
    wb.save(report_name)
