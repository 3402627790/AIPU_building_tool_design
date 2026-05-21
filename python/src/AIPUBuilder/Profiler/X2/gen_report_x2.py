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
import hashlib
import copy
from ...logger import WARN, FATAL, INFO

from .load_perf_data import load_perf_data

from .export_xlsx import export_xlsx
from .export_html import export_html
from .export_json import export_json


def x2_gen_report(graph_json, binary, args):
    graph_name = graph_json["graph_name"]

    freq = args.frequency if args.frequency else graph_json["profile"]["frequency"]
    verbose = int(args.verbose)
    verbose = max(min(verbose, 2), 0)

    perf_data = load_perf_data(graph_json, binary, freq, verbose)

    if args.method in ["csv", "xlsx"]:
        postfix = "xlsx"
        if args.method == "csv":
            WARN("csv format is alias to xlsx format")
        report_name = args.output if args.output else graph_name + f"_perf.{postfix}"
        export_xlsx(report_name, perf_data, freq, verbose)
    elif args.method == "html":
        report_name = args.output if args.output else graph_name + "_perf.html"
        export_html(report_name, perf_data, verbose)
    elif args.method == "json":
        report_name = args.output if args.output else graph_name + "_perf.json"
        ret = export_json(graph_json, perf_data, freq)
        with open(report_name, "w") as f:
            f.write(json.dumps(ret, indent = 2))
    else:
        FATAL(f"[Profiler] not support output file type {args.method}")

    if args.database:
        if args.platform not in ["fgpa", "soc", "emulator", "spm"]:
            FATAL("platform info not provide or incorrect, please recheck")
            return
        freq = args.frequency if args.frequency else graph_json["profile"]["frequency"]
        perf_json = export_json(graph_json, perf_data, None, freq)
        commit_perf_data(perf_json, args.gbversion, ir=graph_json["IR"], label=args.comment, platform=args.platform)


def commit_perf_data(
    json_file,
    gb_version=None,
    lib_version=None,
    compliler_version="",
    label="",
    ir: str="",
    platform="fpga"):

    records = []
    profile_data = json_file
    graph_perf_summary = profile_data["profile_data"]
    new_graph_perf_summary = {}
    if not ir is None:
        import gzip
        import base64
        base64.b64decode
        ir_compress_str = gzip.compress(ir.encode("utf-8"))
        ir_compress_str = base64.b64encode(ir_compress_str).decode()

    new_graph_perf_summary["ir"] = ir_compress_str

    for k in graph_perf_summary:
        new_graph_perf_summary[k.replace(" ", "_").replace("(", "_").replace(")", "_").strip("_").replace("/", "_").split("@")[0]] = graph_perf_summary[k]
    new_graph_perf_summary.pop("subgraphs")
    new_graph_perf_summary["platform"] = platform
    new_graph_perf_summary["graph_name"] = profile_data["graph_name"]
    new_graph_perf_summary["comments"] = label
    new_graph_perf_summary["gb_version"] = gb_version
    new_graph_perf_summary["lib_version"] = lib_version
    new_graph_perf_summary["compiler_version"] = compliler_version

    for sub_graph in profile_data["profile_data"]["subgraphs"]:
        for node in sub_graph["nodes"]:
            name = node["Layer Name"]
            record = {}
            record["perf_data"] = {}
            # for k in ["Cycles", "AIPU Read Bytes", "AIPU Write Bytes", "AIFF Busy Cycles", "DMA Busy Cycles"]:
                # record["perf_data"].append(node[k])
            record["perf_data"] = node
            if "stamp" in record["perf_data"]:
                record["perf_data"].pop("stamp")
            customize_event = {}
            ica_keys = [k for k in record["perf_data"] if k.startswith("Tec")]
            for k in ica_keys:
                customize_event[k] = record["perf_data"].pop(k)
            record["perf_data"]["customize_event"] = customize_event


            for node in profile_data["graph"]["nodes"].values():
                if node["layer node name"] == name:
                    op_type = []
                    params = {}
                    if "internal_nodes" in node:
                        for itn in node["internal_nodes"]:
                            op_type.append(itn["op_type"])
                            if itn["params"]:
                                params.update(itn["params"])

                        op_type = "_".join(op_type)

                        record["op_type"] = op_type
                    else:
                        params = node["params"]
                        record["op_type"] = node["op_type"]
                    record["params"] = params
                    input_names = node["inputs"]
                    output_names = node["outputs"]
                    record["inputs"] = []
                    record["outputs"] = []

                    for i_name in input_names:
                        for t in profile_data["graph"]["tensors"]:
                            if t == i_name:
                                record["inputs"].append(profile_data["graph"]["tensors"][t])

                    for o_name in output_names:
                        for t in profile_data["graph"]["tensors"]:
                            if t == o_name:
                                record["outputs"].append(profile_data["graph"]["tensors"][t])
                    break
            records.append(record)
    node_perfs = []
    dtype_dict = {
        "int8": "b",
        "uint8": "B",
        "int16": "h",
        "uint16": "H",
        "int32": "i",
        "uint32": "I",
        "float16": 'f',
        "float32": "s"
    }
    for r in records:
        perf = {}
        perf["op_type"] = r["op_type"]
        perf["params"] = r["params"]
        perf["params"] = r["params"]

        perf["input_types"] = "".join([dtype_dict[i["dtype"]] for i in r["inputs"]])
        perf["output_types"] = "".join([dtype_dict[i["dtype"]] for i in r["outputs"]])

        perf["input_shapes"] = str([eval(i["shape"]) for i in r["inputs"]])
        perf["output_shapes"] = str([eval(i["shape"]) for i in r["outputs"]])
        for k in r["perf_data"]:
            if k not in ["Input Shapes", "Output Shapes"]:
                perf[k.replace(" ", "_").replace("/", "_").replace("(", "_").replace(")", "_").strip("_").split("@")[0]] = r["perf_data"][k]

        node_perfs.append(perf)

    # open("data.json", "w").write(json.dumps({"graph_perf_summary": new_graph_perf_summary, "node_perfs": node_perfs}))
    data = {"graph_perf_summary": new_graph_perf_summary, "node_perfs": node_perfs}
    import requests
    import json
    url = "http://10.193.9.76:8000/api/modelsummary/"
    r = requests.post(url, json=json.dumps(data))
    if r.status_code == 201:
        INFO("Performance data has been insert to database sucessfully")
        INFO(f"The performance data url is {url}{r.json()['id']}")
    else:
        WARN(f"performance data insert to database failed \n maybe the data format is not correct {r.json()}")

