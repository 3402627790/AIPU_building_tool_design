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
from .prof_x1 import get_perf_data_x1, fetch_custom_record_from_binary
from .html_generator import HtmlGenerator, TickGraphGenerator
from .csv_generator import CsvGenerator
from .get_data_x1 import x1_get_data
from ...logger import WARN, FATAL


def x1_gen_report(graph_json, binary, args):
    graph_name = graph_json["graph_name"]
    verbose = int(args.verbose)
    verbose = max(min(verbose, 2), 0)
    custom_record = fetch_custom_record_from_binary(graph_json, binary)
    nodes_info = get_perf_data_x1(graph_json, binary, args)
    if verbose >= 2 and len(custom_record) > 0:
        report_name = args.output if args.output else graph_name + "_perf.html"
        TickGraphGenerator().gen(graph_json, nodes_info, report_name, custom_record)
    elif args.method == "html":
        report_name = args.output if args.output else graph_name + "_perf.html"
        HtmlGenerator().gen(graph_json, nodes_info, report_name, verbose)
    elif args.method in ["csv", "xlsx"]:
        postfix = "xlsx"
        if args.method == "csv":
            WARN("csv format is alias to xlsx format")
        report_name = args.output if args.output else graph_name + f"_perf.{postfix}"
        CsvGenerator().gen(graph_json, nodes_info, report_name, verbose)
    elif args.method== "json":
        freq = args.frequency if args.frequency else graph_json["profile"]["frequency"]
        report_name = args.output if args.output else graph_name + "_perf.json"
        ret = x1_get_data(graph_json, binary, frequency = freq)
        with open(report_name, "w") as f:
            f.write(json.dumps(ret, indent = 2))
    else:
        FATAL(f"[Profiler] not support output file type {args.method}")
