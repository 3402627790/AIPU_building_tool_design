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
import numpy as np
from AIPUBuilder.logger import *


def get_extra_info_from_bin(graph_json, binary):
    """
    Get extra info from binary

    Parameters
    ----------
    graph_json : str
        The graph_json dict, also return value to the dict

    binary : bytes
        The binary context
    """

    info = graph_json["profile"]


    if info["output size"] > binary.size:
        WARN("[Profiler] The binary file size is smaller than expected in graph.json")
        FATAL("[Profiler] Incompatible json and dump profile data")

    # Try to load emu info if exist
    if info["output size"] < binary.size:
        # parse text from tail of binary
        extra_info = binary[info["output size"]:]
        success = 1
        try:
            header = extra_info[:4].tobytes().decode("utf-8")
            size = extra_info[4:12]
            size.dtype = np.uint64
            size = int(size)
            content = extra_info[12:size].tobytes().decode("utf-8")
            content = json.loads(content)
        except RuntimeError:
            success = 0
        except json.decoder.JSONDecodeError:
            WARN("[Profiler] The JSON field of binary file is corrupted")
            success = 0
        if size != extra_info.size:
            WARN("[Profiler] The extra info size error")
            success = 0
        if header != "emu\0":
            WARN("[Profiler] The extra info header error")
            success = 0
            
        if not success:
            FATAL("[Profiler] Incompatiable json and dump profile data. Please check the graph.json file and profiler result binary file")
        else:
            info["emu"] = content
