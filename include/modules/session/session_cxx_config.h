//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Alex Zheng
 * @Date: 2026-04-14 14:24:30
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2026-04-14 14:15:49
 */
#pragma once

#include "common_def.h"
#include <cstdint>
#include <string>
#include <vector>

namespace cpsruntime
{

/**
 * @brief Session Config
 */
typedef struct SessionConfig
{
    /**
     * Specify AIPU arch, only for running with AIPU Simulator.
     * If running on hardware, do not fill in this parameter.
     *  - 'X3P_1304'/'X3P_1304MP2'/'X3P_1304MP4'
     *  - 'X3S_1304'/'X3S_1304MP2'/'X3S_1304MP4'
     */
    std::string arch = "X3S_1304MP2";

    /** This corresponds to the opening of the Compass runtime profile function.
     * Default is disable.
     */
    bool enable_profile = false;

    /**
     * Profiler max aiff sample point number, only for v3,
     * take effect when 'enable_profile' is true
     *  - '0' uses default sample number
     */
    uint32_t max_aiff_sample_point_num = 0;

    /**
     *  This corresponds to enabling the "tiling" function..
     *  - 'fps': pay close attention to fps
     *  - empty: disable tiling (default)
     */
    std::string tiling_method = {};

    /**
     * Run mode of graph compilation, value from enumeration `CpsQuantMode`
     * default is FASYM
     */
    CpsQuantMode q_mode = CPS_QUANT_FASYM;

    /**
     * Priority of graph execution, value from enumeration `Priority`
     * default is 0
     */
    CpsRunPriority priority = CPS_RUN_PRIORITY_HIGH;

    /**
     * Enable compiler pass, if serveral passes, use ',' to seperate them.
     * Such as:"eliminate_dense_crop"
     */
    std::string enable_pass = {};

    /**
     * Disable compiler pass, if serveral passes, use ',' to seperate them.
     * Such as:"eliminate_dense_crop, fuse_tile_eltwise"
     */
    std::string disable_pass = {};

    /**
     * Ddisable compiler plugin, if serveral passes, use ',' to seperate them.
     * such as:
     *  "HwaLRN, dwconv_pwconv"
     */
    std::string disable_plugin = {};

    /**
     * Setting the SOC SRAM size or On-Chip-Memory size, the allocator will try to use these memory.
     */
    int ocm_size = 0;

    /**
     * Dump the intermediate graph for debugging.
     */
    bool enable_debug_dump = false;

} SessionConfig;

} // namespace cpsruntime