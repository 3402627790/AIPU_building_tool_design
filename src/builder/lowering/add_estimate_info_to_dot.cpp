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
 * @Author: Yuchou.Gan
 * @Date: 2020-02-25 17:58:05
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-12-14 15:01:10
 */
#include <builder/lowering/lowering.h>
#include <core/config.h>
#include <core/core.h>
#include <statistic/statistic_pass.h>
#include <utils/utils.h>

namespace aipubt
{

bool add_estimate_info_to_dot(GraphPtr g)
{
    bool show_perf_info = false;
    if (BuilderConfig::exist(Constants::show_info))
    {
        if (BuilderConfig::get<std::set<std::string>>(Constants::show_info).count("perf"))
        {
            show_perf_info = true;
        }
    }
    if (!show_perf_info)
    {
        return false;
    }
    auto m_nodes = g->nodes();
    for (auto &&n : m_nodes)
    {
        auto sgn = SubGraphNode::from_node(n);
        if (sgn && !sgn->has_plugin())
        {
            add_estimate_info_to_dot(sgn->subgraph());
            continue;
        }
        if (n->attrs().count(Constants::performance_info))
        {
            std::ostringstream oss;
            PerfDataItem item = n->get_attr<PerfDataItem>(Constants::performance_info);
            oss << "estimated cycle: " << (uint64_t)(item.count_cycles)
                << ", R/W: " << utils::get_human_readable_size(item.read_bytes) << "/" << utils::get_human_readable_size(item.write_bytes)
                << std::setprecision(2)
                << "\n R/W BW:" << item.read_bytes / (item.count_cycles + 1e-6) << "(GB/s)/" << item.write_bytes / (item.count_cycles + 1e-6) << "(GB/s)";
            n->add_dot_section(Constants::performance_info, oss.str());
        }
    }
    return false;
}
}; // namespace aipubt
