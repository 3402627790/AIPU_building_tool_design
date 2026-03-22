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
#include <utils/utils.h>

namespace aipubt
{

static std::string show_size(uint32_t size, utils::DataUnit unit)
{
    std::ostringstream oss;
    oss << std::setfill(' ') << std::setw(9) << utils::get_human_readable_size(size, unit, 1);
    return oss.str();
};

bool add_tensor_size_to_dot(GraphPtr g)
{
    bool show_tsize_info = false;
    if (BuilderConfig::exist(Constants::show_info))
    {
        if (BuilderConfig::get<std::set<std::string>>(Constants::show_info).count("tsize"))
        {
            show_tsize_info = true;
        }
    }
    if (!show_tsize_info)
    {
        return false;
    }
    auto m_nodes = g->nodes();
    std::vector<std::string> allocated_wt;
    for (auto &&n : m_nodes)
    {
        auto sgn = SubGraphNode::from_node(n);
        if (sgn && !sgn->has_plugin())
        {
            add_tensor_size_to_dot(sgn->subgraph());
            continue;
        }
        uint32_t i_size = 0;
        uint32_t o_size = 0;
        uint32_t w_size = 0;

        for (auto t : n->inputs())
        {
            i_size += t->mem_size();
        }
        for (auto t : n->outputs())
        {
            o_size += t->mem_size();
        }
        for (auto t : n->constants())
        {
            if (t.second && !contains(allocated_wt, t.second->name()))
            {
                w_size += t.second->mem_size();
                allocated_wt.push_back(t.second->name());
            }
        }
        auto data_unit = utils::get_best_data_unit(i_size + o_size + w_size);
        std::ostringstream oss;
        oss << "i_size: " << show_size(i_size, data_unit) << "\n";
        oss << "o_size: " << show_size(o_size, data_unit) << "\n";
        oss << "w_size: " << show_size(w_size, data_unit) << "\n";
        if (n->params().count("stride_y") > 0 && n->params().count("kernel_y") > 0 && n->params().count("dilation_y"))
        {
            oss << "|";
            int32_t stride_y = n->params()["stride_y"];
            int32_t kernel_y = n->params()["kernel_y"];
            int32_t dilation_y = n->params()["dilation_y"];
            // i_size - o_size = (stri-1) * o_size + (k-1)*(d-1) + (k-stri-pad)
            if (stride_y == 1)
            {
                oss << "overlap = " << (kernel_y - 1) * (dilation_y - 1) + (kernel_y - stride_y);
            }
            else
            {
                oss << "overlap = " << (kernel_y - stride_y);
            }
        }
        n->add_dot_section(".tensor_size", oss.str());
    }
    return false;
}
}; // namespace aipubt
