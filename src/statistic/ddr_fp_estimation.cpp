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
 * @Author: Neo Wang
 * @Date: 2022-05-13 14:12:11
 * @Last Modified by: Tom Gao
 * @Last Modified time: 2023-02-23 15:32:50
 */
#include "ddr_fp_estimation.h"
#include "core/core.h"
#include "utils/utils.h"

namespace aipubt
{

std::string DDRFootPrintEstimation::summary()
{
    std::ostringstream oss;
    oss << "DDR Footprint statistic(estimated):\n"
        << "Read :" << utils::get_human_readable_size(m_res[0].read) << "\n"
        << "Write:" << utils::get_human_readable_size(m_res[0].write);
    return oss.str();
}

bool DDRFootPrintEstimation::operator()(GraphPtr g) const
{
    auto big_buffer_thres = aipubt::get_big_buffer_thresholds();
    auto numbers = big_buffer_thres.size();
    m_res.clear();
    for (size_t i = 0; i < numbers; ++i)
    {
        m_res.push_back({0, 0});
    }
    auto get_big_size = [](uint64_t cur_size, uint64_t big_size) -> uint64_t {
        uint64_t size = cur_size > big_size ? cur_size : 0;
        return size;
    };
    auto is_crop_in_tensor = [](const NodePtr &n, const TensorPtr &t, size_t &read_size) -> bool {
        auto sgn = SubGraphNode::from_node(n);
        if (sgn) // if crop lead need special handle
        {
            auto nodes = sgn->subgraph()->nodes();
            for (auto node : nodes)
            {
                if (node->type() == OpType_Crop && node->parents().size() == 0)
                {
                    auto crop_input_p = std::dynamic_pointer_cast<ProxyTensor>(node->inputs()[0]);
                    if (crop_input_p->proxied_tensor() == t)
                    {
                        read_size = node->outputs()[0]->mem_size();
                        return true;
                    }
                }
            }
        }
        return false;
    };
    if (g)
    {
        std::vector<uint64_t> total_rs(numbers, 0);
        std::vector<uint64_t> total_ws(numbers, 0);
        if (m_set_obj_fp)
        {
            //clear fp info
            for (auto n : g->nodes())
            {
                if (n->type() == OpType_SubGraphInput)
                {
                    continue;
                }
                for (auto t : n->outputs())
                {
                    auto obj = t->memory_offset()->get_base_object();
                    obj->foot_print = 0;
                }
            }
        }

        for (auto n : g->nodes())
        {
            //skip reshape etc.
            if (n->type() == OpType_Reshape ||
                n->type() == OpType_InternalConcatenated ||
                n->type() == OpType_InternalCrop ||
                n->type() == OpType_InternalSlice ||
                n->type() == OpType_InternalSpliter ||
                n->type() == OpType_InternalPad ||
                n->type() == OpType_Input ||
                n->type() == OpType_InternalConstant ||
                n->type() == OpType_SubGraphInput ||
                n->type() == OpType_Dummy ||
                n->type() == OpType_Empty)
            {
                continue;
            }
            std::vector<uint64_t> node_rs(numbers, 0);
            std::vector<uint64_t> node_ws(numbers, 0);

            // special nodes
            auto sgn = SubGraphNode::from_node(n);
            if (sgn && !sgn->has_plugin())
            {
                auto sg = sgn->subgraph();
                DDRFootPrintEstimation fp;
                fp.update_memory_obj(m_set_obj_fp);
                fp(sg);
                auto res = fp.get_result();
                for (size_t i = 0; i < numbers; ++i)
                {
                    auto big_buffer_size = big_buffer_thres[i];
                    node_rs[i] += get_big_size(res[i].read, big_buffer_size);
                    node_ws[i] += get_big_size(res[i].write, big_buffer_size);
                }
            }
            // normal nodes
            else
            {
                for (auto t : n->inputs())
                {
                    auto size = t->mem_size();
                    is_crop_in_tensor(n, t, size);
                    for (size_t i = 0; i < numbers; ++i)
                    {
                        auto big_buffer_size = big_buffer_thres[i];
                        node_rs[i] += get_big_size(size, big_buffer_size);
                    }
                    if (m_set_obj_fp)
                    {
                        auto mmu = t->memory_offset()->get_mmu_object();
                        auto ptr = t->memory_offset()->get_base_object();
                        if (mmu)
                        {
                            mmu->next_block->foot_print += size;
                        }
                        else
                        {
                            /* code */
                            ptr->foot_print += size;
                        }
                    }
                }
                for (auto t : n->outputs())
                {
                    auto size = t->mem_size();
                    for (size_t i = 0; i < numbers; ++i)
                    {
                        auto big_buffer_size = big_buffer_thres[i];
                        node_ws[i] += get_big_size(size, big_buffer_size);
                    }
                    if (m_set_obj_fp)
                    {
                        auto mmu = t->memory_offset()->get_mmu_object();
                        auto ptr = t->memory_offset()->get_base_object();
                        if (mmu)
                        {
                            mmu->next_block->foot_print += size;
                        }
                        else
                        {
                            /* code */
                            ptr->foot_print += size;
                        }
                    }
                }
                if (m_with_const)
                {
                    for (auto kv : n->constants())
                    {
                        if (kv.second)
                        {
                            auto size = kv.second->mem_size();
                            for (size_t i = 0; i < numbers; ++i)
                            {
                                // constants should be on DDR, and could not be allocated in DTCM/SRAM
                                node_rs[i] += size;
                            }
                        }
                    }
                }
                if (m_set_obj_fp)
                {
                    for (auto kv : n->constants())
                    {
                        if (kv.second)
                        {
                            auto size = kv.second->mem_size();
                            for (size_t i = 0; i < numbers; ++i)
                            {
                                // constants should be on DDR, and could not be allocated in DTCM/SRAM
                                kv.second->memory_offset()->foot_print = size;
                            }
                        }
                    }
                }
            }
            for (size_t i = 0; i < numbers; ++i)
            {
                total_rs[i] += node_rs[i];
                total_ws[i] += node_ws[i];
            }
        }

        for (size_t i = 0; i < numbers; ++i)
        {
            m_res[i].read = total_rs[i];
            m_res[i].write = total_ws[i];
        }
    }

    return false;
}
}; // namespace aipubt