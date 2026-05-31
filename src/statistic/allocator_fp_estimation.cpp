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
 * @Last Modified time: 2022-07-31 11:18:29
 */
#include "allocator_fp_estimation.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/core.h"
#include "utils/utils.h"

namespace aipubt
{

std::string AllocatorFPEst::summary()
{
    std::ostringstream oss;
    oss << "Allocator Footprint statistic(estimated):\n"
        << "Feature Map total footprint(RW) : " << utils::get_human_readable_size(m_res);
    return oss.str();
}

static uint32_t size_in_gm(GraphPtr g, uint32_t addr, uint32_t size)
{
    auto sg = std::dynamic_pointer_cast<SubGraph>(g);
    auto ori_graph = sg ? sg->origin_graph() : g;
    uint32_t gm_addr0 = ori_graph->get_attr("gm_addr0");
    uint32_t gm_addr1 = ori_graph->get_attr("gm_addr1");
    uint32_t gm_size0 = ori_graph->get_attr("gm_size0");
    uint32_t gm_size1 = ori_graph->get_attr("gm_size1");
    uint32_t ret = 0;
    if (BuilderConfig::get<bool>(Constants::disable_gm))
    {
        ret = 0;
    }
    else
    {
        if (gm_size1 && (addr >= gm_addr1) && (addr < gm_addr1 + gm_size1))
        {
            if (addr + size > gm_addr1 + gm_size1)
            {
                ret = gm_addr1 + gm_size1 - addr;
            }
            else
            {
                ret = size;
            }
        }
        else if (gm_size0 && (addr >= gm_addr0) && (addr < gm_addr0 + gm_size0))
        {
            if (addr + size > gm_addr0 + gm_size0)
            {
                ret = gm_addr0 + gm_size0 - addr;
            }
            else
            {
                ret = size;
            }
        }
    }
    return ret;
}

static bool is_in_dtcm(uint32_t addr)
{
    bool ret = false;
    if (BuilderConfig::get<bool>(Constants::enable_tcm))
    {
        uint32_t dtcm_size = BuilderConfig::get<uint32_t>(Constants::tcm_size);
        uint32_t dtcm_base = 0xD0000000;
        if (addr >= dtcm_base && addr < (dtcm_base + dtcm_size))
        {
            ret = true;
        }
    }
    return ret;
}

static uint64_t get_tensor_ddr_size(GraphPtr g, TensorPtr t, size_t tensor_size = 0)
{
    uint64_t ddr_size = 0;
    auto obj = t->memory_offset()->get_base_object();
    auto total_size = tensor_size > 0 ? tensor_size : t->mem_size();

    // first part in SRAM
    if (obj->offset() >= 0xe0000000)
    {
        auto mmu = t->memory_offset()->get_mmu_object();
        if (mmu)
        {
            // SRAM + SRAM
            if (mmu->base_offset->offset() >= 0xE0000000 && mmu->next_block->offset() >= 0xE0000000)
            {
                ddr_size = 0;
            }
            // SRAM + DDR/DTCM
            else if (total_size)
            {
                //allocate the second part for it
                auto mmu_second = mmu->next_block;
                ddr_size = is_in_dtcm(mmu_second->offset()) ? 0 : mmu_second->size;
            }
        }
        // SRAM only
        else
        {
            ddr_size = 0;
        }
    }
    // first part in DDR/DTCM
    else
    {
        if (aipubt::get_target()->isainfo().type >= aipubt::ISA_ZYv5) // x2
        {
            ddr_size = total_size - size_in_gm(g, obj->offset(), total_size);
        }
        else
        {
            ddr_size = is_in_dtcm(obj->offset()) ? 0 : total_size;
        }
    }
    return ddr_size;
}

bool AllocatorFPEst::operator()(GraphPtr g) const
{
    m_res = 0;

    if (g)
    {
        for (auto n : g->nodes())
        {
            if (n->type() == OpType_Reshape ||
                n->type() == OpType_InternalConcatenated ||
                n->type() == OpType_InternalCrop ||
                n->type() == OpType_InternalSlice ||
                n->type() == OpType_InternalSpliter ||
                n->type() == OpType_InternalPad ||
                n->type() == OpType_Input ||
                n->type() == OpType_SubGraphInput ||
                n->type() == OpType_Dummy ||
                n->type() == OpType_Empty)
            {
                continue;
            }
            auto sgn = SubGraphNode::from_node(n);
            if (sgn && !sgn->has_plugin())
            {
                auto sg = sgn->subgraph();
                AllocatorFPEst fp;
                fp(sg);
                m_res += fp.get_result();
                continue;
            }

            // normal nodes
            uint64_t node_rs = 0;
            uint64_t node_ws = 0;
            for (auto t : n->inputs())
            {
                auto size = get_tensor_ddr_size(g, t);
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
                                size = get_tensor_ddr_size(g, t, node->outputs()[0]->mem_size());
                            }
                        }
                    }
                }
                node_rs += size;
            }
            for (auto t : n->outputs())
            {
                auto size = get_tensor_ddr_size(g, t);
                node_ws += size;
            }

            // const read
            for (auto kv : n->constants())
            {
                if (kv.second)
                {
                    auto size = kv.second->mem_size();
                    node_rs += size;
                }
            }

            m_res += node_rs;
            m_res += node_ws;
        }
    }

    return false;
}
}; // namespace aipubt