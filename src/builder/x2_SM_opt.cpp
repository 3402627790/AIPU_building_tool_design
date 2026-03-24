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
 * @Date: 2020-07-28 11:11:21 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2026-01-06 17:24:41
 */

#include "x2_SM_opt.h"
#include "builder/lowering/lowering.h"
#include "core/config.h"
#include "core/constants.h"
#include "core/core.h"
#include "core/node.h"
#include "core/subgraph.h"
#include "mem_allocator.h"
#include "optimizer.h"
#include <algorithm>
#include <set>
#include <vector>

namespace aipubt
{

static inline bool sram_free(NodePtr n)
{
    if (!n->attrs().count(Constants::sram_free))
    {
        return false;
    }
    else if (!n->get_attr<bool>(Constants::sram_free))
    {
        return false;
    }

    bool ret = true;
    for (auto t : n->inputs())
    {
        auto base = t->memory_offset()->get_base_object();
        uint32_t base_addr = base->get_base_memory();
        if (base_addr >= 0xf8000000)
        {
            ret = false;
        }
    }
    for (auto t : n->outputs())
    {
        auto base = t->memory_offset()->get_base_object();
        uint32_t base_addr = base->get_base_memory();
        if (base_addr >= 0xf8000000)
        {
            ret = false;
        }
    }

    return ret;
}

static inline bool internal_op(NodePtr n)
{
    bool ret = false;
    if (n->type() == OpType_InternalConcatenated ||
        n->type() == OpType_InternalCrop ||
        n->type() == OpType_InternalSlice ||
        n->type() == OpType_InternalSpliter ||
        n->type() == OpType_InternalPad ||
        n->type() == OpType_SubGraphInput ||
        n->type() == OpType_LayoutConvertor ||
        n->type() == OpType_Reshape ||
        n->type() == OpType_Dummy)
    {
        ret = true;
    }
    return ret;
}

static inline uint32_t weight_size(NodePtr n)
{
    uint32_t ret = 0;
    for (auto c : n->constants())
    {
        ret += c.second->mem_size();
    }
    return ret;
}

bool ensure_weight_in_sm(GraphPtr g, NodePtr next, NodePtrList &splited)
{
    if (get_real_op_type(next) == OpType_Convolution)
    {
        uint32_t sm_size = 256 * 1024; //256KB
        uint32_t preload = sm_size / 2;
        auto sgn_tmp = SubGraphNode::from_node(next);
        if (sgn_tmp)
        {
            next = sgn_tmp->unfold()[0];
        }
        auto ws = weight_size(next);
        auto oc = next->outputs()[0]->shape().c_total();
        int split_c = preload / (ws / oc);
        // align to 32 then  align 8
        if (!(split_c & (~0x1f)))
        {
            split_c = split_c & (~0x7);
            if (!split_c)
            {
                return false;
            }
        }
        else
        {
            split_c = split_c & (~0x1f);
        }
        std::vector<int> splits = {split_c};
        int c = oc - split_c;
        while (c > 0)
        {
            if (c > split_c)
            {
                splits.push_back(split_c);
                c -= split_c;
            }
            else
            {
                splits.push_back(c);
                break;
            }
        }

        splited = split_conv_output_c_(g, next, splits);
        return true;
    }
    return false;
}

// pre load weight to sm for next layer
void x2_sm_opt(SubGraphNodePtr sgn)
{
    if (!sgn)
    {
        return;
    }
    //1. find the subgraph that SRAM is free
    //2. find out the nodes that can move weight to sram( previous node not use sram, and itself not use neither)
    //3. if weight is to large, try to split
    //4. setup optimization info for builder to generate correct code and memory address

    //for simply the problem, we will flat the sg, make the hierarch simple
    sgn->flat();
    auto sg = sgn->subgraph();
    sg->topological_sort();
    auto nodes = sg->nodes();
    std::vector<NodePtrList> sm_nodes;
    NodePtrList chain;
    for (auto n : nodes)
    {
        if (internal_op(n))
        {
            continue;
        }

        if (sram_free(n))
        {
            chain.push_back(n);
        }
        else
        {
            if (chain.size() > 1)
            {
                sm_nodes.push_back(chain);
            }
            chain = NodePtrList();
        }
    }
    if (chain.size() > 1)
    {
        sm_nodes.push_back(chain);
        chain = NodePtrList();
    }

    uint32_t sm_size = 256 * 1024; //256KB
    uint32_t preload = sm_size / 2;
    for (auto &chain : sm_nodes)
    {
        for (uint32_t i = 0; i < chain.size() - 1; ++i)
        {
            auto next = chain[i + 1];
            auto ws = weight_size(next);
            if (ws < preload) // the weight is not aligned TODO!!! after align, the size may bigger
            {
                //TODO put weight in SM
                //TODO disbale descriptor chain
                chain[i]->set_attr("_movwtosm", NodeRef(next));
                chain[i]->set_attr("_preload_sm_size", (int)preload);
                next->set_attr("_ldwfromsm", true);
                INFO() << next->name() << " weight size = " << ws;
            }
            // INFO() << chain[i + 1]->name() << " need split channel";
        }
    }
}
} // namespace aipubt
