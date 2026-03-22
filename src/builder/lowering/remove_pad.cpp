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
 * @Author: Lang Hu
 * @Date: 2025-04-12 10:17:18
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2025-04-12 10:17:18
 */
#include "builder/lowering/lowering.h"
#include "builder/ops/ops.h"
#include "core/config.h"
#include "core/core.h"

namespace aipubt
{

static bool nonsupport_input_layout(NodePtr node)
{
    std::set<DataLayout> nonsupport_layout = {
        DataLayout_NCHWC16T8,
        DataLayout_NCHWC32T8,
        DataLayout_NCHWC16T1,
        DataLayout_NCHWC32T1,
        DataLayout_NHWCSparse,
        DataLayout_NCHWC16Sparse,
        DataLayout_NCHWC32Sparse,
        DataLayout_NCHWC16T8Sparse,
        DataLayout_NCHWC32T8Sparse,
        DataLayout_NCHWC16T1Sparse,
        DataLayout_NCHWC32T1Sparse,
        DataLayout_NDCHWC16T8,
        DataLayout_NDCHWC32T8,
        DataLayout_NDCHWC16T1,
        DataLayout_NDCHWC32T1,
    };
    if (nonsupport_layout.count(node->inputs()[0]->shape().layout()))
    {
        return true;
    }
    return false;
}

static bool is_memory_not_changed(NodePtr node)
{
    if (!node->is_dynamic())
    {
        return false;
    }
    if (!get_real_node(node)->attrs().count(Constants::n_align))
    {
        return false;
    }
    if (node->inputs()[0]->shape().n() != node->outputs()[0]->shape().n())
    {
        return false;
    }
    return true;
}

void remove_pad(GraphPtr g)
{
    auto criterion = [](NodePtr node) -> bool {
        if (get_real_op_type(node) != OpType_Pad)
        {
            return false;
        }
        if (nonsupport_input_layout(node))
        {
            return false;
        }
        if (!is_memory_not_changed(node))
        {
            return false;
        }
        // not support internal pad: TODO fixme optimize it
        std::set<OpType> nonsupport_output_type =
            {
                OpType_InternalConcatenated,
            };
        for (auto c : node->children())
        {
            if (c->type() == OpType_LayoutConvertor && is_a_remapping_convertor(c, false))
            {
                for (auto t : c->children())
                {
                    if (nonsupport_output_type.count(t->type()))
                    {
                        return false;
                    }
                }
            }
            if (nonsupport_output_type.count(c->type()))
            {
                return false;
            }
        }
        return true;
    };

    auto nodes = g->nodes();
    for (auto &&n : nodes)
    {
        if (!criterion(n))
        {
            continue;
        }

        auto real_node = get_real_node(n);
        auto internalpad_node = Node::create(real_node->name(), OpType_InternalPad);
        internalpad_node->params()["mode"] = real_node->params()["mode"];
        internalpad_node->params()["constant_value"] = real_node->params()["constant_value"];
        internalpad_node->params()["pads"] = real_node->params()["pads"];
        internalpad_node->attrs() = std::move(real_node->attrs());
        if (real_node->params().count("ds_pads"))
        {
            internalpad_node->params()["ds_pads"] = real_node->params()["ds_pads"];
        }
        ADD_LOCATION_BY_NODE(internalpad_node, n);
        g->replace_node_by(n, internalpad_node);
    }
}

} // namespace aipubt