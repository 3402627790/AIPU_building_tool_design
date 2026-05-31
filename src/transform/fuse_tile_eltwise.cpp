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
/*
 * @Author: Samuel Han
 * @Date: 2022-03-30 13:38:32
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-07-05 10:40:45
 */

#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include "quantization_transform/utils/quantize_utils.h"

namespace aipubt
{

static inline bool criteria(NodePtr tile, NodePtr n1, int &c_dim)
{
    auto inp = tile->inputs()[0]->shape();
    // auto out=tile->outputs()[0]->shape();
    // std::vector<int> repeats=tile->params()["repeats"];
    int count_n1 = 0;
    for (int i = 0; i < inp.dim(); ++i)
    {
        if (inp[i] != 1)
        {
            count_n1++;
            c_dim = i;
        }
    }
    if (count_n1 > 1)
    {
        return false;
    }
    std::string method = n1->params()["method"];
    std::string with_activation = n1->params()["with_activation"];
    // ADD/SUB/MUL only support NONE or RELU
    if ((with_activation != Constants::act_none && with_activation != Constants::act_relu) ||
        method == "MIN" || method == "MAX")
    {
        return false;
    }

    if (n1->inputs()[0]->shape().dim() > 4 || n1->inputs()[1]->shape().dim() > 4)
    {
        return false;
    }

    bool both_tile = std::all_of(n1->inputs().begin(), n1->inputs().end(), [](const TensorPtr t)
                                 { return t->node()->type() == OpType_Tile; });
    if (both_tile)
    {
        return false;
    }
    if (n1->is_dynamic() &&
        tile->outputs()[0]->shape().symbol() != n1->outputs()[0]->shape().symbol())
    {
        return false;
    }

    return true;
}

bool fuse_tile_eltwise(GraphPtr g)
{
    static std::map<std::string, OpType> tmap = {
        {"ADD", OpType_Add},
        {"SUB", OpType_Sub},
        {"MUL", OpType_Mul},
        {"DIV", OpType_Div},
    };
    static GraphMatcherPattern pattern = {
        {{"0", OpType_Tile}, {"1", OpType_Eltwise}},
        {{"0", "1"}}};
    static GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);
    auto matched = matcher.match(g);
    std::set<NodePtr> removed;
    bool is_changed = false;
    while (matched.size())
    {
        bool is_merged = false;
        for (auto &&m : matched)
        {
            int c_dim = -1;
            if (!criteria(m[0], m[1], c_dim))
            {
                continue;
            }
            if (removed.count(m[0]) || removed.count(m[1]))
            {
                continue;
            }
            auto inp = m[0]->inputs()[0];
            auto out = m[0]->outputs()[0];
            auto el_out = m[1]->outputs()[0];
            // if (c_dim > 0)
            // {
            //     //insert reshape to make lib happy
            //     std::vector<int> new_shape = {inp->shape()[c_dim]};
            //     for (int i = c_dim + 1; i < inp->shape().dim(); ++i)
            //     {
            //         new_shape.push_back(1);
            //     }
            //     inp = ops::reshape(inp, new_shape);
            // }
            std::string method = m[1]->params()["method"];
            auto node = Node::create(g->get_valid_node_name(method), tmap[method]);
            node->params() = m[1]->params();
            // remove activation and method
            node->params().erase("method");
            // node->params().erase("with_activation");
            node->add_input(m[1]->inputs());
            node->add_input(inp, node->remove_input(out));
            node->add_output(el_out);
            g->add_node(node);
            ADD_LOCATION_BY_NODE(node, m[0], m[1]);

            // remove etlwise
            m[1]->clear_inputs();
            m[1]->clear_outputs();
            g->remove_node(m[1]);
            removed.insert(m[1]);

            if (out->consumers().size() == 0 && !out->is_io_tensor())
            {
                m[0]->clear_inputs();
                m[0]->clear_outputs();
                g->remove_node(m[0]);
                removed.insert(m[0]);
            }
            is_changed = true;
            is_merged = true;
        }
        if (!is_merged)
        {
            break;
        }
        matched = matcher.match(g);
    }

    return is_changed;
}
}; // namespace aipubt