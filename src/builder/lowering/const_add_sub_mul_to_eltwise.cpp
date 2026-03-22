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
 * @Author: Yuxian Chen
 * @Date: 2024-02-21 14:56:07
 * @Last Modified by: Yuxian Chen
 * @Last Modified time: 2024-02-21 14:56:07
 */
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/ops/ops.h"

namespace aipubt
{
static bool criteria(NodePtr n, NodePtr const_node)
{
    auto target = get_target();
    // TODO: Make this pass only work for X1 now(need check performance for other targets)
    if (target->name().substr(0, 2) != "X1")
    {
        return false;
    }
    std::vector<uint32_t> s0 = n->inputs()[0]->shape();
    std::vector<uint32_t> s1 = n->inputs()[1]->shape();
    std::vector<uint32_t> out_s0 = n->outputs()[0]->shape();
    auto const_out = const_node->outputs()[0];
    if (n->inputs()[0] == const_out) // the first input is const
    {
        if (s1 != out_s0)
        {
            return false;
        }
    }
    else
    { // the second input is const
        if (s0 != out_s0)
        {
            return false;
        }
    }
    if (const_out->consumers().size() > 1)
    {
        return false;
    }
    // Do not convert to eltwise for these conditions:
    // 1) shape size(prod) of const_out is 1
    // 2) input/output is 4d and channels is 8-aligned
    // 3) the axis of broadcast is the last dim
    if (const_out->shape().size() == 1)
    {
        return false;
    }
    auto dim = out_s0.size();
    if (dim == 4 && n->outputs()[0]->shape().c() % 8 == 0)
    {
        std::vector<uint32_t> reshaped_const_shape(const_out->shape());
        while (out_s0.size() > reshaped_const_shape.size())
        {
            reshaped_const_shape.insert(reshaped_const_shape.begin(), 1);
        }
        std::vector<uint32_t> broadcast_axes;
        for (uint32_t i = 0; i < dim; i++)
        {
            if (reshaped_const_shape[i] != out_s0[i])
            {
                broadcast_axes.push_back(i);
            }
        }
        if (broadcast_axes.size() == 1 && broadcast_axes[0] == dim - 1)
        {
            return false;
        }
    }
    return true;
}

void const_add_sub_mul_to_eltwise(GraphPtr g)
{
    static std::map<OpType, std::string> method_map = {
        {OpType_Add, "ADD"},
        {OpType_Sub, "SUB"},
        {OpType_Mul, "MUL"},
    };
    std::vector<GraphMatcherPattern> patterns = {
        {{{"0", OpType_Constant}, {"1", OpType_Mul}}, {{"0", "1"}}},
        {{{"0", OpType_Constant}, {"1", OpType_Add}}, {{"0", "1"}}},
        {{{"0", OpType_Constant}, {"1", OpType_Sub}}, {{"0", "1"}}},
    };
    for (auto &pattern : patterns)
    {
        GraphMatcher matcher(pattern);
        matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
        auto matched = matcher.match(g);
        for (auto &&m : matched)
        {
            auto const_node = m[0];
            auto n = m[1];
            if (!criteria(n, const_node))
            {
                continue;
            }

            // modify weight, output shape and layout of const node
            TensorPtr const_weight = const_node->constants()["weights"];
            auto zeros = ComputeTensor::zeros(n->outputs()[0]->shape(), const_weight->type());
            auto new_weight = ComputeTensor(const_weight) + zeros;
            auto layout = n->outputs()[0]->shape().layout();
            new_weight->shape().set_layout(layout);
            new_weight->monopolize_data();
            const_node->constants()["weights"] = new_weight;
            const_node->outputs()[0]->shape() = new_weight->shape();
            MODIFY_NODE(const_node);

            // replace mul/add/sub to Eltwise
            auto elt_node = Node::create(g->get_valid_node_name(n->name() + "_elt"), OpType_Eltwise);
            elt_node->constants() = std::move(n->constants());
            elt_node->params() = std::move(n->params());
            elt_node->params()["method"] = method_map[n->type()];
            if (!elt_node->params().count("with_activation"))
            {
                elt_node->params()["with_activation"] = "NONE";
            }
            ADD_LOCATION_BY_NODE(elt_node, n);
            g->replace_node_by(n, elt_node);
        }
    }
}
}; // namespace aipubt
