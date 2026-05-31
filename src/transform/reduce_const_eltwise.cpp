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
#include "quantization_transform/utils/quantize_utils.h"

namespace aipubt
{

static inline bool criteria_for_eltwise(NodePtr n1)
{
    std::string method = n1->params()["method"];
    std::string with_activation = n1->params()["with_activation"];
    if ((with_activation != Constants::act_none && with_activation != Constants::act_relu && with_activation != Constants::act_clip) ||
        method == "MIN" || method == "MAX")
    {
        return false;
    }

    if (n1->inputs()[0]->shape().dim() > 4 || n1->inputs()[1]->shape().dim() > 4)
    {
        return false;
    }

    if (with_activation == Constants::act_clip)
    {
        if (n1->outputs()[0]->type() != Tensor::INT8 && n1->outputs()[0]->type() != Tensor::UINT8)
        {
            return false;
        }
        if (!n1->params().count("scale_value") || !n1->params().count("shift_value") ||
            !n1->params().count("clip_max") || !n1->params().count("clip_min"))
        {
            return false;
        }

        int shift = n1->params()["shift_value"];
        if (shift < 0)
        {
            return false;
        }
        int clip_max = n1->params()["clip_max"];
        int clip_min = n1->params()["clip_min"];
        int val = 0, zp = 0;
        zp = n1->outputs()[0]->quantization().offset();
        switch (n1->outputs()[0]->type())
        {
        case Tensor::INT8:
            val = INT8_MAX;
            break;
        case Tensor::UINT8:
            val = UINT8_MAX;
            break;
        default:
            val = 0;
            break;
        }

        if (clip_min != 0 || clip_max != (val + zp) << std::min(shift, 8))
        {
            return false;
        }
    }

    return true;
}

static inline bool criteria(NodePtr constant)
{
    auto const_consumers = constant->outputs()[0]->consumers();
    for (auto child : const_consumers)
    {
        if (child->type() != OpType_Eltwise || !criteria_for_eltwise(child))
        {
            return false;
        }
    }

    auto w = constant->constants()["weights"];
    auto t = w;
    for (int32_t i = 0; i < w->shape().dim(); i++)
    {
        auto ct = shrink_tensor_data(t, i, true);
        if (ct)
        {
            t = ct;
        }
    }
    if (t->mem_size() == w->mem_size())
    {
        return false;
    }
    // currently AIFF only support one dimemension broadcasting
    int not_1 = 0;
    for (auto d : t->shape())
    {
        if (d != 1)
        {
            not_1++;
            if (not_1 > 1)
            {
                return false;
            }
        }
    }

    constant->constants()["weights"] = t;

    return true;
}

bool reduce_const_eltwise(GraphPtr g)
{
    static std::map<std::string, OpType> tmap = {
        {"ADD", OpType_Add},
        {"SUB", OpType_Sub},
        {"MUL", OpType_Mul},
        {"DIV", OpType_Div},
    };
    static GraphMatcherPattern pattern = {
        {{"0", OpType_Constant}, {"1", OpType_Eltwise}},
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
            if (!criteria(m[0]))
            {
                continue;
            }
            if (removed.count(m[0]) || removed.count(m[1]))
            {
                continue;
            }
            auto out = m[0]->outputs()[0];
            out->shape() = m[0]->constants()["weights"]->shape();

            auto const_consumers = m[0]->outputs()[0]->consumers();
            for (auto eltwise : const_consumers)
            {
                auto el_out = eltwise->outputs()[0];
                std::string method = eltwise->params()["method"];
                std::string with_activation = eltwise->params()["with_activation"];
                auto node = Node::create(g->get_valid_node_name(method), tmap[method]);
                node->params() = eltwise->params();
                //remove activation and method
                node->params().erase("method");
                if (with_activation == Constants::act_none)
                {
                    node->params().erase("with_activation");
                }
                else if (with_activation == Constants::act_clip)
                {
                    node->params()["with_activation"] = Constants::act_relu;
                    node->params().erase("clip_min");
                    node->params().erase("clip_max");
                }
                node->add_input(eltwise->inputs());
                node->add_input(out, node->remove_input(out));
                node->add_output(el_out);
                g->add_node(node);
                ADD_LOCATION_BY_NODE(node, m[0], eltwise);

                //remove etlwise
                eltwise->clear_inputs();
                eltwise->clear_outputs();
                g->remove_node(eltwise);
                removed.insert(eltwise);
            }

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