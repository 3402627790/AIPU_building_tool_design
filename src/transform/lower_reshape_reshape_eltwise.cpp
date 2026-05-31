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
 * @Date: 2020-05-26 17:25:56 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-03-27 11:25:36
 */
#include "builder/lowering/lowering.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include <algorithm>
#include <set>
#include <vector>

namespace aipubt
{

static bool criteria(const std::vector<NodePtr> &nodes)
{
    auto n0 = nodes[0];
    auto n1 = nodes[1];
    if (n0->outputs()[0]->is_io_tensor() || n1->outputs()[0]->is_io_tensor())
    {
        return false;
    }

    return true;
}
bool lower_reshape_reshape_eltwise(GraphPtr g)
{
    GraphMatcherPattern pattern = {
        {{"0", OpType_Reshape}, {"1", OpType_Reshape}, {"2", OpType_Eltwise}},
        {{"0", "2"}, {"1", "2"}}};
    GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
    auto matched = matcher.match(g);
    bool is_changed = false;
    while (matched.size())
    {
        bool is_fused = false;
        for (auto &&m : matched)
        {
            if (!criteria(m))
            {
                continue;
            }
            /**
             *  when reshape input is different:
             * reshape---\              -----\
             *           |--eltwise ==>      |--eltwise---reshape
             * reshape---/              -----/ 
             * 
             * when two reshape input are different dim:
             * 4D>5D>smaller
             * reshape0---\                        -----\
             *            |--eltwise ==>                 |--eltwise---reshape2
             * reshape1---/                reshape1-----/ 
             */
            auto reshape0 = m[0];
            auto reshape1 = m[1];
            auto eltwise = m[2];

            auto elt_out = eltwise->outputs()[0];
            auto reshape0_out = reshape0->outputs()[0];
            auto reshape0_inp = reshape0->inputs()[0];
            auto reshape1_out = reshape1->outputs()[0];
            auto reshape1_inp = reshape1->inputs()[0];

            if (reshape0->inputs()[0]->shape() == reshape1->inputs()[0]->shape())
            {
                eltwise->add_input(reshape0->inputs()[0], eltwise->remove_input(reshape0->outputs()[0]));
                eltwise->add_input(reshape1->inputs()[0], eltwise->remove_input(reshape1->outputs()[0]));
                auto elt_out_new = std::make_shared<Tensor>(g->get_valid_tensor_name(elt_out->name()), reshape0_inp->shape(), elt_out->type());
                reshape0->clear_inputs();
                reshape1->clear_inputs();
                reshape0->clear_outputs();
                reshape1->clear_outputs();

                eltwise->clear_outputs();

                elt_out_new->quantization() = elt_out->quantization();
                eltwise->add_output(elt_out_new);
                reshape0->add_input(elt_out_new);
                reshape0->add_output(elt_out);
                g->remove_node(reshape1);
            }
            else
            {
                NodePtr slower_reshape = reshape0;
                NodePtr hold_reshape = reshape1;
                if (reshape0_inp->shape().dim() != reshape1_inp->shape().dim())
                {
                    std::map<int, NodePtr> choice = {
                        {reshape0_inp->shape().dim(), reshape0},
                        {reshape1_inp->shape().dim(), reshape1},
                    };
                    if (choice.count(4))
                    {
                        slower_reshape = choice.at(4);
                    }
                    else if (choice.count(5))
                    {
                        slower_reshape = choice.at(5);
                    }
                    else if (reshape0_inp->shape().dim() < reshape1_inp->shape().dim())
                    {
                        slower_reshape = choice.at(reshape0_inp->shape().dim());
                    }
                    else
                    {
                        slower_reshape = choice.at(reshape1_inp->shape().dim());
                    }
                    hold_reshape = choice.at(reshape0_inp->shape().dim());
                    if (hold_reshape == slower_reshape)
                    {
                        hold_reshape = choice.at(reshape1_inp->shape().dim());
                    }
                }

                auto slower_reshape_inp = slower_reshape->inputs()[0];
                auto slower_reshape_out = std::make_shared<Tensor>(g->get_valid_tensor_name(elt_out->name()), slower_reshape_inp->shape(), elt_out->type());
                slower_reshape_out->quantization() = elt_out->quantization();
                eltwise->add_input(slower_reshape->inputs()[0], eltwise->remove_input(slower_reshape->outputs()[0]));
                slower_reshape->clear_inputs();
                slower_reshape->clear_outputs();
                eltwise->clear_outputs();
                eltwise->add_output(slower_reshape_out);
                hold_reshape->outputs()[0]->shape() = slower_reshape_inp->shape();
                hold_reshape->params()["shape"] = slower_reshape_inp->shape();
                if (hold_reshape->params().count("ds_output_shape"))
                {
                    hold_reshape->params()["ds_output_shape"] =
                        std::vector<std::vector<tongue::Expr>>{{slower_reshape_inp->shape().symbol()}};
                }
                slower_reshape->add_input(slower_reshape_out);
                slower_reshape->add_output(elt_out);
            }

            is_changed = true;
            is_fused = true;
        }
        if (!is_fused)
        {
            break;
        }
        matched = matcher.match(g);
    }
    return is_changed;
}
}; // namespace aipubt
