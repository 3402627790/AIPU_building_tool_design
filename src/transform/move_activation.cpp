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
 * @Author: Chao Liu
 * @Date: 2022-05-27 16:25:43
 * @Last Modified by: Chao Liu
 * @Last Modified time: 2022-05-27 15:14:43
 */
#include "core/core.h"
#include "core/ops/ops.h"
#include "core/compute_tensor.h"
#include "passes.h"
#include <algorithm>
#include <set>
namespace aipubt
{

static inline bool creteria(const std::vector<NodePtr> &nodes)
{
    auto trans = nodes[0];//transpose
    auto activation = (nodes.size()==2)? nodes[1]: nodes[2];//activation
    if(activation->outputs()[0]->is_io_tensor())
        return false;
    std::vector<int> perm = trans->params()["perm"];
    std::vector<int> const_perm={0,1,3,2};
    if (perm != const_perm)
    {
        return false;
    }
    return true;
}

bool move_activation(GraphPtr g)
{
    std::vector<GraphMatcherPattern> patterns = {
        {{{"0", OpType_Transpose}, {"1", OpType_Activation}}, {{"0", "1"}}},
        {{{"0", OpType_Transpose}, {"1", OpType_Reshape},{"2",OpType_Activation}}, {{"0", "1"}, {"1","2"}}},
    };
    bool is_changed = false;
    for(auto &pattern : patterns)
    {
        GraphMatcher matcher(pattern);
        matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
        auto matched = matcher.match(g);
        while (matched.size())
        {
            bool is_fused = false;
            for (auto &&m : matched)
            {
                if (!creteria(m))
                {
                    continue;
                }
                if(m.size()==2)
                {
                    auto trans = m[0];
                    auto activation = m[1];
                    auto last = m[0]->parents()[0];
                    auto next = m[1]->children()[0];
                    activation->remove_input(0);
                    activation->add_input(last->outputs()[0]);
                    auto act_out = std::make_shared<Tensor>(g->get_valid_tensor_name("activation"), activation->inputs()[0]->shape(),activation->outputs()[0]->type());
                    activation->remove_output(0);
                    activation->add_output(act_out);

                    trans->remove_input(0);
                    trans->add_input(activation->outputs()[0]);
                    trans->remove_output(0);
                    trans->add_output(next->inputs()[0]);
                    is_changed = true;
                    is_fused = true;
                }
                else
                {
                    auto trans = m[0];
                    auto reshape = m[1];
                    auto activation = m[2];

                    auto last = trans->parents()[0];
                    auto next = activation->children()[0];

                    activation->remove_input(0);
                    activation->add_input(last->outputs()[0]);
                    auto act_out = std::make_shared<Tensor>(g->get_valid_tensor_name("activation"), activation->inputs()[0]->shape(),activation->outputs()[0]->type());
                    activation->remove_output(0);
                    activation->add_output(act_out);

                    trans->remove_input(0);
                    trans->add_input(activation->outputs()[0]);

                    reshape->remove_output(0);
                    reshape->add_output(next->inputs()[0]);

                    is_changed = true;
                    is_fused = true;
                }
            }
            if (!is_fused)
            {
                break;
            }
        matched = matcher.match(g);
        }
    }
    return is_changed;
}

}