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
 * @Date: 2022-05-04 16:25:43
 * @Last Modified by: Chao Liu
 * @Last Modified time: 2022-05-16 15:14:43
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
    auto n1 = nodes[0];//transpose0
    auto n2 = (nodes.size()==2)? nodes[1]: nodes[2];//transpose1
    if(n2->outputs()[0]->is_io_tensor())
        return false;
    std::vector<int> perm0 = n1->params()["perm"];
    std::vector<int> perm1 = n2->params()["perm"];
    std::vector<unsigned int> perm;
    if(perm0.size() != perm1.size())
        return false;
    for (size_t i = 0; i < perm1.size(); i++)
    {
        perm.push_back((unsigned int)(perm0[perm1[i]]));
    }
    for (size_t i = 0; i < perm.size(); i++)
    {
        if(i!=perm[i])
            return false;
    }
    if(n1->inputs()[0]->shape() != n2->outputs()[0]->shape())
    {
        return false;
    }
    return true ;
}


bool eliminate_transpose_transpose(GraphPtr g)
{
    std::vector<GraphMatcherPattern> patterns = {
        {{{"0", OpType_Transpose}, {"1", OpType_Transpose}}, {{"0", "1"}}},
        {{{"0", OpType_Transpose}, {"1", OpType_Activation},{"2",OpType_Transpose}}, {{"0", "1"}, {"1","2"}}},
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
                    //eliminate transpose0 and transpose1
                    auto last = m[0]->parents()[0];
                    for (auto child : m[1]->children())
                    {
                        child->replace_input_by(m[1]->outputs()[0], m[0]->inputs()[0]);
                    }
                    m[0]->remove_input(0);
                    m[1]->remove_output(0);
                    g->remove_node(m[0]);
                    g->remove_node(m[1]);
                    is_changed = true;
                    is_fused = true;
                }
                else
                {
                    //transpose + activation + transpose for codeformer
                    //transpose + activation + transpose =>new activation
                    auto last = m[0]->parents()[0];
                    auto activation = m[1];
                    activation->remove_input(0);
                    activation->add_input(last->outputs()[0]);
                    auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("activation"), activation->inputs()[0]->shape(),activation->outputs()[0]->type());
                    activation->remove_output(0);
                    activation->add_output(output);
                    for (auto child : m[2]->children())
                    {
                        child->replace_input_by(m[2]->outputs()[0], activation->outputs()[0]);
                    }
                    g->remove_node(m[0]);
                    g->remove_node(m[2]);
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