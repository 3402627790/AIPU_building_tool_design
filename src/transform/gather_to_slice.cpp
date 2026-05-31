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
 * @Author: Xiaolong Zhang
 * @Date: 2023-06-2 14:15:49
 * @Last Modified by: Xialong Zhang
 * @Last Modified time: 2023-06-2 14:24:36
 */

#include "core/compute_tensor.h"
#include "core/config.h"
#include "core/core.h"
#include "core/ops/ops.h"

namespace aipubt
{

// align with parser: gather -> slice -> crop
static inline bool criteria(NodePtr constant, NodePtr gather)
{
    if (gather->inputs().size() != 2)
    {
        return false;
    }
    if (gather->params().count("batch_dims") != 0 && gather->params()["batch_dims"] != 0)
    {
        return false;
    }
    if (constant->outputs()[0]->size() != 1)
    {
        return false;
    }
    return true;
}

bool gather_to_slice(GraphPtr g)
{
    static aipubt::GraphMatcherPattern pattern = {
        {{"0", aipubt::OpType_Constant}, {"1", aipubt::OpType_Gather}},
        {{"0", "1"}},
    };
    aipubt::GraphMatcher matcher(pattern);
    matcher.set_mode(aipubt::GraphMatcher::MatchMode::StrictMode);
    auto matched = matcher.match(g);

    bool is_changed = false;
    for (auto &&m : matched)
    {
        if (!criteria(m[0], m[1]))
        {
            continue;
        }

        int inp_rank = m[1]->inputs()[0]->shape().dim();
        int axis = m[1]->params()["axis"];
        if (axis < 0)
        {
            axis += inp_rank;
        }

        ComputeTensor c(m[0]->constants()["weights"]);
        c = (c + m[0]->outputs()[0]->quantization().offset()) * m[0]->outputs()[0]->quantization().scale();
        c = c.astype(Tensor::INT32);
        TensorPtr new_c = c;
        int indice = new_c->data<int>()[0];

        std::vector<int> begin(inp_rank, 0);
        std::vector<int> stride(inp_rank, 1);
        std::vector<int> end(m[1]->inputs()[0]->shape());
        begin[axis] = indice;
        end[axis] = indice + 1;
        auto slice_t = ops::slice(m[1]->inputs()[0], begin, end, stride);
        NodePtr output_n = slice_t->node();
        ADD_LOCATION_BY_NODE(slice_t->node(), m[0], m[1]);
        if (m[0]->outputs()[0]->shape().dim() != 1)
        {
            auto reshape_t = ops::reshape(slice_t, TensorShape(m[1]->outputs()[0]->shape()));
            auto reshape_n = reshape_t->node();
            output_n = reshape_n;
            ADD_LOCATION_BY_NODE(reshape_n, m[0], m[1]);
        }
        output_n->clear_outputs();
        output_n->add_output(m[1]->outputs()[0]);

        m[0]->clear_outputs();
        m[1]->clear_inputs();
        m[1]->clear_outputs();
        g->remove_node(m[0]);
        g->remove_node(m[1]);

        is_changed = true;
    }
    return is_changed;
}
}; // namespace aipubt