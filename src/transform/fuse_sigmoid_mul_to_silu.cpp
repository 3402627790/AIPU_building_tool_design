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
#include "quantization_transform/generate_lut.h"

namespace aipubt
{

static inline bool criteria(NodePtr activation, NodePtr eltwise)
{
    if ((std::string)activation->params()["method"] != "SIGMOID" ||
        activation->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    if (eltwise->type() == OpType_Eltwise && eltwise->params()["method"] != "MUL")
    {
        return false;
    }
    bool ret = std::any_of(eltwise->inputs().begin(), eltwise->inputs().end(), [activation](TensorPtr t) { return t == activation->inputs()[0]; });
    if (!ret)
    {
        return false;
    }
    return true;
}

bool fuse_sigmoid_mul_to_silu(GraphPtr g)
{
    std::vector<GraphMatcherPattern> patterns = {
        {{{"0", OpType_Activation}, {"1", OpType_Mul}}, {{"0", "1"}}},
        {{{"0", OpType_Activation}, {"1", OpType_Eltwise}}, {{"0", "1"}}},
    };
    bool is_changed = false;
    for (auto &pattern : patterns)
    {
        aipubt::GraphMatcher matcher(pattern);
        matcher.set_mode(aipubt::GraphMatcher::MatchMode::StrictMode);
        auto matched = matcher.match(g);
        while (matched.size())
        {
            bool is_merged = false;
            for (auto &&m : matched)
            {
                if (!criteria(m[0], m[1]))
                {
                    continue;
                }
                if (m[1]->outputs()[0]->quantization().scales().size() != 0 &&
                    m[1]->outputs()[0]->quantization().offsets().size() != 0)
                {
                    m[0]->inputs()[0]->quantization().set_quantized(true);
                    m[1]->outputs()[0]->quantization().set_quantized(true);
                }
                auto silu_t = ops::silu(m[0]->inputs()[0], m[1]->outputs()[0]->quantization());
                auto silu_n = silu_t->node();
                silu_n->clear_outputs();
                silu_n->add_output(m[1]->outputs()[0]);
                g->add_node(silu_n);
                ADD_LOCATION_BY_NODE(silu_n, m[0], m[1]);
                m[0]->clear_inputs();
                m[0]->clear_outputs();
                m[1]->clear_inputs();
                m[1]->clear_outputs();
                g->remove_node(m[0]);
                g->remove_node(m[1]);
                is_changed = true;
                is_merged = true;
            }
            if (!is_merged)
            {
                break;
            }
            matched = matcher.match(g);
        }
    }

    return is_changed;
}
}; // namespace aipubt