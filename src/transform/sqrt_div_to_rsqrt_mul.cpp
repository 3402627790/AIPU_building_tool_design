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
 * @Date: 2022-05-16 11:17:29
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-05-16 16:35:11
 */

#include "core/core.h"
#include "core/ops/ops.h"

namespace aipubt
{

static bool criteria(std::vector<NodePtr> nodes)
{
    std::vector<Tensor::Dtype> _FLOAT_DTYPE = {Tensor::BFP16, Tensor::FP16, Tensor::FP32, Tensor::FP64};
    for (auto n : nodes)
    {
        for (auto ot : n->outputs())
        {
            // even if use the original tensor in sqrt or div, the meaning of tensor has changed.
            // so we need to check this.
            if (ot->is_io_tensor())
            {
                return false;
            }

            // TODO: when OpsAPI can quantize the aipu_scale/zp, then we can generate the rsqrt lut;
            // and we will support Quant CompassIR
            // now only support Float CompassIR
            if (!contains(_FLOAT_DTYPE, ot->type()))
            {
                return false;
            }
        }
    }
    return true;
}

bool sqrt_div_to_rsqrt_mul(GraphPtr g)
{
    // div(x, sqrt(y)) => mul(x, rsqrt(y))
    std::vector<GraphMatcherPattern> patterns = {
        {{{"0", OpType_Sqrt}, {"1", OpType_Tile}, {"2", OpType_Div}}, {{"0", "1"}, {"1", "2"}}},
        {{{"0", OpType_Sqrt}, {"1", OpType_Div}}, {{"0", "1"}}},
    };

    bool is_changed = false;
    for (auto &pattern : patterns)
    {
        GraphMatcher matcher(pattern);
        matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
        auto matched = matcher.match(g);
        while (matched.size())
        {
            bool is_merged = false;
            for (auto &&m : matched)
            {
                if (!criteria(m))
                {
                    continue;
                }
                auto m_size = m.size();

                auto sqrt_inp_t = m[0]->inputs()[0];
                auto sqrt_out_t = m[0]->outputs()[0];
                auto div_inp_ts = m[m_size - 1]->inputs();
                auto div_out_t = m[m_size - 1]->outputs();
                bool unquantifiable = sqrt_inp_t->quantization().unquantifiable();
                sqrt_inp_t->quantization().set_unquantifiable(true);
                auto rsqrt_ot = ops::rsqrt(sqrt_inp_t);
                sqrt_inp_t->quantization().set_unquantifiable(unquantifiable);

                auto rsqrt_node = rsqrt_ot->node();
                rsqrt_node->clear_outputs();
                rsqrt_node->add_output(sqrt_out_t);

                auto mul_ot = ops::mul(div_inp_ts[0], div_inp_ts[1]);
                auto mul_node = mul_ot->node();
                mul_node->clear_outputs();
                mul_node->add_output(div_out_t);

                g->add_node(rsqrt_node);
                g->add_node(mul_node);
                ADD_LOCATION_BY_NODE(rsqrt_node, m[0], m[m_size - 1]);
                ADD_LOCATION_BY_NODE(mul_node, m[0], m[m_size - 1]);

                m[0]->clear_inputs();
                m[0]->clear_outputs();
                m[m_size - 1]->clear_inputs();
                m[m_size - 1]->clear_outputs();
                g->remove_node(m[0]);
                g->remove_node(m[m_size - 1]);

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
} // namespace aipubt
