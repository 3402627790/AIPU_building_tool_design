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
 * @Author: Samuel Han
 * @Date: 2022-10-28 15:59:56
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-04-21 14:28:25
 */

#include "core/core.h"
#include "core/location.h"
#include "core/ops/ops.h"
#include "passes.h"
#include "quantization_transform/common.h"

namespace qtlibn
{
using namespace aipubt;

// case 1:
//    dequantize   dequantize
//          \          /
//             eltwise
//                |
//             activation
//                |
//             quantize
//
// case 2:
//    dequantize   dequantize
//          \          /
//             eltwise
//                |
//             activation
//            /     |
//     quantize     quantize
//

static bool criteria(std::vector<aipubt::NodePtr> &ns)
{
    auto act = ns[3];
    std::string method = act->params()["method"];
    std::set<std::string> supoprted_m = {
        "RELU",
        "RELU6"};
    if (!supoprted_m.count(method))
    {
        return false;
    }
    if (ns.size() == 6)
    {
        auto q_0 = ns[4];
        auto q_1 = ns[5];
        if (q_0->outputs()[0]->quantization() != q_1->outputs()[0]->quantization())
        {
            return false;
        }
    }

    return true;
}

void fuse_dq_n_fp_op_q(aipubt::GraphPtr g)
{
    std::vector<aipubt::GraphMatcherPattern> patterns = {
        {{{"0", aipubt::OpType_DeQuantize}, {"1", aipubt::OpType_DeQuantize}, {"2", aipubt::OpType_Eltwise}, {"3", aipubt::OpType_Activation}, {"4", aipubt::OpType_Quantize}},
         {{"0", "2"}, {"1", "2"}, {"2", "3"}, {"3", "4"}}},
        {{{"0", aipubt::OpType_DeQuantize}, {"1", aipubt::OpType_DeQuantize}, {"2", aipubt::OpType_Eltwise}, {"3", aipubt::OpType_Activation}, {"4", aipubt::OpType_Quantize}, {"5", aipubt::OpType_Quantize}},
         {{"0", "2"}, {"1", "2"}, {"2", "3"}, {"3", "4"}, {"3", "5"}}},
        {{{"0", aipubt::OpType_DeQuantize}, {"1", aipubt::OpType_DeQuantize}, {"2", aipubt::OpType_Add}, {"3", aipubt::OpType_Activation}, {"4", aipubt::OpType_Quantize}},
         {{"0", "2"}, {"1", "2"}, {"2", "3"}, {"3", "4"}}},
        {{{"0", aipubt::OpType_DeQuantize}, {"1", aipubt::OpType_DeQuantize}, {"2", aipubt::OpType_Add}, {"3", aipubt::OpType_Activation}, {"4", aipubt::OpType_Quantize}, {"5", aipubt::OpType_Quantize}},
         {{"0", "2"}, {"1", "2"}, {"2", "3"}, {"3", "4"}, {"3", "5"}}},
    };
    for (auto &pattern : patterns)
    {
        aipubt::GraphMatcher matcher(pattern);
        matcher.set_mode(aipubt::GraphMatcher::MatchMode::StrictMode);
        auto matched = matcher.match(g);
        for (auto &&m : matched)
        {
            if (!criteria(m))
            {
                continue;
            }
            auto elt = m[2];
            auto dq_0 = elt->inputs()[0]->node();
            auto dq_1 = elt->inputs()[1]->node();
            auto act = m[3];
            auto q_0 = m[4];

            auto elt_inputs = elt->inputs();
            elt->replace_input_by(elt_inputs[0], dq_0->inputs()[0]);
            elt->replace_input_by(elt_inputs[1], dq_1->inputs()[0]);
            elt->replace_output_by(elt->outputs()[0], q_0->outputs()[0]);
            if (m.size() == 6)
            {
                auto q_1 = m[5];
                for (auto c : q_1->outputs()[0]->consumers())
                {
                    c->replace_input_by(q_1->outputs()[0], elt->outputs()[0]);
                }
                g->remove_node(q_1);
            }

            std::string method = act->params()["method"];
            elt->params()["with_activation"] = method;
            g->remove_node(dq_0);
            g->remove_node(dq_1);
            g->remove_node(act);
            g->remove_node(q_0);
        }
    }
    return;
}

} // namespace qtlibn