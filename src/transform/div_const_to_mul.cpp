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
 * @Date: 2022-04-11 14:24:01
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-03-21 12:49:06
 */

#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include "quantization_transform/utils/quantize_utils.h"
#include <algorithm>

namespace aipubt
{

static bool criteria(NodePtr div, NodePtr cst)
{
    return Tensor::is_float(div->outputs()[0]->type()) &&
           Tensor::is_float(cst->outputs()[0]->type()) &&
           div->inputs()[1] == cst->outputs()[0] &&
           cst->constants()["weights"]->data();
}

bool div_const_to_mul(GraphPtr g)
{
    std::vector<GraphMatcherPattern> patterns = {
        {{{"0", OpType_Constant}, {"1", OpType_Div}}, {{"0", "1"}}},
    };

    bool is_changed = false;
    for (auto &pattern : patterns)
    {
        GraphMatcher matcher(pattern);
        matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
        auto matched = matcher.match(g);
        for (auto &&m : matched)
        {
            if (!criteria(m[1], m[0]))
            {
                continue;
            }
            auto cst = m[0];
            ComputeTensor data = cst->constants()["weights"];
            auto dt = data->type();
            data = 1.f / data;
            cst->constants()["weights"] = data.astype(dt);
            MODIFY_NODE(cst);
            auto mul = Node::create(m[1]->name(), OpType_Mul);
            g->replace_node_by(m[1], mul);
            ADD_LOCATION_BY_NODE(mul, m[1]);
            is_changed = true;
        }
    }

    return is_changed;
}

} // namespace aipubt
