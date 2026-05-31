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
 * @Date: 2022-03-30 13:55:50
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-03-30 16:35:20
 */

#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"

namespace aipubt
{

static inline bool criteria(NodePtr n)
{
    if (n->type() == OpType_Cast && n->inputs()[0]->type() == n->outputs()[0]->type())
    {
        std::string to_dtype = n->params()["to_dtype"];
        if (Tensor::is_float(n->outputs()[0]->type()))
        {
            return true;
        }
        else
        {
            if (n->params().count("scale_value") && n->params().count("shift_value"))
            {
                auto scale_value = n->params()["scale_value"];
                auto shift_value = n->params()["shift_value"];
                float ts = static_cast<float>(scale_value) * pow(0.5, shift_value);
                if (ts != 1.0)
                {
                    return false;
                }
            }
            else if (n->constants().count("scale") && n->constants().count("shift"))
            {
                auto scale_ct = ComputeTensor(n->constants()["scale"]);
                auto shift_ct = ComputeTensor(n->constants()["shift"]);

                std::vector<int32_t> shift_shape = shift_ct->shape();
                auto base_pow = ComputeTensor::full(shift_shape, 0.5);
                std::vector<int32_t> ts = (scale_ct * base_pow.pow(shift_ct)).astype(Tensor::INT32);
                auto all_is_one = std::all_of(ts.begin(), ts.end(), [](int32_t z) { return z == 1.0 ? true : false; });
                if (!all_is_one)
                {
                    return false;
                }
            }

            if (n->inputs()[0]->quantization().scales().size() > 0 && n->outputs()[0]->quantization().scales().size() > 0)
            {
                //    auto in_scale = n->inputs()[0]->quantization().scales();
                //    auto ot_scale = n->outputs()[0]->quantization().scales();
                auto in_zerop = n->inputs()[0]->quantization().offsets();
                auto ot_zerop = n->outputs()[0]->quantization().offsets();
                if (in_zerop != ot_zerop)
                {
                    return false;
                }
            }
        }
    }
    else
    {
        return false;
    }

    return true;
}

bool eliminate_cast(GraphPtr g)
{
    bool is_changed = false;
    auto nodes = g->nodes();
    for (auto &&n : nodes)
    {
        if (!criteria(n))
        {
            continue;
        }
        auto reshape = Node::create(n->name(), OpType_Reshape);
        reshape->params()["shape"] = n->outputs()[0]->shape();
        ADD_LOCATION_BY_NODE(reshape, n);
        g->replace_node_by(n, reshape);
        is_changed = true;
    }
    return is_changed;
}

}; // namespace aipubt