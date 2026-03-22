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
 * @Author: Barney Zhao
 * @Date: 2022-11-03 10:24:00
 * @Last Modified by: Barney Zhao
 * @Last Modified time: 2022-11-03 11:30:00
 */
#include "builder/lowering/lowering.h"
#include "core/core.h"
#include <algorithm>
#include <builder/target.h>
namespace aipubt
{
void abs_to_leakyrelu(GraphPtr g)
{
    auto criterion = [](NodePtr node) -> bool
    {
        auto target = aipubt::get_target();
        aipubt::ISAType arch = target->isainfo().type;
        // only AIFF, not HWA
        if (arch < aipubt::ISA_ZYv2)
        {
            return false;
        }


        if (node->inputs().empty())
        {
            return false;
        }
        bool res = (OpType_Abs == node->type());
        return res;
    };

    auto m_nodes = g->nodes();
    for (auto &&n : m_nodes)
    {
        if (!criterion(n))
        {
            continue;
        }

        int negative_slope_value = -1;
        int scale_value = 1;
        auto scale_type = Tensor::UINT8;

        NodePtr leakyrelu_node = Node::create(n->name(), OpType_Activation);
        leakyrelu_node->params()["method"] = "LEAKYRELU";
        if (Tensor::is_float(n->inputs()[0]->type()))
        {
            leakyrelu_node->params()["negative_slope"] = static_cast<float>(negative_slope_value);
            leakyrelu_node->params()["negative_slope_value"] = static_cast<float>(negative_slope_value);
            leakyrelu_node->params()["negative_slope_type"] = Tensor::FP32;
        }
        else
        {
            leakyrelu_node->params()["negative_slope_type"] = Tensor::INT8;
            leakyrelu_node->params()["negative_slope_shift"] = 0;
            leakyrelu_node->params()["negative_slope_value"] = negative_slope_value;
            leakyrelu_node->params()["scale_type"] = scale_type;
            leakyrelu_node->params()["scale_value"] = scale_value;
            leakyrelu_node->params()["shift_type"] = Tensor::INT8;
            leakyrelu_node->params()["shift_value"] = 0;
        }
        // negative in aipuexe are both need func(x+zp)-zp
        // but abs in android still need quant params
        if (OpType_Abs == n->type())
        {
            if (n->params().count("scale_type"))
            {
                leakyrelu_node->params()["scale_type"] = n->params()["scale_type"];
            }
            if (n->params().count("scale_value"))
            {
                leakyrelu_node->params()["scale_value"] = n->params()["scale_value"];
            }
            if (n->params().count("shift_type"))
            {
                leakyrelu_node->params()["shift_type"] = n->params()["shift_type"];
            }
            if (n->params().count("shift_value"))
            {
                leakyrelu_node->params()["shift_value"] = n->params()["shift_value"];
            }
        }

        ADD_LOCATION_BY_NODE(leakyrelu_node, n);
        g->replace_node_by(n, leakyrelu_node);
    }
}
}; // namespace aipubt
