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
 * @Author: Yuchou.Gan
 * @Date: 2020-02-25 17:58:05
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-08-21 16:40:57
 */
#include "builder/lowering/lowering.h"
#include "core/core.h"
#include <algorithm>
#include <vector>

#define ALIGN(x, y) ((x + y - 1) / y * y)
namespace aipubt
{

// this pass is for channel=16, we will record the original input c and output c for conv/fc
void record_original_param(GraphPtr g)
{
    auto m_nodes = g->nodes();
    for (auto &&n : m_nodes)
    {
        if (n->inputs().size() == 0 || n->outputs().size() == 0)
        {
            continue;
        }
        TensorShape in_shape = n->inputs()[0]->shape();
        TensorShape out_shape = n->outputs()[0]->shape();
        if (n->type() == OpType_Convolution || n->type() == OpType_ConvTranspose || n->type() == OpType_DepthwiseConv)
        {
            n->set_attr("original_input_c", in_shape.c());
            n->set_attr("original_output_c", out_shape.c());
            if (n->type() == OpType_DepthwiseConv)
            {
                n->set_attr("is_dwconv", true);
            }
            else
            {
                n->set_attr("is_dwconv", false);

                n->set_attr("per_channel_quant", false);
                if (!n->params().count("group"))
                {
                    n->params()["group"] = 1;
                }
                int group = n->params()["group"];
                if ((group == 1) && (n->constants().count("scale") || n->constants().count("shift")))
                {
                    n->set_attr("per_channel_quant", true);
                }
            }
        }
        if (n->type() == OpType_FullyConnected)
        {
            n->set_attr("per_channel_quant", false);
            if (n->constants().count("scale") || n->constants().count("shift"))
            {
                n->set_attr("per_channel_quant", true);
            }
        }
    }
}
}; // namespace aipubt
