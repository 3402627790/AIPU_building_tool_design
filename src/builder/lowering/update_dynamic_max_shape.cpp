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
 * @Author: Lang Hu
 * @Date: 2023-12-20 09:56:04
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2023-12-20 09:56:04
 */
#include "builder/builder.h"
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/core.h"

namespace aipubt
{
void update_dynamic_max_shape(GraphPtr g)
{
    if (BuilderConfig::exist(Constants::max_input_shape))
    {
        auto in_shapes = parse_input_shape(g, Constants::max_input_shape);
        auto in_tensors = g->get_input_tensors();
        for (uint32_t i = 0; i < in_tensors.size(); ++i)
        {
            in_tensors[i]->shape() = TensorShape(in_shapes[i], in_tensors[i]->shape().layout());
        }
        auto nodes = g->nodes();
        for (auto n : nodes)
        {
            if (n->type() == OpType_Resize &&
                (n->inputs()[0]->shape().dim() == 4 || n->inputs()[0]->shape().dim() == 5))
            {
                if (!n->params().count("ratio_x"))
                {
                    float ratio_x = (float)n->outputs()[0]->shape().w() / (float)n->inputs()[0]->shape().w();
                    n->params()["ratio_x"] = ratio_x;
                }
                if (!n->params().count("ratio_y"))
                {
                    float ratio_y = (float)n->outputs()[0]->shape().h() / (float)n->inputs()[0]->shape().h();
                    n->params()["ratio_y"] = ratio_y;
                }
            }
        }
        for (auto n : nodes)
        {
            if (n->type() == OpType_Input)
            {
                continue;
            }
            auto shapes = n->infer_shape();
            if (shapes.size() != n->outputs().size())
            {
                WARN() << "Failed to infer shape for node: " << n->name() << "(" << n->type() << ")";
                continue;
            }
            for (uint32_t i = 0; i < n->outputs().size(); ++i)
            {
                n->outputs()[i]->shape() = shapes[i];
            }
        }
    }
}
} // namespace aipubt
