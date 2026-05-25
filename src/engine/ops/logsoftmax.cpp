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
 * @Author: Kejia Qiang
 * @Date: 2023-03-20 16:18:46
 * @Last Modified by: Kejia Qiang
 * @Last Modified time: 2023-03-21 11:06:49
 */
#include "core/compute_tensor.h"
#include "core/node.h"
#include "engine/engine.h"
namespace aipubt
{
namespace engine
{
    static void logsoftmax_forward(NodePtr node)
    {
        ComputeTensor inp = node->inputs()[0];
        int axis = node->params()["axis"];
        auto exp_inp = (inp - inp->max<float_t>()).exp();
        auto axis_sum = exp_inp.sum({axis}, true).astype(inp->type());
        ComputeTensor softmax = exp_inp / axis_sum;
        auto logsoftmax = softmax.log().astype(node->outputs()[0]->type());
        node->outputs()[0]->share_data_from(logsoftmax);
        if (node->placeholders().size() == 0)
        {
            node->placeholders().push_back(softmax);
            node->placeholders().push_back(logsoftmax);
        }
    }
    EXEOP_REGISTER(OpType_LogSoftmax, logsoftmax_forward, DTCriteria::FP_ONLY);
} // namespace engine

} // namespace aipubt