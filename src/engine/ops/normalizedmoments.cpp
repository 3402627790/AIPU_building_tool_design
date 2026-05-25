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
 * @Date: 2023-10-25 22:02:48
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-10-25 22:03:15
 */

#include "core/compute_tensor.h"
#include "core/node.h"
#include "engine/engine.h"
namespace aipubt
{
namespace engine
{
    static void normalizedmoments_forward(NodePtr node)
    {
        ComputeTensor mean_t = node->inputs()[0];
        ComputeTensor var_t = node->inputs()[1];
        ComputeTensor shift_t = node->inputs()[2];
        int32_t counts = node->params()["counts"];

        auto shifted_mean = mean_t.astype(Tensor::FP32) / counts;
        auto mean_out = shifted_mean + shift_t.astype(Tensor::FP32);
        auto mean_square = shifted_mean * shifted_mean;
        auto var_out = var_t.astype(Tensor::FP32) / counts - mean_square;

        if (node->placeholders().size() == 0)
        {
            node->placeholders().push_back(mean_square);
        }
        else
        {
            node->placeholders()[0] = mean_square;
        }
        node->outputs()[0]->share_data_from(mean_out);
        node->outputs()[1]->share_data_from(var_out);
    }
    EXEOP_REGISTER(OpType_NormalizedMoments, normalizedmoments_forward, DTCriteria::FP_ONLY);
} // namespace engine

} // namespace aipubt
