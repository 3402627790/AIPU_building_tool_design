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
 * @Date: 2023-04-12 10:16:55
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-04-12 11:51:04
 */

#include "core/compute_tensor.h"
#include "core/node.h"
#include "engine/engine.h"
namespace aipubt
{
namespace engine
{
    static void squared_difference_forward(NodePtr node)
    {
        //only support float forward yet
        ComputeTensor in0_ct = node->inputs()[0];
        ComputeTensor in1_ct = node->inputs()[1];
        auto minus_output = in0_ct - in1_ct;
        auto output = minus_output.pow(2);
        node->placeholders().push_back(minus_output);
        output = output.astype(node->outputs()[0]->type());

        node->outputs()[0]->share_data_from(output);
    }

    EXEOP_REGISTER(OpType_SquaredDifference, squared_difference_forward, DTCriteria::FP_ONLY);
} // namespace engine

} // namespace aipubt
