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
 * @Author: Neo Wang 
 * @Date: 2023-03-20 16:18:46 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-03-20 17:13:57
 */
#include "core/compute_tensor.h"
#include "core/node.h"
#include "engine/engine.h"
namespace aipubt
{
namespace engine
{

    EXEOP_REGISTER(
        OpType_Activation, [](NodePtr node) {
                ComputeTensor input = node->inputs()[0];
                ComputeTensor output = input * input.sigmoid();
                output = output.astype(node->outputs()[0]->type());
                node->outputs()[0]->share_data_from(output); },
        DTCriteria::FP_ONLY,
        [](NodePtr node) -> bool {
            return node->params()["method"] == "SILU";
        });
} // namespace engine

} // namespace aipubt