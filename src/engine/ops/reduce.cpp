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
 * @Date: 2023-04-17 12:59:04
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-04-17 17:57:44
 */

#include "core/compute_tensor.h"
#include "core/node.h"
#include "engine/engine.h"

namespace aipubt
{
namespace engine
{

    static void reduce_l2_forward(NodePtr node)
    {
        // even if we donot consider the in_dtype, but it is better to forward float32/64
        auto in_type = node->inputs()[0]->type();
        if (!Tensor::is_float(in_type))
        {
            WARN() << "using input_dtype = " << in_type << " maybe cause overflow or miss match after setting dtype.";
        }
        std::vector<int32_t> axes = node->params()["axis"];
        auto inp = ComputeTensor::create(node->inputs()[0]);
        auto pow2 = inp * inp;

        auto sum = pow2.sum(axes);
        auto out = sum.sqrt().astype(in_type);

        node->placeholders().push_back(sum);
        node->outputs()[0]->set_type(in_type);
        node->outputs()[0]->share_data_from(*out);
    }

    EXEOP_REGISTER(OpType_Reduce,
                   reduce_l2_forward,
                   DTCriteria::FP_ONLY,
                   [](NodePtr node) -> bool { return node->params()["method"] == "L2"; });
} // namespace engine

} // namespace aipubt
