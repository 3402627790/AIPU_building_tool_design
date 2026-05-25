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
 * @Date: 2023-04-17 12:59:04
 * @Last Modified by: Kejia Qiang
 * @Last Modified time: 2023-04-17 17:57:44
 */

#include "core/compute_tensor.h"
#include "core/node.h"
#include "engine/engine.h"
#include "quantization_transform/utils/data_type.h"

namespace aipubt
{
namespace engine
{
    static void pow_forward(NodePtr node)
    {
        ComputeTensor base = node->inputs()[0];
        ComputeTensor power = node->inputs()[1];
        ComputeTensor base_abs = base.abs() + ComputeTensor(1e-6f);
        auto power_fp = power.astype(node->inputs()[0]->type());
        auto log_inp = base_abs.log();
        auto log_oup = log_inp * power_fp;
        auto exp_oup = log_oup.exp();
        node->placeholders().push_back(log_inp);
        node->placeholders().push_back(log_oup);
        node->placeholders().push_back(exp_oup);
        auto out = base.pow(power_fp);
        node->outputs()[0]->set_type(out->type());
        node->outputs()[0]->share_data_from(out);
    }

    EXEOP_REGISTER(OpType_Pow,
                   pow_forward,
                   DTCriteria::FP_ONLY);
} // namespace engine
} // namespace aipubt