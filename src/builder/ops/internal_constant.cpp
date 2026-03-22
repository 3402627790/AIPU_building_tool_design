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
 * @Author: Samule Han
 * @Date: 2020-03-13 17:15:02
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2021-11-03 14:39:47
 */

#include "builder/mem_remapping.h"
#include "builder/optimizer.h"
#include "builder/param_check.h"
#include "builder/plugin.h"
#include "builder/ro_generator.h"
#include "builder/target.h"
#include "builder/text_generator.h"
#include "core/compute_tensor.h"
#include "core/constants.h"
#include "core/logger.h"
#include <set>
#include <vector>
namespace aipubt
{

RO_GENERATOR(OpType_InternalConstant)
{
}
RO_GENERATOR_END(OpType_InternalConstant);

TEXT_GENERATOR(OpType_InternalConstant)
{
    //do layout convertor here
    if (node->attrs().count("_const_layout_cvt") && node->get_attr<bool>("_const_layout_cvt"))
    {
        return "";
    }
    auto out = node->outputs()[0];
    ComputeTensor weight = node->constants()["weights"];
    auto olo = out->shape().layout();
    if (olo != DataLayout_Flat && olo != DataLayout_NHWC)
    {
        if (weight->shape().dim() == 4)
        {
            std::vector<int> shape = weight->shape();
            auto c = shape[3];
            if (olo == DataLayout_NCHWC32)
            {
                shape[3] = c / 32;
                shape.push_back(32);
                weight = weight.reshape(shape).permute({0, 3, 1, 2, 4});
                node->constants()["weights"]->shape() = weight->shape();
                node->constants()["weights"]->share_data_from(weight);

                node->set_attr("_const_layout_cvt", true);
            }
            else if (olo == DataLayout_NCHWC16)
            {
                shape[3] = c / 16;
                shape.push_back(16);
                weight = weight.reshape(shape).permute({0, 3, 1, 2, 4});
                node->constants()["weights"]->shape() = weight->shape();
                node->constants()["weights"]->share_data_from(weight);
                node->set_attr("_const_layout_cvt", true);
            }
        }
    }
    return "";
}
TEXT_GENERATOR_END(OpType_InternalConstant);

MEMORY_REMAPPING_OP(OpType_InternalConstant)
{
    // make output shared weight's memory
    auto out = node->outputs()[0];
    auto weight = node->constants()["weights"];
    out->memory_offset()->set_base_offset(weight->memory_offset());
    out->memory_offset()->relative_offset = 0;
}
MEMORY_REMAPPING_OP_END(OpType_InternalConstant);
}; // namespace aipubt
