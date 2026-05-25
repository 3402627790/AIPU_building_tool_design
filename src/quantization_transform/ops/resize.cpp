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
 * @Date: 2022-04-06 17:06:35
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-10-24 22:31:25
 */

#include "quantization_transform/ops/resize.h"
#include "quantization_transform/ops/no_quantization.h"

namespace qtlibn
{
void ResizeOp::quantize(aipubt::NodePtr node)
{
    auto inp_t = node->inputs()[0];
    auto out_t = node->outputs()[0];
    NoQuantization::quantize_transparent(out_t, inp_t);

    if (node->attrs().count("resize_degrade_to_nearest") && node->get_attr("resize_degrade_to_nearest"))
    {
        node->params()["method"] = "NEAREST";
    }
    int32_t interp_shift = 0;
    if ((std::string)node->params()["method"] == "BILINEAR")
    {
        interp_shift = 13;
    }
    node->params()["interp_shift_value"] = interp_shift;
    node->params()["interp_shift_type"] = aipubt::Tensor::INT8;
}

REGISTER_QUANTIZATION_OP(Resize);

}; // namespace qtlibn
