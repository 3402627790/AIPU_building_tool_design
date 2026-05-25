//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2022 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Samuel Han
 * @Date: 2022-05-18 22:15:45
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-05-18 22:35:11
 */

#include "quantization_transform/ops/maxroipool.h"
#include "quantization_transform/ops/no_quantization.h"

namespace qtlibn
{
void MaxRoiPoolOp::quantize(aipubt::NodePtr node)
{
    auto inp0_t = node->inputs()[0];
    auto out_t = node->outputs()[0];
    NoQuantization::quantize_transparent(out_t, inp0_t);

    if (!node->attrs().count(Constants::params_quantized) || !node->get_attr(Constants::params_quantized))
    {
        float roi_scale = 1.f / node->inputs()[1]->quantization().scale();

        auto spatial_scale = param_utils::get_param(node, "spatial");
        float spatial_scale_y = spatial_scale[0];
        float spatial_scale_x = spatial_scale[1];
        spatial_scale_y = round(spatial_scale_y * roi_scale * 65535);
        spatial_scale_x = round(spatial_scale_x * roi_scale * 65535);
        node->params()["spatial"] = {static_cast<int32_t>(spatial_scale_y),
                                     static_cast<int32_t>(spatial_scale_x)};
    }
}

REGISTER_QUANTIZATION_OP(MaxRoiPool);

} // namespace qtlibn