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
 * @Date: 2022-10-20 17:11:37
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-05-23 10:56:17
 */

#include "quantization_transform/passes/passes.h"

namespace qtlibn
{

void convert_relu6_to_clip(aipubt::GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto &node : nodes)
    {

        if (node->params().count("with_activation") &&
            (std::string)node->params()["with_activation"] == "RELU6" &&
            (!aipubt::Tensor::is_float(node->outputs()[0]->type())))
        {
            float clip_max = 6.0;
            float clip_min = 0.0;
            DEBUG() << "qtlib will convert relu6 to clip to adpat android case.";
            int32_t q_out_bits = node->outputs()[0]->quantization().bits();
            float cscale = node->outputs()[0]->quantization().scale();
            int czerop = node->outputs()[0]->quantization().offset();
            if (q_out_bits <= 8)
            {
                cscale = cscale * 256;
                czerop = 0;
            }
            int32_t t_qmin = dtype_utils::bits_to_range_min(32, true);
            int32_t t_qmax = dtype_utils::bits_to_range_max(32, true);
            auto cmax = static_cast<int64_t>(quant_utils::linear_quantize(clip_max, cscale, czerop));
            auto cmin = static_cast<int64_t>(quant_utils::linear_quantize(clip_min, cscale, czerop));
            cmax = quant_utils::clip<int64_t, int32_t>(cmax, t_qmin, t_qmax);
            cmin = quant_utils::clip<int64_t, int32_t>(cmin, t_qmin, t_qmax);
            node->params()["clip_max"] = static_cast<int32_t>(cmax);
            node->params()["clip_min"] = static_cast<int32_t>(cmin);
            node->params()["with_activation"] = "CLIP";
        }
    }
}
} // namespace qtlibn
