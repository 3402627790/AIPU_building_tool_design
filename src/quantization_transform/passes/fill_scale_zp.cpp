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
 * @Date: 2022-03-07 18:57:23
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-12-09 13:51:49
 */

#include "passes.h"
#include <core/compute_tensor.h>

namespace qtlibn
{
using namespace aipubt;
// fill scale/zp to get the same size.
void pad_scale_zp(aipubt::GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto &node : nodes)
    {
        for (auto ct : node->constants())
        {
            std::vector<int32_t> zps = ct.second->quantization().offsets();
            std::vector<float> scales = ct.second->quantization().scales();
            auto scale_num = scales.size();
            auto zp_num = zps.size();
            if (scale_num < zp_num)
            {
                auto num = zp_num - scale_num;
                std::vector<float> fill(num, scales[0]);
                scales = ComputeTensor::concat({scales, fill});
                ct.second->quantization().set_scales(scales);
                DEBUG() << "In node: " << node->name() << ", and optype = " << aipubt::optype_to_string(node->type())
                        << ", the tensor " << ct.second->name() << " scale_num < zp_num, qtlib will use the scale[0] to fill the scales.";
            }
            else if (scale_num > zp_num)
            {
                auto num = scale_num - zp_num;
                std::vector<int32_t> fill(num, zps[0]);
                zps = ComputeTensor::concat({zps, fill});
                ct.second->quantization().set_offsets(zps);
                DEBUG() << "In node: " << node->name() << ", and optype = " << aipubt::optype_to_string(node->type())
                        << ", the tensor " << ct.second->name() << " scale_num > zp_num, qtlib will use the zp[0] to fill the offsets.";
            }

            if (node->type() == aipubt::OpType_DepthwiseConv)
            {
                int32_t scale_num = ct.second->quantization().scales().size();
                int32_t num_output = node->params()["num_output"];
                if (scale_num == 1 && scale_num != num_output)
                {
                    auto scales = ct.second->quantization().scales();
                    auto offsets = ct.second->quantization().offsets();
                    std::vector<float> filled_scales(num_output, scales[0]);
                    std::vector<int32_t> filled_offsets(num_output, offsets[0]);
                    ct.second->quantization().set_scales(filled_scales);
                    ct.second->quantization().set_offsets(filled_offsets);
                }
            }
        }
    }
}
} // namespace qtlibn
