//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2023 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Samuel Han
 * @Date: 2023-05-08 17:44:12
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-05-19 19:40:34
 */

#include "quantization_transform/passes/passes.h"

namespace qtlibn
{
// we appoint that first output tensor should include 'lut_items_bits'.
void update_min_max_range_to_tensor(aipubt::GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto &node : nodes)
    {
        for (auto &ct : node->constants())
        {
            auto key = ct.first;
            if (!ct.second->quantization().data_unchanged() && node->params().count(key + "_range"))
            {
                std::string range_key = key + "_range";
                std::vector<float> min_max_range = node->params()[range_key];
                if (ct.second->quantization().scales().size() != min_max_range.size() / 2)
                {
                    //depthwise conv weights needs per-channel, but qat asymmetric is per-tensor
                    continue;
                }
                if (min_max_range.size() == 2)
                {
                    ct.second->quantization().set_mins({min_max_range[0]});
                    ct.second->quantization().set_maxs({min_max_range[1]});
                }
                else
                {
                    WARN() << "min/max range of " << range_key << " has " << min_max_range.size() << " elements, now we expect 2 elements, which means [mins, maxs].";
                }
            }
        }

        if (node->params().count("layer_top_range"))
        {
            std::vector<std::vector<float>> layer_top_range = node->params()["layer_top_range"];
            if (layer_top_range.size() != node->outputs().size())
            {
                WARN() << "layer_top_range size(=" << layer_top_range.size() << ") != node outputs size(" << node->outputs().size() << ")";
            }
            int32_t o_size = node->outputs().size();
            for (int32_t o = 0; o < o_size; o++)
            {
                if (layer_top_range[o].size() == 2)
                {
                    node->outputs()[o]->quantization().set_mins({layer_top_range[o][0]});
                    node->outputs()[o]->quantization().set_maxs({layer_top_range[o][1]});
                }
                else
                {
                    WARN() << "layer_top range has " << layer_top_range[o].size() << " elements, now we expect 2 elements, which means [mins, maxs].";
                }
            }
        }
    }
}
} // namespace qtlibn
