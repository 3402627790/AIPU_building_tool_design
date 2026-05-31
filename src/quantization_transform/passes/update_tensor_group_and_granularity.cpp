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
 * @Date: 2022-04-07 17:22:19
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-12-09 10:15:39
 */

#include "passes.h"

namespace qtlibn
{

void update_tensor_group_and_granularity(aipubt::NodePtr n)
{
    std::vector<aipubt::OpType> _has_group_ops = {
        aipubt::OpType_Convolution,
        aipubt::OpType_Convolution3D,
        aipubt::OpType_ConvTranspose3D,
        aipubt::OpType_ConvTranspose,
        aipubt::OpType_DepthwiseConv,
    };
    if (!aipubt::contains(_has_group_ops, n->type()))
    {
        return;
    }
    if (n->params().count("group"))
    {
        auto group = n->params()["group"];
        for (auto t : n->constants())
        {
            if (t.second->quantization().group() != 1)
            {
                continue;
            }

            int scale_size = t.second->quantization().scales().size();
            if (scale_size != 1 && scale_size < (t.second->shape()[0]))
            {
                int32_t perchannels = t.second->quantization().scales().size();
                t.second->quantization().set_group(perchannels);
            }
            else if (group != 1)
            {
                t.second->quantization().set_group(group);
            }
        }
    }

    for (auto ot : n->outputs())
    {
        auto s_size = ot->quantization().scales().size();
        auto granu = ot->quantization().granularity();
        if (s_size > 1 && granu == aipubt::Quantization::Per_Tensor)
        {
            ot->quantization().set_granularity(aipubt::Quantization::Per_Channel);
        }
    }

    for (auto plh : n->placeholders())
    {
        auto s_size = plh->quantization().scales().size();
        auto granu = plh->quantization().granularity();
        if (s_size > 1 && granu == aipubt::Quantization::Per_Tensor)
        {
            plh->quantization().set_granularity(aipubt::Quantization::Per_Channel);
        }
    }

    for (auto ct : n->constants())
    {
        auto ot = ct.second;
        int32_t s_size = ot->quantization().scales().size();
        auto granu = ot->quantization().granularity();
        auto group = ot->quantization().group();
        int32_t per_channel_size = n->outputs()[0]->shape().c();
        if (group > 1 && s_size == 1)
        {
            // DEBUG() << "node.name = " << n->name() << " has group > 1 but only one scale/zp, we change to per-channel.";
            // float scale = ot->quantization().scale();
            // int32_t offset = ot->quantization().offset();
            // std::vector<float> scale_repeat(per_channel_size, scale);
            // std::vector<int32_t> offset_repeat(per_channel_size, offset);
            // ot->quantization().set_scales(scale_repeat);
            // ot->quantization().set_offsets(offset_repeat);
            ot->quantization().set_granularity(aipubt::Quantization::Per_Tensor);
        }
        else if (group > 1 && s_size == group && s_size != per_channel_size)
        {
            std::vector<float> scales = ot->quantization().scales();
            std::vector<int32_t> offsets = ot->quantization().offsets();
            int32_t step = per_channel_size / group;
            if (group * step != per_channel_size)
            {
                WARN() << "group(=" << group << ") * step(=" << step << ") != output c(=" << per_channel_size << ").";
            }
            std::vector<float> repeat_scales(per_channel_size);
            std::vector<int32_t> repeat_offsets(per_channel_size);
            for (int32_t i = 0; i < group; i++)
            {
                std::fill(repeat_scales.begin() + i * step, repeat_scales.begin() + (i + 1) * step, scales[i]);
                std::fill(repeat_offsets.begin() + i * step, repeat_offsets.begin() + (i + 1) * step, offsets[i]);
            }
            ot->quantization().set_scales(repeat_scales);
            ot->quantization().set_offsets(repeat_offsets);
            ot->quantization().set_granularity(aipubt::Quantization::Per_Channel);
        }
        // else if (!ot->quantization().data_unchanged())
        // {
        //     DEBUG() << "node.name = " << n->name() << " has group == 1 but only one scale/zp, we change to per-channel.";
        //     float scale = ot->quantization().scale();
        //     int32_t offset = ot->quantization().offset();
        //     std::vector<float> scale_repeat(per_channel_size, scale);
        //     std::vector<int32_t> offset_repeat(per_channel_size, offset);
        //     ot->quantization().set_scales(scale_repeat);
        //     ot->quantization().set_offsets(offset_repeat);
        //     ot->quantization().set_granularity(aipubt::Quantization::Per_Channel);
        // }

        if (s_size > 1 && granu == aipubt::Quantization::Per_Tensor)
        {
            DEBUG() << ot->name() << " changes per-tensor to per-channel";
            ot->quantization().set_granularity(aipubt::Quantization::Per_Channel);
        }
    }
}

void update_tensor_group_and_granularity(aipubt::GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto &n : nodes)
    {
        update_tensor_group_and_granularity(n);
    }
}

} // namespace qtlibn
