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
 * @Date: 2022-06-29 11:51:49
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-06-29 18:00:26
 */

#include "core/core.h"
#include "core/ops/ops.h"

namespace aipubt
{
template <typename T>
std::vector<std::pair<T, T>> vector_to_pair(const std::vector<std::vector<T>> &vector)
{
    std::vector<std::pair<T, T>> pair;
    for (size_t i = 0; i < vector.size(); ++i)
    {
        pair.push_back({vector[i][0], vector[i][1]});
    }
    return pair;
}

static bool criteria(const NodePtr n0, const NodePtr n1)
{

    std::vector<std::vector<int32_t>> pads = n0->params()["pads"];
    std::vector<std::vector<int32_t>> crops = n1->params()["crops"];
    auto crop_n_inp_shape = n1->inputs()[0]->shape();
    std::string mode = n0->params()["mode"];

    if (pads.size() != crops.size() || mode != "CONSTANT" || n0->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    auto pad_inp_shape = n0->inputs()[0]->shape();
    auto crop_out_shape = n1->outputs()[0]->shape();
    if (pad_inp_shape != crop_out_shape)
    {
        return false;
    }
    if (pad_inp_shape.symbol() != crop_out_shape.symbol())
    {
        return false;
    }
    for (uint32_t c = 0; c < crops.size(); c++)
    {
        /**
         * pad op: layer_bottom_shape=[1,111,2], pads=[[0,0],[0,111],[0,0]]
         * crop op: layer_top_shape=[1,111,2], crops=[[0,1],[112,222],[0,2]]
         * after swap:
         * crop op: layer_bottom_shape=[1,111,2], crops=[[0,1],[111,111],[0,2] ==> layer_top_shape=[1,0,2]
         * cause unreasonable, so we skip this case which not swap the pad and crop.
         */
        auto pad_c_dim_val = pad_inp_shape[c];
        auto crop = crops[c];
        if (pad_c_dim_val < crop[0])
        {
            return false;
        }
    }

    for (uint32_t i = 0; i < pads.size(); i++)
    {
        auto pad = pads[i];
        auto crop = crops[i];
        if (pad[0] == 0 && pad[1] == 0)
        {
            // pad[0:2] == [0, 0] will have not effect on swapping the pad and crop.
            continue;
        }
        else if (pad[0] == 0 && pad[1] > 0)
        {
            auto i_dim_val = crop_n_inp_shape[i];
            if (crop[1] != i_dim_val)
            {
                return false;
            }
        }
        else if (pad[0] > 0 && pad[1] == 0)
        {
            if (crop[0] != 0)
            {
                return false;
            }
        }
        else
        {
            auto i_dim_val = crop_n_inp_shape[i];
            if (crop[0] != 0 || crop[1] != i_dim_val)
            {
                return false;
            }
        }
    }

    return true;
}

bool swap_pad_crop(GraphPtr g)
{
    GraphMatcherPattern pattern = {
        {{"0", OpType_Pad}, {"1", OpType_Crop}},
        {{"0", "1"}}};
    GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
    auto matched = matcher.match(g);
    bool is_changed = false;
    while (matched.size())
    {
        bool is_merged = false;
        for (auto &&m : matched)
        {
            if (!criteria(m[0], m[1]))
            {
                continue;
            }
            std::vector<std::vector<uint32_t>> pads = m[0]->params()["pads"];
            std::vector<std::vector<int32_t>> crops = m[1]->params()["crops"];
            auto pad_inpt = m[0]->inputs()[0];
            auto crop_outt = m[1]->outputs()[0];

            auto pad_inp_shape = pad_inpt->shape();
            std::vector<std::pair<int32_t, int32_t>> new_crops;
            for (uint32_t c = 0; c < crops.size(); c++)
            {
                auto crop = crops[c];
                auto pad_dim_val = pad_inp_shape[c];
                auto nc = std::make_pair(std::min(crop[0], pad_dim_val), std::min(crop[1], pad_dim_val));
                new_crops.push_back(nc);
            }
            std::vector<std::vector<tongue::Expr>> ds_crops;
            if (m[1]->params().count("ds_crops"))
            {
                ds_crops = m[1]->params()["ds_crops"];
            }
            auto new_crop_outt = ops::crop(pad_inpt, new_crops, vector_to_pair(ds_crops));
            auto crop_node = new_crop_outt->node();
            auto new_pad_outt = ops::pad(new_crop_outt, vector_to_pair<uint32_t>(pads));
            auto pad_node = new_pad_outt->node();
            pad_node->clear_outputs();
            pad_node->add_output(crop_outt);
            ADD_LOCATION_BY_NODE(crop_node, m[0], m[1]);
            ADD_LOCATION_BY_NODE(pad_node, m[0], m[1]);

            g->remove_node(m[0]);
            g->remove_node(m[1]);

            is_changed = true;
            is_merged = true;
        }
        if (!is_merged)
        {
            break;
        }
        matched = matcher.match(g);
    }
    return is_changed;
}

} // namespace aipubt
