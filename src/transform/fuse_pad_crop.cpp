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
 * @Author: Tom Gao
 * @Date: 2022-03-22 10:25:43
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-09-10 21:52:49
 */
#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include <algorithm>
#include <set>

namespace aipubt
{

static std::vector<std::vector<uint32_t>>
crops_to_nums(const std::vector<std::vector<int32_t>> &crops,
              const TensorShape &in_shape)
{
    std::vector<std::vector<uint32_t>> nums;
    for (uint32_t i = 0; i < crops.size(); ++i)
    {
        nums.push_back({static_cast<uint32_t>(crops[i][0]),
                        static_cast<uint32_t>(in_shape[i] - crops[i][1])});
    }
    return nums;
}

static std::vector<std::vector<int32_t>>
nums_to_crops(const std::vector<std::vector<int32_t>> &nums,
              const TensorShape &in_shape)
{
    std::vector<std::vector<int32_t>> crops;
    for (uint32_t i = 0; i < nums.size(); ++i)
    {
        crops.push_back({static_cast<int32_t>(nums[i][0]),
                         static_cast<int32_t>(in_shape[i] - nums[i][1])});
    }
    return crops;
}
static std::vector<std::vector<tongue::Expr>>
ds_crops_to_nums(const std::vector<std::vector<tongue::Expr>> &ds_crops,
                 const SymbolShape &in_shape)
{
    std::vector<std::vector<tongue::Expr>> nums;
    for (uint32_t i = 0; i < ds_crops.size(); ++i)
    {
        nums.push_back({tongue::Expr(ds_crops[i][0]),
                        in_shape[i] - tongue::Expr(ds_crops[i][1])});
    }
    return nums;
}

static std::vector<std::vector<tongue::Expr>>
nums_to_ds_crops(const std::vector<std::vector<int32_t>> &nums,
                 const SymbolShape &in_shape)
{
    std::vector<std::vector<Expr>> crops;
    for (uint32_t i = 0; i < nums.size(); ++i)
    {
        crops.push_back({tongue::Expr(nums[i][0]),
                         in_shape[i] - tongue::Expr(nums[i][1])});
    }
    return crops;
}

static inline bool creteria(NodePtr n1, NodePtr n2)
{
    std::vector<std::vector<int>> pads = n1->params()["pads"];
    std::vector<std::vector<int>> crops = n2->params()["crops"];
    if (pads.size() != crops.size() || n1->outputs()[0]->is_io_tensor())
    {
        return false;
    }

    // The expression cannot perform size comparison, so it is excluded.
    if (n1->params().count("ds_pads"))
    {
        std::vector<std::vector<tongue::Expr>> ds_pads = n1->params()["ds_pads"];
        for (auto exprs : ds_pads)
        {
            if (std::any_of(exprs.begin(), exprs.end(), [](tongue::Expr e)
                            { return !e->is_constant(); }))
            {
                return false;
            }
        }
    }

    // crop value is fixed.
    if (n2->params().count("ds_crops"))
    {
        std::vector<std::vector<tongue::Expr>> ds_crops = n2->params()["ds_crops"];
        auto ds_crop_nums = ds_crops_to_nums(ds_crops, n1->outputs()[0]->shape().symbol());
        for (auto exprs : ds_crop_nums)
        {
            if (std::any_of(exprs.begin(), exprs.end(), [](tongue::Expr e)
                            { return !e->is_constant(); }))
            {
                return false;
            }
        }
    }
    auto crop_nums = crops_to_nums(crops, n1->outputs()[0]->shape());
    bool ret = false;
    for (size_t i = 0; i < pads.size(); ++i)
    {
        for (size_t j = 0; j < 2; ++j)
        {
            ret |= (pads[i][j] != 0 && crop_nums[i][j] != 0);
        }
    }
    return ret;
}

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

bool fuse_pad_crop(GraphPtr g)
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
            if (!creteria(m[0], m[1]))
            {
                continue;
            }
            std::vector<std::vector<uint32_t>> pads = m[0]->params()["pads"];
            std::vector<std::vector<int32_t>> crops = m[1]->params()["crops"];
            auto inp_shape = m[0]->inputs()[0]->shape();
            auto pad_out_shape = m[0]->outputs()[0]->shape();
            auto outp_shape = m[1]->outputs()[0]->shape();
            auto crop_nums = crops_to_nums(crops, pad_out_shape);
            std::vector<std::vector<uint32_t>> pads_new;
            std::vector<std::vector<int32_t>> crop_nums_new;
            for (uint32_t i = 0; i < pads.size(); ++i)
            {
                std::vector<uint32_t> pads_tmp;
                std::vector<int32_t> crops_tmp;
                for (uint32_t j = 0; j < 2; ++j)
                {
                    int merged_num = pads[i][j] - crop_nums[i][j];
                    if (merged_num > 0)
                    {
                        pads_tmp.push_back(merged_num);
                        crops_tmp.push_back(0);
                    }
                    else
                    {
                        pads_tmp.push_back(0);
                        crops_tmp.push_back(0 - merged_num);
                    }
                }
                pads_new.push_back(pads_tmp);
                crop_nums_new.push_back(crops_tmp);
            }

            auto inp = m[0]->remove_input(0);
            auto out = m[1]->remove_output(0);
            std::map<std::string, ops::PadMode> mode_table =
                {
                    {"CONSTANT", ops::PadMode::Constant},
                    {"REFLECT", ops::PadMode::Reflect},
                    {"SYMMETRIC", ops::PadMode::Symmetric},
                };
            auto pad_out_new = ops::pad(inp, vector_to_pair(pads_new), mode_table[m[0]->params()["mode"]], (int32_t)m[0]->params()["constant_value"]);
            auto crops_new = nums_to_crops(crop_nums_new, pad_out_new->shape());
            std::vector<std::pair<tongue::Expr, tongue::Expr>> ds_crops_new;
            if (m[1]->params().count("ds_crops"))
            {
                auto ds_crops_symb = nums_to_ds_crops(crop_nums_new, pad_out_new->shape().symbol());
                ds_crops_new = vector_to_pair(ds_crops_symb);
            }
            auto crop_out = ops::crop(pad_out_new, vector_to_pair(crops_new), ds_crops_new);
            auto crop_new = crop_out->node();
            crop_new->remove_output(crop_out);
            crop_new->add_output(out);
            ADD_LOCATION_BY_TENSOR(pad_out_new, m[0], m[1]);
            ADD_LOCATION_BY_NODE(crop_new, m[0], m[1]);
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
}; // namespace aipubt
