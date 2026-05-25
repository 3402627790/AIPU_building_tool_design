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
 * @Author: Neo Wang
 * @Date: 2023-03-20 15:35:10
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-03-20 16:57:08
 */
#include "engine/engine.h"
#include "engine.h"
#include <map>
#include <vector>

namespace aipubt
{

namespace engine
{
    struct OpExecutor
    {
        std::function<void(NodePtr)> exe;
        std::function<bool(NodePtr)> criteria;
    };

    std::map<OpType, std::vector<OpExecutor>> &get_engine_ops()
    {
        static std::map<OpType, std::vector<OpExecutor>> en_ops;
        return en_ops;
    }
    bool run_node(NodePtr n) //TODO support Subgraph
    {
        if (!n)
        {
            return false;
        }
        for (auto e : get_engine_ops()[n->type()])
        {
            if (e.criteria(n))
            {
                e.exe(n);
                return true;
            }
        }
        return false;
    }
    void register_op(OpType op, std::function<void(NodePtr)> exefun, std::function<bool(NodePtr)> criteria)
    {
        get_engine_ops()[op].push_back({exefun, criteria});
    }

    void register_op(OpType op, std::function<void(NodePtr)> exefun, DTCriteria dt, std::function<bool(NodePtr)> criteria)
    {
        static auto dtype_check = [](NodePtr n, const std::vector<Tensor::Dtype> &dtypes, std::function<bool(NodePtr)> criteria = nullptr) -> bool {
            for (auto inp : n->inputs())
            {
                if (!contains(dtypes, inp->type()))
                {
                    return false;
                }
            }
            for (auto inp : n->outputs())
            {
                if (!contains(dtypes, inp->type()))
                {
                    return false;
                }
            }
            if (criteria)
            {
                return criteria(n);
            }
            return true;
        };
        if (!criteria)
        {
            criteria = [](NodePtr) { return true; };
        }
        const std::map<DTCriteria, std::function<bool(NodePtr)>> c =
            {
                {ALL_TYPE, criteria},
                {INT8_ONLY, [=](NodePtr n) { return dtype_check(n, {Tensor::Dtype::INT8, Tensor::Dtype::UINT8}, criteria); }},
                {INT16_ONLY, [=](NodePtr n) { return dtype_check(n, {Tensor::Dtype::INT16, Tensor::Dtype::UINT16}, criteria); }},
                {INT_ONLY, [=](NodePtr n) { return dtype_check(n,
                                                               {Tensor::Dtype::UINT32, Tensor::Dtype::INT32,
                                                                Tensor::Dtype::INT8, Tensor::Dtype::UINT8,
                                                                Tensor::Dtype::INT16, Tensor::Dtype::UINT16},
                                                               criteria); }},
                {FP32_ONLY, [=](NodePtr n) { return dtype_check(n, {Tensor::Dtype::FP32}, criteria); }},
                {FP16_ONLY, [=](NodePtr n) { return dtype_check(n, {Tensor::Dtype::FP16}, criteria); }},
                {BFP16_ONLY, [=](NodePtr n) { return dtype_check(n, {Tensor::Dtype::BFP16}, criteria); }},
                {FP_ONLY, [=](NodePtr n) { return dtype_check(n, {Tensor::Dtype::FP64, Tensor::Dtype::FP32, Tensor::Dtype::FP16, Tensor::Dtype::BFP16}, criteria); }},
            };
        register_op(op, exefun, c.at(dt));
    }

} // namespace engine
} // namespace aipubt