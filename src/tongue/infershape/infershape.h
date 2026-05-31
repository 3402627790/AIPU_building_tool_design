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
 * @Date: 2023-03-20 15:23:11 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-03-20 17:42:43
 */
#pragma once
#include <core/constants.h>
#include <core/node.h>
#include <core/optype.h>
#include <core/tensor_shape.h>
#include <functional>
#include <map>
#include <tongue/symbolic.h>
#include <vector>

namespace tongue
{
using namespace aipubt;
using InferFunc = std::function<std::vector<SymbolShape>(NodePtr, const std::vector<SymbolShape> &)>;

std::vector<SymbolShape> infer_shape(NodePtr n, const std::vector<SymbolShape> &inputs, bool backward = false);
std::map<OpType, InferFunc> &get_infer_map(bool backward = false);

struct RegisterOpInferShape
{
    RegisterOpInferShape(OpType op, InferFunc fn, InferFunc backward_fn)
    {
        get_infer_map()[op] = fn;
        if (backward_fn)
        {
            get_infer_map(true)[op] = backward_fn;
        }
    }
    RegisterOpInferShape(OpType op, InferFunc fn) : RegisterOpInferShape(op, fn, nullptr) {}
};

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b
#define UNIQUE_NAME(base) CONCAT(base, __COUNTER__)
#define INFERSHAPE_REGISTER(op, ...) \
    static tongue::RegisterOpInferShape UNIQUE_NAME(_symbol_infershape##op)(op, __VA_ARGS__);
} //namespace tongue
