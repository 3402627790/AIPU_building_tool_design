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
#ifndef SRC_ENGINE_ENGINE_H
#define SRC_ENGINE_ENGINE_H
#include "core/node.h"
#include "core/optype.h"
#include <functional>

namespace aipubt
{

namespace engine
{

    bool run_node(NodePtr n);

    enum DTCriteria //input/output data type checking
    {
        ALL_TYPE = 0,
        INT8_ONLY = 1,
        INT16_ONLY = 2,
        INT_ONLY = 3, // INT8+INT16+INT32
        FP32_ONLY = 4,
        FP16_ONLY = 5,
        BFP16_ONLY = 6,
        FP_ONLY = 7, //support all float data type
    };
    void register_op(OpType op, std::function<void(NodePtr)> exefun, std::function<bool(NodePtr)> criteria);
    void register_op(OpType op, std::function<void(NodePtr)> exefun, DTCriteria dt = ALL_TYPE, std::function<bool(NodePtr)> criteria = nullptr);

    struct RegisterOpExe
    {
        RegisterOpExe(OpType op, std::function<void(NodePtr)> exefun, DTCriteria dt, std::function<bool(NodePtr)> criteria = nullptr)
        {
            register_op(op, exefun, dt, criteria);
        }
        RegisterOpExe(OpType op, std::function<void(NodePtr)> exefun, std::function<bool(NodePtr)> criteria = nullptr)
        {
            register_op(op, exefun, criteria);
        }
    };

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b
#define UNIQUE_NAME(base) CONCAT(base, __COUNTER__)
#define EXEOP_REGISTER(op, ...) \
    static aipubt::engine::RegisterOpExe UNIQUE_NAME(_engine_op##op)(op, __VA_ARGS__);
} // namespace engine

} // namespace aipubt
#endif //!_SRC_ENGINE_ENGINE_H