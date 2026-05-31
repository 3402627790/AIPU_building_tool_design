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
 * @Author: Kyle Huang
 * @Date: 2020-11-1 10:04:00
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-08-26 14:55:52
 */
#ifndef SPLIT_NODE_OUTPUT_C_H
#define SPLIT_NODE_OUTPUT_C_H
#include "core/basepass.h"
#include "core/core.h"
#include "builder/target.h"

namespace aipubt
{
class SplitNodeOutputC : public PassBase<SplitNodeOutputC>
{
private:
    mutable std::vector<int> m_output_c_vec;

public:
    SplitNodeOutputC(){};
    ~SplitNodeOutputC(){};
    bool operator()(GraphPtr g) const override;
    void set_output_c(const std::vector<int> &output_c_vec) const;
    bool split_output_c(GraphPtr g, NodePtr n) const;
};
} // namespace aipubt
#endif //!CORE_BUILDER_LOWERING_HAS_CONSTANT_H
