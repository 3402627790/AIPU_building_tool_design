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
 * @Author: Tom.Gao
 * @Date: 2022-03-03 14:27:40
 * @Last Modified by: Tom.Gao
 * @Last Modified time: 2022-03-03 14:27:40
 */
#ifndef OP_DATA_TYPES_H
#define OP_DATA_TYPES_H
#include "core/tensor.h"
#include <cstddef>
#include <memory>
#include <vector>

namespace aipubt
{
struct Op_Dtype
{
    std::vector<Tensor::Dtype> in_dtypes;
    std::vector<Tensor::Dtype> out_dtypes;
    static std::string to_string(const Op_Dtype &dtype);
    bool operator<(const Op_Dtype &op_dtype) const;
};

std::ostream &operator<<(std::ostream &os, const Op_Dtype &dtype);

} //namespace aipubt
#endif //!OP_DATA_TYPES_H
