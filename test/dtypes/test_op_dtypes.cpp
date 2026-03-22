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
 * @Author: Tom.Gao 
 * @Date: 2022-03-04 10:30:48 
 * @Last Modified by: Tom.Gao
 * @Last Modified time: 2022-03-04 10:30:48
 */
#include <iostream>
#include <string>
#include <algorithm>
#include <set>
#include <vector>

#include <utils/string_utils.h>
#include <dtypes/op_dtypes.h>
using namespace aipubt;
void test_op_dtypes()
{
    auto dtypes_to_strings = [](std::vector<Tensor::Dtype> dtypes) -> std::vector<std::string> {
        std::vector<std::string> strings;
        for (auto dtype: dtypes)
        {
            strings.push_back(Tensor::dtype_to_string(dtype));
        }
        return strings;
    };
    // OpType_Convolution
    auto optype = OpType_Pooling;
    auto op_dtypes = Op_Dtypes::get_dtypes(optype);
    DEBUG() << "Valid in/out Dtypes for " << optype_to_string(optype);
    for (size_t i = 0; i < op_dtypes.size(); i++)
    {
        auto op_dtype = op_dtypes[i];
        DEBUG() << "\tvalid_pair[" << i << "]: "
                << "in: " << utils::join_string(",", dtypes_to_strings(op_dtype.in_dtypes)) << ",\t"
                << "out: " << utils::join_string(",", dtypes_to_strings(op_dtype.out_dtypes));
    }
}