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
 * @Date: 2022-03-03 14:06:21
 * @Last Modified by: Tom.Gao
 * @Last Modified time: 2022-03-03 14:06:21
 */
#ifndef OP_DTYPES_H
#define OP_DTYPES_H

#include "dtypes/plugin.h"

namespace aipubt
{
class Op_Dtypes
{
public:
    static std::vector<Op_Dtype> get_dtypes(OpType optype);
};
}; // namespace aipubt

#endif //!OP_DTYPES_H