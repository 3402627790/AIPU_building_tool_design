
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
 * @Date: 2022-05-13 14:12:11 
 * @Last Modified by: Tom Gao
 * @Last Modified time: 2022-07-31 12:14:16
 */
#ifndef CORE_BUILDER_LOWERING_HAS_CONSTANT_H
#define CORE_BUILDER_LOWERING_HAS_CONSTANT_H
#include "core/basepass.h"
#include "core/core.h"
namespace aipubt
{

class HashConstant : public PassBase<HashConstant>
{
public:
    bool operator()(GraphPtr g) const override;
};

};     // namespace aipubt
#endif //!CORE_BUILDER_LOWERING_HAS_CONSTANT_H
