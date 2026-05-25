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
 * @Date: 2022-05-19 12:39:00
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-05-19 12:42:45
 */

#ifndef __QT_OPS_SIGN_H__
#define __QT_OPS_SIGN_H__

#include "quantization_transform/ops/op.h"
#include "quantization_transform/ops/op_register.h"

namespace qtlibn
{
class SignOp : public Op
{
public:
    SignOp() {}
    ~SignOp() {}

    void quantize(aipubt::NodePtr node) override;
};

} // namespace qtlibn

#endif