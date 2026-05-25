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
 * @Date: 2024-08-29 22:29:28
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2024-08-29 22:33:26
 */

#ifndef __QT_OPS_CELU_H__
#define __QT_OPS_CELU_H__

#include "quantization_transform/ops/op.h"
#include "quantization_transform/ops/op_register.h"

namespace qtlibn
{
class CeluOp : public Op
{
public:
    CeluOp() {}
    ~CeluOp() {}

    void quantize(aipubt::NodePtr node) override{};
    void approximate(aipubt::NodePtr node) override;
    // quantize func in ActivationOp
};

}; // namespace qtlibn
#endif