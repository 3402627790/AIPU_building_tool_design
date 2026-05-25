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
 * @Date: 2022-04-03 19:28:20
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-10-19 11:21:24
 */

#ifndef __QT_SOFTMAX_H__
#define __QT_SOFTMAX_H__

#include "quantization_transform/ops/op.h"
#include "quantization_transform/ops/op_register.h"

namespace qtlibn
{
class SoftmaxOp : public Op
{
public:
    SoftmaxOp() {}
    ~SoftmaxOp() {}

    void quantize(aipubt::NodePtr node) override;
    void approximate(aipubt::NodePtr node) override;
    static void generate_softmax_lut(aipubt::NodePtr node, aipubt::TensorPtr in_t);

private:
    void uniform_quantization_impl(aipubt::NodePtr node);
    void log2_quantization_impl(aipubt::NodePtr node);
};

} // namespace qtlibn

#endif