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
 * @Date: 2020-05-08 14:56:54
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-04-12 18:49:37
 */

#ifndef __QUANTIZATION_OP_H__
#define __QUANTIZATION_OP_H__

#include "quantization_transform/adaptor.h"
#include "quantization_transform/common.h"
#include "quantization_transform/generate_lut.h"
#include "quantization_transform/ops/op_register.h"
#include "quantization_transform/utils/datatransfer_utils.h"
#include "quantization_transform/utils/param_utils.h"

namespace qtlibn
{
class Op
{
public:
    virtual ~Op() {}
    virtual void quantize(aipubt::NodePtr node) = 0;
    virtual void forward(aipubt::NodePtr node) {}
    virtual void approximate(aipubt::NodePtr node);

    AdaptorPtr get_adaptor(void) const;
    void set_adaptor(AdaptorPtr adaptor);

private:
    AdaptorPtr m_adaptor = nullptr;
};

}; // namespace qtlibn
#endif