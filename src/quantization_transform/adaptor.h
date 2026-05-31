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
 * @Date: 2022-03-11 18:08:52
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-08-03 22:26:03
 */

#ifndef __ADAPTOR_H__
#define __ADAPTOR_H__

#include "quantization_transform/run_mode.h"
#include "quantization_transform/transforms/base_transfrom.h"
#include "quantization_transform/transforms/register.h"

namespace qtlibn
{

class Adaptor;
using AdaptorPtr = std::shared_ptr<Adaptor>;
class Adaptor
{
public:
    explicit Adaptor(RunMode run_mode) : m_run_mode(run_mode)
    {
        m_transform = TransformRegistry::get_transform(m_run_mode);
    }

    void calculate_max_min(aipubt::NodePtr node);
    void calculate_scale_zp(aipubt::NodePtr node);
    void calculate_activation_max_min(aipubt::TensorPtr tensor);
    void dequantize_constant(aipubt::NodePtr node);
    void dequantize_param(aipubt::NodePtr node);

private:
    RunMode m_run_mode = RunMode::Sym;
    BaseTransform *m_transform = nullptr;
};

}; // namespace qtlibn
#endif
