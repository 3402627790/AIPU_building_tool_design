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
 * @Date: 2023-05-16 11:02:24
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-05-16 11:54:59
 */

#include "passes.h"

namespace qtlibn
{

void compress_bias(aipubt::GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto &node : nodes)
    {
        for (auto ct : node->constants())
        {
            if (ct.first == "biases" && !aipubt::Tensor::is_float(ct.second->type()))
            {
                auto bits = ct.second->quantization().bits();
                auto dtype = ct.second->quantization().aipu_dtype();
                auto issigned = aipubt::Tensor::is_signed(dtype);
                float qmin = static_cast<float>(dtype_utils::bits_to_range_min(bits, issigned));
                float qmax = static_cast<float>(dtype_utils::bits_to_range_max(bits, issigned));

                aipubt::ComputeTensor bias_ct = ct.second;
                auto bias_ct_max = bias_ct.max();
                auto bias_ct_min = bias_ct.min();
                bias_ct_max = bias_ct_max.astype(aipubt::Tensor::FP32);
                bias_ct_min = bias_ct_min.astype(aipubt::Tensor::FP32);
                float q_b_max = ((aipubt::TensorPtr)bias_ct_max)->data<float>()[0];
                float q_b_min = ((aipubt::TensorPtr)bias_ct_min)->data<float>()[0];

                if (q_b_max > qmax || q_b_min < qmin)
                {
                    bias_ct = bias_ct.clamp(qmin, qmax);
                    auto bk_quantization = ct.second->quantization();
                    node->constants()["biases"] = bias_ct;
                    node->constants()["biases"]->quantization() = bk_quantization;
                }
            }
        }
    }
}

} // namespace qtlibn
