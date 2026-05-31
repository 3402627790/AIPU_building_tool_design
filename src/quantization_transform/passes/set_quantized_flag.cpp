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
 * @Date: 2022-03-07 18:57:23
 * @Last Modified by: Ruodi Lu
 * @Last Modified time: 2022-04-11 16:13:40
 */

#include "quantization_transform/passes/passes.h"

namespace qtlibn
{
void set_quantized_flag(aipubt::GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto &node : nodes)
    {
        std::vector<aipubt::TensorPtr> tensors = node->outputs();
        for (aipubt::TensorPtr &tensor : tensors)
        {
            //Todo: conflict with android, solve later
            if (tensor->quantization().aipu_dtype() == aipubt::Tensor::Dtype::VOID)
            {
                tensor->quantization().set_aipu_dtype(aipubt::Tensor::FP32);
                tensor->quantization().set_quantized(false);
            }
            else
            {
                // tensor->quantization().set_quantized(true);
            }
        }
    }
}
} // namespace qtlibn
