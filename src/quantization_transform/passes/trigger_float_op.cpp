//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2023 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Samuel Han
 * @Date: 2023-03-06 15:29:50
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-01-06 13:38:46
 */

#include "core/fp16/half.h"
#include "quantization_transform/passes/passes.h"

namespace qtlibn
{
void trigger_float_op(aipubt::GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto &node : nodes)
    {
        // if n.type=quantize, the input_tensor.type is float dtype,so we skip it
        if (!node->attrs().count(Constants::trigger_float_op) || node->type() == aipubt::OpType_Quantize)
        {
            continue;
        }
        else
        {
            auto _has_float_dtype = [](aipubt::TensorPtr t) -> bool {
                auto is_fp = aipubt::Tensor::is_float(t->type());
                auto is_iot = t->is_io_tensor();
                if (is_fp && !is_iot)
                {
                    return true;
                }
                else if (is_fp && is_iot) //input/output is io_tensor and float, will keep the origin float,becasue we insert the cast op
                {
                    return true;
                }
                return false;
            };
            std::vector<aipubt::Tensor::Dtype> setted_dtypes = quant_utils::preferred_fp_dtype(node);
            if (setted_dtypes[0] == aipubt::Tensor::VOID && setted_dtypes[1] == aipubt::Tensor::VOID)
            {
                continue;
            }

            // auto act_dtype = setted_dtypes[0];
            // auto wht_dtype = setted_dtypes[1];
            bool float_dtype = false;
            for (auto &tensors : {node->inputs(), node->outputs()})
            {
                for (auto t : tensors)
                {
                    float_dtype |= _has_float_dtype(t);
                    break;
                }
            }
            for (auto &ct : node->constants())
            {
                if (aipubt::Tensor::is_float(ct.second->type()) && ct.first != "biases")
                {
                    float_dtype |= true;
                    break;
                }
            }

            if (float_dtype)
            {
                node->set_attr("unquantifiable", true);
            }
        }
    }
}

} // namespace qtlibn
