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
 * @Date: 2022-03-26 23:10:35
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-08-23 16:19:17
 */

#include "quantization_transform/ops/hash_lookup_table.h"

namespace qtlibn
{
void HashLookupTableOp::quantize(aipubt::NodePtr node)
{
    bool sign = aipubt::Tensor::is_signed(node->inputs()[0]->quantization().aipu_dtype());
    if (node->outputs()[0]->quantization().mins().size() > 0)
    {
        auto mins = node->outputs()[0]->quantization().mins()[0];
        if (mins >= 0)
        {
            sign = false;
        }
    }
    dtype_utils::set_aipu_dtype(node->outputs()[0], sign);
    dtype_utils::set_aipu_dtype(node->outputs()[1], false);

    Op::quantize(node);
    float miss_data = 0; //default = 0
    if (node->params().count("miss_default_data"))
    {
        miss_data = node->params()["miss_default_data"];
    }
    if (node->attrs().count("pad_quantized") && node->get_attr("pad_quantized"))
    {
        node->params()["miss_default_data"] = static_cast<int32_t>(miss_data);
    }
    else
    {
        float aipu_scale = node->outputs()[0]->quantization().scale();
        quant_utils::zp_qdtype aipu_zp = node->outputs()[0]->quantization().offset();
        aipubt::Tensor::Dtype o_dtype = node->outputs()[0]->quantization().aipu_dtype();
        int32_t q_min = dtype_utils::dtype_to_range_min(o_dtype);
        int32_t q_max = dtype_utils::dtype_to_range_max(o_dtype);
        int32_t new_miss_data = static_cast<int32_t>(quant_utils::linear_quantize_clip(miss_data, aipu_scale, aipu_zp, q_min, q_max));
        node->params()["miss_default_data"] = new_miss_data;
    }
}

REGISTER_QUANTIZATION_OP(HashLookupTable);

}; // namespace qtlibn