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
 * @Date: 2023-01-13 14:37:57
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-01-17 21:59:00
 */

#include "quantization_transform/passes/passes.h"

namespace qtlibn
{

static inline bool criteria(aipubt::NodePtr node)
{
    if (!(node->attrs().count("int8_to_uint8") && (bool)node->get_attr("int8_to_uint8")))
    {
        return false;
    }
    // if (node->type() != aipubt::OpType_Convolution && node->type() != aipubt::OpType_Concat)
    // {
    //     return false;
    // }
    // if (node->type() == aipubt::OpType_Convolution)
    // {
    //     std::string with_activation = node->params()["with_activation"];
    //     if (with_activation != "RELU")
    //     {
    //         return false;
    //     }
    // }

    auto output_dtype = node->outputs()[0]->type();
    if (output_dtype != aipubt::Tensor::INT8)
    {
        return false;
    }

    auto zps = node->outputs()[0]->quantization().offsets();
    auto zp_is_neg_128 = zps.all_of([](int32_t zp) -> bool { return zp == -128 ? true : false; });
    if (!zp_is_neg_128)
    {
        return false;
    }
    return true;
}

void change_data_type_for_conv_relu(aipubt::GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto &n : nodes)
    {
        if (!criteria(n))
        {
            continue;
        }
        auto src_dtype = n->outputs()[0]->type();
        auto aim_dtype = aipubt::Tensor::UINT8;
        n->outputs()[0]->set_type(aim_dtype);
        auto zps = n->outputs()[0]->quantization().offsets() + 128;
        n->outputs()[0]->quantization().set_offsets(zps);
        // n->outputs()[0]->quantization().set_primitive_dtype(aim_dtype);
        if (n->outputs()[0]->quantization().aipu_dtype() != aipubt::Tensor::VOID &&
            n->outputs()[0]->quantization().aipu_dtype() == src_dtype)
        {
            n->outputs()[0]->quantization().set_aipu_dtype(aim_dtype);
        }
    }
}

} // namespace qtlibn
