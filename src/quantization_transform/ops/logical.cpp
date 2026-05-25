//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2022 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Samuel Han
 * @Date: 2022-03-23 19:22:58
 * @Last Modified by:   Runner Zhong
 * @Last Modified time: 2022-04-22 17:03:40
 */

#include "quantization_transform/ops/logical.h"
using namespace std;

namespace qtlibn
{
void LogicalOp::quantize(aipubt::NodePtr node)
{

    string method = node->params()["method"];
    vector<string> supported_method = {"ADD", "OR", "NOT", "XOR"};
    vector<string>::iterator it;
    it = find(supported_method.begin(), supported_method.end(), method);

    if (it == supported_method.end())
    {
        bool qinvariant0 = node->inputs()[0]->quantization().qinvariant();
        bool qinvariant1 = node->inputs()[1]->quantization().qinvariant();

        float in_scale0 = node->inputs()[0]->quantization().scale();
        float in_scale1 = node->inputs()[1]->quantization().scale();

        float inp_scale_max = max(in_scale0, in_scale1);

        if (qinvariant0 && !qinvariant1)
        {
            inp_scale_max = in_scale0;
        }
        if (qinvariant1 && !qinvariant0)
        {
            inp_scale_max = in_scale1;
        }

        uint32_t scale0 = quant_utils::clip(static_cast<uint32_t>((inp_scale_max / in_scale0) * 8192), 0U, 32767U);
        uint32_t scale1 = quant_utils::clip(static_cast<uint32_t>((inp_scale_max / in_scale1) * 8192), 0U, 32767U);
        node->params()["scale_value"] = {scale0, scale1};
        node->params()["scale_type"] = {aipubt::Tensor::UINT16, aipubt::Tensor::UINT16};
    }
    node->outputs()[0]->quantization().set_bits(8);
    node->outputs()[0]->quantization().set_qinvariant(true);
    node->outputs()[0]->quantization().set_scales({1.});
    node->outputs()[0]->quantization().set_offsets({0});
    node->outputs()[0]->quantization().set_quantized(true);
    dtype_utils::set_aipu_dtype(node->outputs()[0], false);
}

REGISTER_QUANTIZATION_OP(Logical);

} // namespace qtlibn
