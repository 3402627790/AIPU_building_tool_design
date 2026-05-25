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
 * @Author: Matt Chen
 * @Date: 2022-04-06 19:22:58
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-05-12 17:16:34
 */

#include "quantization_transform/ops/rgb2yuv.h"

namespace qtlibn
{
void RgbToYuvOp::quantize(aipubt::NodePtr node)
{

    std::string conversion = "BT709", format = "I420";
    if (node->params().count("conversion"))
    {
        conversion = std::string(param_utils::get_param(node, "conversion"));
    }
    node->params()["conversion"] = conversion;

    if (node->params().count("format"))
    {
        format = std::string(param_utils::get_param(node, "format"));
    }
    node->params()["format"] = format;

    if (!node->attrs().count(Constants::params_quantized) || !node->get_attr(Constants::params_quantized))
    {
        uint32_t qbits = 10;
        std::vector<float_t> local_coes = param_utils::get_param(node, "coefficient");
        std::vector<int16_t> coes;
        for (unsigned int i = 0; i < local_coes.size(); ++i)
        {
            float_t tmp_coe = local_coes[i];
            if (i < 6)
            {
                coes.push_back((int16_t)tmp_coe);
            }
            else
            {
                coes.push_back(static_cast<int16_t>(std::round(tmp_coe * std::pow(2, qbits))));
            }
        }
        node->params()["coefficient"] = coes;
        node->params()["coefficient_shift"] = qbits;
        node->params()["coefficient_dtype"] = aipubt::Tensor::UINT16;
    }

    // assign output quantization
    dtype_utils::set_aipu_dtype(node->outputs()[0], aipubt::Tensor::is_signed(aipubt::Tensor::UINT8)); // uint8 by default
    node->outputs()[0]->quantization().set_qinvariant(node->inputs()[0]->quantization().qinvariant());
    node->outputs()[0]->quantization().set_scale(1);
    node->outputs()[0]->quantization().set_offset(0);
    node->outputs()[0]->quantization().set_bits(param_utils::get_param(node, "bits"));
}

REGISTER_QUANTIZATION_OP(RgbToYuv);

}; // namespace qtlibn
