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
 * @Author: Runner Zhong
 * @Date: 2022-03-23 19:22:58
 * @Last Modified by:   Runner Zhong
 * @Last Modified time: 2022-04-22 17:40:41
 */

#include "quantization_transform/ops/gemm.h"
using namespace std;

namespace qtlibn
{
void GemmOp::quantize(aipubt::NodePtr node)
{
    int inp_bit = node->inputs()[0]->quantization().bits();
    float alpha = node->params()["alpha"];
    float beta = 1.0;

    bool sign0 = aipubt::Tensor::is_signed(node->inputs()[0]->type());
    bool sign1 = aipubt::Tensor::is_signed(node->inputs()[1]->type());

    bool qinvariant0 = node->inputs()[0]->quantization().qinvariant();
    bool qinvariant1 = node->inputs()[1]->quantization().qinvariant();

    bool out_sign = sign0 || sign1 || (alpha < 0);
    bool out_qinvariant = qinvariant0 && qinvariant1;

    if (node->inputs().size() == 3)
    {
        beta = node->params()["beta"];
        bool sign2 = aipubt::Tensor::is_signed(node->inputs()[2]->type());
        bool qinvariant2 = node->inputs()[2]->quantization().qinvariant();
        out_sign = out_sign || sign2 || (beta < 0);
        out_qinvariant = out_qinvariant && qinvariant2;
    }

    if (out_qinvariant)
    {
        node->outputs()[0]->quantization().set_scale(1);
        node->outputs()[0]->quantization().set_offset(0);
        float out_min = node->outputs()[0]->quantization().mins()[0];
        float out_max = node->outputs()[0]->quantization().maxs()[0];
        int tmp_qbit = dtype_utils::range_to_bits(out_min, out_max, out_sign);
        node->outputs()[0]->quantization().set_bits(tmp_qbit);
        dtype_utils::set_aipu_dtype(node->outputs()[0], out_sign);
        node->outputs()[0]->quantization().set_qinvariant(out_qinvariant);
    }
    else
    {
        Op::quantize(node);
    }

    if (node->inputs().size() == 3)
    {
        float in_scale0 = node->inputs()[0]->quantization().scale();
        float in_scale1 = node->inputs()[1]->quantization().scale();
        float in_scale2 = node->inputs()[2]->quantization().scale();
        float out_scale = node->outputs()[0]->quantization().scale();
        float branch0_scale = std::abs(alpha) / (in_scale0 * in_scale1);
        float branch1_scale = std::abs(beta) / in_scale2;
        float max_inp_scale = std::max(branch0_scale, branch1_scale);

        int32_t q_branch0_scale = (int)((max_inp_scale / branch0_scale) * 256);
        int32_t q_branch1_scale = (int)((max_inp_scale / branch1_scale) * 256);

        float total_scale = out_scale / max_inp_scale;
        float do_scale;
        int8_t do_shift;
        int out_bit = node->outputs()[0]->quantization().bits();
        quant_utils::get_scale_approximation_params(total_scale, do_scale, do_shift, out_bit);
        auto scale_dtype = dtype_utils::bits_to_dtype(out_bit, false);
        node->params()["scale_value"] = {(int)do_scale, q_branch0_scale, q_branch1_scale};
        node->params()["shift_value"] = (int)do_shift + 8;
        node->params()["scale_type"] = {scale_dtype, aipubt::Tensor::UINT16, aipubt::Tensor::UINT16};
        node->params()["shift_type"] = aipubt::Tensor::INT8;
    }
    else if (node->inputs().size() == 2)
    {
        float in_scale0 = node->inputs()[0]->quantization().scale();
        float in_scale1 = node->inputs()[1]->quantization().scale();
        float out_scale = node->outputs()[0]->quantization().scale();

        float do_scale;
        int8_t do_shift;
        float total_scale = std::abs(alpha) * out_scale / (in_scale0 * in_scale1);
        quant_utils::get_scale_approximation_params(total_scale, do_scale, do_shift, inp_bit);

        auto scale_dtype = dtype_utils::bits_to_dtype(inp_bit, true);

        node->params()["scale_value"] = (int)do_scale;
        node->params()["shift_value"] = do_shift;
        node->params()["scale_type"] = scale_dtype;
        node->params()["shift_type"] = aipubt::Tensor::INT8;
    }
    else
    {
        ERROR() << "Not support yet";
    }
}

REGISTER_QUANTIZATION_OP(Gemm);

} // namespace qtlibn
