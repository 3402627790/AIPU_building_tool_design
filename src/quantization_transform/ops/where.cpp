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
 * @Date: 2022-04-11 14:52:35
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-04-20 15:58:01
 */

#include "quantization_transform/ops/where.h"

namespace qtlibn
{

void WhereOp::quantize(aipubt::NodePtr node)
{
    if (node->inputs().size() > 1)
    {
        // 3 inputs:{condition, x1, x2} and 1 output  ==> out = select(condition_tensor, x1_tensor, x2_tensor)
        auto inp1_t = node->inputs()[1];
        auto inp2_t = node->inputs()[2];
        auto out_t = node->outputs()[0];

        auto inp1_qinvariant = inp1_t->quantization().qinvariant();
        auto inp2_qinvariant = inp2_t->quantization().qinvariant();
        auto inp1_dtype = inp1_t->quantization().aipu_dtype();
        auto inp2_dtype = inp2_t->quantization().aipu_dtype();

        if (inp1_qinvariant != inp2_qinvariant)
        {
            WARN() << "Except for the qinvariant of second input is same to the third input, but now is not and may cause accuracy issue.";
        }
        auto out_signed = aipubt::Tensor::is_signed(inp1_dtype) || aipubt::Tensor::is_signed(inp2_dtype);

        if (inp2_qinvariant && inp1_qinvariant)
        {
            out_t->quantization().set_scales({1.0});
            out_t->quantization().set_offsets({0});
            auto o_bits = std::max(inp1_t->quantization().bits(), inp2_t->quantization().bits());
            out_t->quantization().set_bits(o_bits);
            dtype_utils::set_aipu_dtype(out_t, out_signed);
            out_t->quantization().set_qinvariant(true);
        }
        else
        {
            auto o_bits = out_t->quantization().bits();
            dtype_utils::set_aipu_dtype(out_t, out_signed);

            Op::quantize(node);

            auto o_scale = out_t->quantization().scale();
            auto i1_scale = inp1_t->quantization().scale();
            auto i2_scale = inp2_t->quantization().scale();
            float ts1 = o_scale / i1_scale;
            float ts2 = o_scale / i2_scale;

            if (inp2_qinvariant && (!inp1_qinvariant))
            {
                ts2 = 1.0;
            }
            if (inp1_qinvariant && (!inp2_qinvariant))
            {
                ts1 = 1.0;
            }

            float do_scale1, do_scale2;
            int8_t do_shift1, do_shift2;
            int32_t multiplier_bits = o_bits;
            if (node->attrs().count("multiplier_bits"))
            {
                multiplier_bits = node->get_attr("multiplier_bits");
            }
            quant_utils::get_scale_approximation_params(ts1, do_scale1, do_shift1, multiplier_bits);
            quant_utils::get_scale_approximation_params(ts2, do_scale2, do_shift2, multiplier_bits);
            auto unsigned_multiplier_dtype = dtype_utils::bits_to_dtype(multiplier_bits, false);
            node->params()["scale_value"] = std::vector<uint32_t>{static_cast<uint32_t>(do_scale1), static_cast<uint32_t>(do_scale2)};
            node->params()["scale_type"] = std::vector<aipubt::Tensor::Dtype>(2, unsigned_multiplier_dtype);
            node->params()["shift_value"] = std::vector<int8_t>{do_shift1, do_shift2};
            node->params()["shift_type"] = std::vector<aipubt::Tensor::Dtype>(2, aipubt::Tensor::INT8);
        }
    }
    else
    {
        for (auto ot : node->outputs())
        {
            ot->quantization().set_scales({1.0});
            ot->quantization().set_offsets({0});
            int32_t o_bits = ot->quantization().bits() > 16 ? 16 : ot->quantization().bits();
            ot->quantization().set_bits(o_bits);
            dtype_utils::set_aipu_dtype(ot, false);
            ot->quantization().set_qinvariant(true);
        }
    }
}

REGISTER_QUANTIZATION_OP(Where);

} // namespace qtlibn
