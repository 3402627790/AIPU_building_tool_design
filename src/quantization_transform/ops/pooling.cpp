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
 * @Date: 2022-04-03 19:28:30
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2024-07-29 16:54:56
 */

#include "quantization_transform/ops/pooling.h"
#include "core/compute_tensor.h"
#include "quantization_transform/generate_lut.h"

namespace qtlibn
{
static aipubt::ComputeTensor sqrt_func(aipubt::ComputeTensor value, std::map<std::string, float> cal_param = {})
{
    auto masked = value < 0.0;
    masked = (1.0 - masked) * value;
    return masked.sqrt();
}

void PoolingOp::quantize(aipubt::NodePtr node)
{
    std::string pmethod = node->params()["method"];
    if (pmethod == "L1")
    {
        dtype_utils::set_aipu_dtype(node->outputs()[0], false);
        Op::quantize(node);
        float ts;
        float do_scale;
        int8_t do_shift;
        ts = node->outputs()[0]->quantization().scale() / node->inputs()[0]->quantization().scale();

        quant_utils::get_scale_approximation_params(ts, do_scale, do_shift, 16, node->name(), false, 15, 63);
        node->params()["scale_value"] = static_cast<uint32_t>(do_scale);
        node->params()["scale_type"] = aipubt::Tensor::UINT16;
        node->params()["shift_value"] = static_cast<uint32_t>(do_shift);
        node->params()["shift_type"] = aipubt::Tensor::INT8;
    }
    else if (pmethod == "L2")
    {
        node->placeholders()[0]->quantization().set_bits(16);
        dtype_utils::set_aipu_dtype(node->outputs()[0], false);
        dtype_utils::set_aipu_dtype(node->placeholders()[0], false);
        Op::quantize(node);
        auto out_scale = node->placeholders()[0]->quantization().scale();
        auto inp_scale = node->inputs()[0]->quantization().scale();
        // auto o_bits = node->outputs()[0]->quantization().bits();
        int32_t multiplier_bits = 16;
        if (node->attrs().count("multiplier_bits"))
        {
            multiplier_bits = node->get_attr("multiplier_bits");
        }
        float ts = out_scale / (inp_scale * inp_scale);
        float do_scale;
        int8_t do_shift;
        quant_utils::get_scale_approximation_params(ts, do_scale, do_shift, multiplier_bits, "L2pooling", false, 15, 63);
        node->params()["scale_value"] = static_cast<uint32_t>(do_scale);
        node->params()["scale_type"] = dtype_utils::bits_to_dtype(multiplier_bits, false);
        node->params()["shift_value"] = do_shift;
        node->params()["shift_type"] = aipubt::Tensor::INT8;
        //sqrt lut
        aipubt::TensorPtr ph = node->placeholders()[0];
        aipubt::TensorPtr out = node->outputs()[0];
        aipubt::Quantization inp_quant = node->inputs()[0]->quantization();
        node->inputs()[0]->quantization() = node->placeholders()[0]->quantization();
        node->constants()["sqrt_lut"] = generate_lut(node, sqrt_func);
        node->inputs()[0]->quantization() = inp_quant;
    }
    else if (pmethod == "AVG")
    {
        int32_t in_zp = node->inputs()[0]->quantization().offset();
        int32_t out_zp = node->outputs()[0]->quantization().offset();

        // it's better to check quantization params same or not, if same, no quantization, then if globalavgpool, scale+shift?
        int32_t kernel_x = node->params()["kernel_x"];
        int32_t kernel_y = node->params()["kernel_y"];
        int32_t in_h = node->inputs()[0]->shape().h();
        int32_t in_w = node->inputs()[0]->shape().w();
        int32_t pad_bottom = node->params()["pad_bottom"];
        int32_t pad_top = node->params()["pad_top"];
        int32_t pad_left = node->params()["pad_left"];
        int32_t pad_right = node->params()["pad_right"];
        bool is_global_avgpool = kernel_x == in_w && kernel_y == in_h && pad_bottom == 0 && pad_left == 0 && pad_right == 0 && pad_top == 0;
        if (is_global_avgpool && !(node->attrs().count("avgpool_from_reducemean") && (bool)node->get_attr("avgpool_from_reducemean")))
        {
            Op::quantize(node);
            float in_scale = node->inputs()[0]->quantization().scale();
            float out_scale = node->outputs()[0]->quantization().scale();
            if (std::abs(in_scale - out_scale) > quant_utils::EPSINON && !(in_zp == 0 && out_zp == 0))
            {
                ERROR() << node << ": pooling now is in_scale(= " << in_scale << ") -  out_scale(= " << out_scale << ") > 0.00001, both zp must be 0, but now in_zp = "
                        << in_zp << ", and out_zp = " << out_zp;
            }

            DEBUG() << "aib original global avgpool (not reducemean split), qtlib will calculate the scale/shift and write to ir.";
            // in global avgpool the input.scale may not be equal to out.scale, so we quantize the output independently.
            // and merge the out.scale/in.scale to doscale/doshift.
            float ts = out_scale / (in_scale * kernel_x * kernel_y);
            int32_t multiplier_bits = node->outputs()[0]->quantization().bits();

            float do_scale;
            int8_t do_shift;
            quant_utils::get_scale_approximation_params(ts, do_scale, do_shift, multiplier_bits);
            node->params()["scale_value"] = static_cast<int32_t>(do_scale);
            node->params()["shift_value"] = do_shift;
            node->params()["scale_type"] = dtype_utils::bytes_to_unsigned_qdtype((multiplier_bits + 7) / 8);
            node->params()["shift_type"] = aipubt::Tensor::INT8;
        }
        else
        {
            NoQuantization::quantize(node);
        }
    }
    else
    {
        NoQuantization::quantize(node);
    }
}

REGISTER_QUANTIZATION_OP(Pooling);

}; // namespace qtlibn
