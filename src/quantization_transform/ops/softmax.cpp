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
 * @Date: 2020-09-08 09:58:17
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-05-07 18:07:36
 */

#include "quantization_transform/ops/softmax.h"
#include "quantization_transform/utils/math_utils.h"

namespace qtlibn
{
void SoftmaxOp::generate_softmax_lut(aipubt::NodePtr node, aipubt::TensorPtr in_t)
{
    auto in_bits = in_t->quantization().bits();
    uint32_t lut_len = pow(2, in_bits);
    if (node->attrs().count("lut_items_in_bits"))
    {
        int32_t lut_bits = node->get_attr<aipubt::NodeParamValue>(Constants::lut_items_in_bits);
        lut_len = pow(2, lut_bits);
    }
    if (lut_len > 65535)
    {
        WARN() << "input tensor use more than 16 bits quantization, which will cause softmax's lut tensor very long";
    }
    int32_t max_value_in_bits = 20;
    if (node->attrs().count("scaling_bits"))
    {
        auto scaling_bits = node->get_attr<aipubt::NodeParamValue>("scaling_bits");
        max_value_in_bits = scaling_bits[0];
    }
    float min_val = 0.0;
    float max_val = (1 << max_value_in_bits) - 1;
    float aipu_input_scale = in_t->quantization().scale();
    float aipu_input_zp = in_t->quantization().offset();
    //android has beta attr
    float beta = 1.0;
    if (node->params().count("beta"))
    {
        beta = (float)node->params()["beta"];
        node->params().erase("beta");
    }
    aipubt::Tensor::Dtype lut_dtype = dtype_utils::range_to_dtype(0, max_val, false);
    float max_inp = quant_utils::clip(round(log(max_val) * aipu_input_scale / beta - aipu_input_zp), std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max());
    // float max_inp = quant_utils::clip(round((log(max_val) - aipu_input_zp / aipu_input_scale) * aipu_input_scale / beta - aipu_input_zp), std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max());
    // float max_inp = log(max_val) - aipu_input_zp / aipu_input_scale;
    float step = float(pow(2, in_bits) - 1) / (lut_len - 1);
    int32_t lut_idx = 0;
    float value = 0;
    std::string lut_name = node->name() + "softmax_lut_tensor";
    aipubt::TensorPtr softmax_lut_tensor;
    if (lut_dtype == aipubt::Tensor::UINT16)
    {

        uint16_t *lut = new uint16_t[lut_len];
        for (float idx = 0; idx <= pow(2, in_bits) - 1; idx += step, lut_idx++)
        {
            // value = (idx - pow(2, in_bits) + 1.0 + aipu_input_zp) * beta / aipu_input_scale + max_inp;
            value = (idx + max_inp - pow(2, in_bits) + aipu_input_zp) * beta / aipu_input_scale;
            value = quant_utils::clip(round(exp(value)), min_val, max_val);
            *(lut + lut_idx) = static_cast<uint16_t>(value);
        }
        softmax_lut_tensor = std::make_shared<aipubt::Tensor>(lut_name, (char *)lut, lut_len * sizeof(int16_t));
    }
    else
    {
        //default to uint32_t lut
        uint32_t *lut = new uint32_t[lut_len];
        for (float idx = 0; idx <= pow(2, in_bits) - 1; idx += step, lut_idx++)
        {
            value = (idx + max_inp - pow(2, in_bits) + aipu_input_zp) * beta / aipu_input_scale;
            // value = (idx + max_inp - pow(2, in_bits) + 1.0 + aipu_input_zp) * beta / aipu_input_scale;
            // acc better
            // value = (idx - pow(2, in_bits) + 1.0 + aipu_input_zp) * beta / aipu_input_scale + max_inp;
            // aibenchmark l1 better
            // value = (idx - pow(2, in_bits) + aipu_input_zp) * beta / aipu_input_scale + max_inp;
            value = quant_utils::clip(round(exp(value)), min_val, max_val);
            *(lut + lut_idx) = static_cast<uint32_t>(value);
        }

        softmax_lut_tensor = std::make_shared<aipubt::Tensor>(lut_name, (char *)lut, lut_len * sizeof(uint32_t));
        lut_dtype = aipubt::Tensor::UINT32;
    }
    softmax_lut_tensor->set_type(lut_dtype);
    aipubt::TensorShape lut_shape(lut_len);
    softmax_lut_tensor->shape() = lut_shape;
    softmax_lut_tensor->quantization().set_bits(dtype_utils::dtype_to_bits(lut_dtype));
    node->constants()["lut"] = softmax_lut_tensor;
}

void SoftmaxOp::uniform_quantization_impl(aipubt::NodePtr node)
{
    // set output dtype
    dtype_utils::set_aipu_dtype(node->outputs()[0], false);
    Op::quantize(node);

    //generate softmax lut
    generate_softmax_lut(node, node->inputs()[0]);

    auto aipu_output_scale = node->outputs()[0]->quantization().scale();
    auto out_bits = node->outputs()[0]->quantization().bits();

    float do_scale;
    int8_t do_shift;

    int32_t multiplier_bits = out_bits;
    if (node->attrs().count("multiplier_bits"))
    {
        multiplier_bits = node->get_attr("multiplier_bits");
    }

    quant_utils::get_scale_approximation_params(aipu_output_scale, do_scale, do_shift, multiplier_bits);
    node->params()["scale_value"] = static_cast<uint32_t>(do_scale);
    node->params()["scale_type"] = dtype_utils::bits_to_dtype(multiplier_bits, false);
    node->params()["shift_value"] = do_shift;
    node->params()["shift_type"] = aipubt::Tensor::INT8;
    node->params()["quantize_method"] = "LUT";
}

void SoftmaxOp::log2_quantization_impl(aipubt::NodePtr node)
{
    // set output dtype
    dtype_utils::set_aipu_dtype(node->outputs()[0], false);
    Op::quantize(node);

    node->outputs()[0]->quantization().set_scales({255.0});
    node->outputs()[0]->quantization().set_offsets({0});

    // float beta = node->params().count("beta") ? (float)node->params()["beta"] : 1.0;
    float beta = 1.0;
    if (node->params().count("beta"))
    {
        beta = (float)node->params()["beta"];
        node->params().erase("beta");
    }

    auto inp_t = node->inputs()[0];
    auto inp_scale = inp_t->quantization().scale();
    float inp_rescale = 1.442695 * beta / inp_scale;

    float do_scale;
    int8_t do_shift;

    // auto out_bits = node->outputs()[0]->quantization().bits();
    // int32_t multiplier_bits = out_bits;
    int32_t multiplier_bits = 13;
    if (node->attrs().count("multiplier_bits"))
    {
        multiplier_bits = node->get_attr("multiplier_bits");
    }

    quant_utils::get_scale_approximation_params(inp_rescale, do_scale, do_shift, multiplier_bits);
    node->params()["input_scale_value"] = static_cast<uint32_t>(do_scale);
    node->params()["input_scale_type"] = dtype_utils::bits_to_dtype(multiplier_bits, false);
    node->params()["input_shift_value"] = do_shift;
    node->params()["input_shift_type"] = aipubt::Tensor::INT8;

    auto axis = node->params()["axis"];
    auto axis_shape_value = node->inputs()[0]->shape()[axis];
    auto bits_of_axis_shape_value = std::floor(std::log2((float)axis_shape_value));
    node->params()["pow2_out_q"] = 31 - static_cast<int32_t>(bits_of_axis_shape_value);

    // float new_scale = std::pow(2, out_bits) - 1;
    // node->outputs()[0]->quantization().set_scales({new_scale});
    // node->outputs()[0]->quantization().set_offsets({0});

    int32_t adjust_q = 1;
    if (node->attrs().count("scaling_bits"))
    {
        aipubt::NodeParamValue scaling_bits = node->get_attr<aipubt::NodeParamValue>("scaling_bits");
        adjust_q = scaling_bits[1];
    }
    node->params()["adjust_q"] = adjust_q;
    node->params()["quantize_method"] = "FAST_EXP";

    if (adjust_q >= 8)
    {
        std::string log2_lut_name = node->graph() ? node->graph()->get_valid_tensor_name("softmax_log2_lut") : "softmax_log2_lut";
        auto log2_lut_len = math_utils::table_log2_q15.size();
        aipubt::TensorPtr log2_lut = std::make_shared<aipubt::Tensor>(log2_lut_name, (char *)&(math_utils::table_log2_q15[0]), log2_lut_len * sizeof(int16_t));
        std::string pow2_lut_name = node->graph() ? node->graph()->get_valid_tensor_name("softmax_pow2_lut") : "softmax_pow2_lut";
        auto pow2_lut_len = math_utils::table_pow2_q15.size();
        aipubt::TensorPtr pow2_lut = std::make_shared<aipubt::Tensor>(pow2_lut_name, (char *)&(math_utils::table_pow2_q15[0]), pow2_lut_len * sizeof(uint16_t));

        node->constants()["log2_lut"] = log2_lut;
        node->constants()["pow2_lut"] = pow2_lut;
    }
}

void SoftmaxOp::quantize(aipubt::NodePtr node)
{
    std::string impl_name = "uniform";
    if (node->attrs().count("scaling_bits"))
    {
        aipubt::NodeParamValue scaling_bits = node->get_attr<aipubt::NodeParamValue>("scaling_bits");
        int32_t o_zp = 0;
        if (node->outputs()[0]->quantization().offsets().size() > 0)
        {
            o_zp = node->outputs()[0]->quantization().offset();
        }
        if (scaling_bits.arr.size() == 2 && scaling_bits[0] >= 1 && o_zp == 0)
        {
            impl_name = "log2";
        }
    }
    if (impl_name == "uniform")
    {
        uniform_quantization_impl(node);
    }
    else if (impl_name == "log2")
    {
        log2_quantization_impl(node);
    }
    else
    {
        ERROR() << node << ": now only support [uniform, log2] quantization implementation.";
    }
}

void SoftmaxOp::approximate(aipubt::NodePtr node)
{
    Op::approximate(node);
    int32_t approx_params = -1;
    if (node->attrs().count("approx_params"))
    {
        approx_params = node->get_attr<aipubt::NodeParamValue>("approx_params");
    }
    if (approx_params == 1)
    {
        int32_t lut_items_in_bits = dtype_utils::dtype_to_bits(node->inputs()[0]->quantization().aipu_dtype());
        if (node->attrs().count("lut_items_in_bits"))
        {
            lut_items_in_bits = node->get_attr("lut_items_in_bits");
        }
        int32_t step = std::pow(2, lut_items_in_bits) + 1;
        aipubt::ComputeTensor lut = aipubt::ComputeTensor::linspace(0.0, 1.0, step);
        aipubt::ComputeTensor pow2 = aipubt::ComputeTensor(float(2.0));
        lut = pow2.pow(lut).astype(aipubt::Tensor::FP32);
        lut = dtype_utils::to_fp24(lut);
        node->constants()["lut"] = lut;
        node->params()["is_perf_mode"] = true;
        node->params()["lut_mode"] = "EXP";
    }
    else
    {
        node->params()["is_perf_mode"] = false;
    }
}

REGISTER_QUANTIZATION_OP(Softmax);
}; // namespace qtlibn
