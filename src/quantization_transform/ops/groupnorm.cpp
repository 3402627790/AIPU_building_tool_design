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
 * @Author: Kejia Qiang
 * @Date: 2022-05-11 11:03:30
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-12-02 16:29:30
 */

#include "quantization_transform/ops/groupnorm.h"
#include "quantization_transform/utils/math_utils.h"

namespace qtlibn
{
void GroupNormOp::quantize(aipubt::NodePtr node)
{
    aipubt::ComputeTensor inp = node->inputs()[0];
    int q_bits_activation = node->outputs()[0]->quantization().bits();

    std::string normalized_name = node->name() + "_qt_normalized";
    aipubt::TensorPtr normalized_t = std::make_shared<aipubt::Tensor>(normalized_name,
                                                                      aipubt::TensorShape(1),
                                                                      aipubt::Tensor::FP32);
    normalized_t->quantization() = node->inputs()[0]->quantization();
    normalized_t->quantization().set_bits(q_bits_activation);

    int32_t group = param_utils::get_param(node->params(), "group", true, aipubt::NodeParamValue(1));

    bool is_groupnorm = node->type() == aipubt::OpType_GroupNorm ? true : false;
    std::vector<int32_t> axes;
    int32_t dim_num = node->inputs()[0]->shape().dim();
    if (is_groupnorm)
    {
        for (int32_t i = 1; i < dim_num; i++)
        {
            axes.push_back(i);
        }
    }
    else if (!node->params().count("axis"))
    {
        for (int32_t i = 1; i < dim_num - 1; i++)
        {
            axes.push_back(i);
        }
    }
    else
    {
        axes = node->params()["axis"];
    }
    std::vector<int32_t> scope_norm;
    int32_t scope_prod = 1;
    int32_t sn = 0;
    for (int32_t i = 0; i < dim_num; i++)
    {
        if (std::find(axes.begin(), axes.end(), i) == axes.end())
        {
            sn = 1;
        }
        else
        {
            sn = node->inputs()[0]->shape()[i];
        }
        scope_norm.push_back(sn);
        scope_prod *= sn;
    }

    int32_t multiplier_bits = q_bits_activation;
    if (node->attrs().count("multiplier_bits"))
    {
        multiplier_bits = (int)node->get_attr("multiplier_bits");
    }

    int32_t prod_n = scope_prod / group;
    float sqrt_n = std::sqrt(prod_n);

    float alpha = 1.0;
    int8_t adj = 0;
    if (q_bits_activation <= 8 && sqrt_n > 127.)
    {
        adj = std::ceil(std::log2(127.5 / sqrt_n / 4));
        alpha = std::pow(2, adj);
    }

    sqrt_n *= alpha;
    normalized_t->quantization().set_mins({0 - sqrt_n});
    normalized_t->quantization().set_maxs({sqrt_n});
    normalized_t->quantization().set_quantized(false);
    normalized_t->quantization().set_scales({});
    normalized_t->quantization().set_offsets({});
    node->placeholders().push_back(normalized_t);

    dtype_utils::set_aipu_dtype(node->outputs()[0], true);
    dtype_utils::set_aipu_dtype(normalized_t, true);
    Op::quantize(node);
    // INFO() << "layernorm normalized qinfos: " << normalized_t->quantization();
    // INFO() << "output qinfos: " << node->outputs()[0]->quantization();

    if (q_bits_activation <= 8)
    {
        int32_t ceil_exp = std::ceil(std::log2(prod_n));
        int32_t complement = 6;

        int32_t var_shift = (30 + ceil_exp - 31 - (complement >> 1));
        var_shift += (((var_shift & 1) == 1) ? 1 : 0);
        node->params()["var_shift_value"] = var_shift;
        node->params()["var_shift_type"] = dtype_utils::range_to_dtype(0, var_shift, false);
        node->params()["norm_scale_value"] = 1;
        node->params()["norm_scale_type"] = dtype_utils::range_to_dtype(0, 1, false);
        node->params()["norm_shift_value"] = (var_shift >> 1) + adj;
        node->params()["norm_shift_type"] = dtype_utils::range_to_dtype(0, var_shift >> 1, false);
    }
    else
    {
        float_t norm_do_scale;
        int8_t norm_do_shift;
        float ts = normalized_t->quantization().scale() / 127.0;
        quant_utils::get_scale_approximation_params(ts, norm_do_scale, norm_do_shift, multiplier_bits);

        node->params()["var_shift_value"] = 0;
        node->params()["var_shift_type"] = aipubt::Tensor::INT8;
        node->params()["norm_scale_value"] = static_cast<int32_t>(norm_do_scale);
        node->params()["norm_scale_type"] = dtype_utils::range_to_dtype(0, norm_do_scale, false);
        node->params()["norm_shift_value"] = norm_do_shift;
        node->params()["norm_shift_type"] = aipubt::Tensor::INT8;
    }

    if (node->constants().count("weights"))
    {
        // Copy input tensor
        auto bk_input_type = node->inputs()[0]->type();
        auto bk_inp_quantization = node->inputs()[0]->quantization();
        node->inputs()[0]->set_type(normalized_t->type());
        node->inputs()[0]->quantization() = normalized_t->quantization();

        std::string unify_shift_mode = node->attrs().count("unify_shifts_mode") ? (std::string)node->get_attr<aipubt::NodeParamValue>("unify_shifts_mode") : "none";
        if (node->type() != aipubt::OpType_GroupNorm)
        {
            node->set_attr("unify_shifts_mode", aipubt::NodeParamValue("none"));
            DEBUG() << "change unify_shifts_mode from " << unify_shift_mode << " to none";
        }
        node->set_attr("disable_absorb_input_zp_to_bias", true);
        BaseLinearOp::quantize(node);
        node->set_attr("unify_shifts_for_aiff", unify_shift_mode);
        node->inputs()[0]->set_type(bk_input_type);
        node->inputs()[0]->quantization() = bk_inp_quantization;
    }
    else
    {
        float out_do_scale;
        int8_t out_do_shift;
        float ts = node->outputs()[0]->quantization().scale() / normalized_t->quantization().scale();
        quant_utils::get_scale_approximation_params(ts, out_do_scale, out_do_shift, multiplier_bits);
        node->params()["scale_value"] = static_cast<int32_t>(out_do_scale);
        node->params()["scale_type"] = dtype_utils::range_to_dtype(0.0, out_do_scale, true);
        node->params()["shift_value"] = static_cast<int32_t>(out_do_shift);
        node->params()["shift_type"] = dtype_utils::range_to_dtype(0.0, (float)out_do_shift, true);
    }
    node->constants()["lut"] = aipubt::ComputeTensor(math_utils::inverse_sqrt_table_193).astype(aipubt::Tensor::INT16);
    node->constants()["lut"]->set_type(aipubt::Tensor::INT16);
}

void GroupNormOp::approximate(aipubt::NodePtr node)
{
    Op::approximate(node);
}

REGISTER_QUANTIZATION_OP(GroupNorm);

}; // namespace qtlibn
