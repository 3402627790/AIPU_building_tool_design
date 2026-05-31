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
 * @Date: 2022-03-14 10:24:39
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-08-18 16:12:43
 */

#include "quantization_transform/utils/groupnorm_utils.h"
#include "core/compute_tensor.h"
namespace groupnorm_utils
{
template <typename Dtype>
static void groupnorm_forward_cast(aipubt::NodePtr node)
{
    float epsilon = node->params()["epsilon"];
    int group = node->params()["group"];
    std::vector<int> axis = node->params()["axis"];

    //TODO: add is_groupnorm flag and channel_axis: CP-802
    //now only support split group by dim c
    aipubt::TensorShape inp_shape = node->inputs()[0]->shape();
    int inp_n = inp_shape.n();
    int inp_h = inp_shape.h();
    int inp_w = inp_shape.w();
    int inp_c = inp_shape.c();

    if (!node->outputs()[0]->quantization().quantized())
    {
        //default nhwc format
        auto input_tensor = node->inputs()[0];
        auto i_ctensor = aipubt::ComputeTensor::create(input_tensor).reshape(inp_n, inp_h, inp_w, group, inp_c / group);
        //not keep dims for mean and var tensor
        auto mean_ctensor = i_ctensor.mean(axis);
        auto var_ctensor = (i_ctensor - mean_ctensor).pow(2).mean(axis);
        auto ngamma_ctensor = (Dtype)1 / ((Dtype)epsilon + var_ctensor).sqrt();
        auto normalized_ctensor = ((i_ctensor - mean_ctensor) * ngamma_ctensor).reshape(inp_n, inp_h, inp_w, inp_c);

        node->placeholders().push_back(ngamma_ctensor);
        node->placeholders().push_back(normalized_ctensor);

        auto gamma = aipubt::ComputeTensor::create<Dtype>(1.0);
        auto beta = aipubt::ComputeTensor::create<Dtype>(0.0);

        if (node->constants().count("weights") && node->constants()["weights"]->data<Dtype>())
        {
            aipubt::TensorPtr weights = node->constants()["weights"];
            if (!aipubt::Tensor::is_float(weights->type()))
            {
                auto cweights = aipubt::ComputeTensor::zeros(weights->shape(), aipubt::Tensor::FP32);
                for (int i = 0; i < weights->shape().size(); i++)
                {
                    cweights->data<float>()[i] = static_cast<float>(weights->data<int>()[i]);
                }
                gamma = (Dtype)weights->quantization().offset() + cweights;
            }
            else
            {
                gamma = (Dtype)weights->quantization().offset() + aipubt::ComputeTensor::create(weights);
            }
        }
        if (node->constants().count("biases") && node->constants()["biases"]->data<Dtype>())
        {
            aipubt::TensorPtr biases = node->constants()["biases"];
            if (!aipubt::Tensor::is_float(biases->type()))
            {
                auto cbiases = aipubt::ComputeTensor::zeros(biases->shape(), aipubt::Tensor::FP32);
                for (int i = 0; i < cbiases->shape().size(); i++)
                {
                    cbiases->data<float>()[i] = static_cast<float>(biases->data<int>()[i]);
                }
                beta = (Dtype)node->constants()["biases"]->quantization().offset() + cbiases;
            }
            else
            {
                beta = (Dtype)node->constants()["biases"]->quantization().offset() + aipubt::ComputeTensor::create(node->constants()["biases"]);
            }
        }
        auto output_ctensor = beta + gamma * normalized_ctensor;
        node->outputs()[0]->share_data_from(*output_ctensor);
    }
    //Todo: add quantized forward
}

static std::map<aipubt::Tensor::Dtype, std::function<void(aipubt::NodePtr)>> DTYPE_TO_FORWARD_MAP{
    {aipubt::Tensor::UINT8, groupnorm_forward_cast<uint8_t>},
    {aipubt::Tensor::INT8, groupnorm_forward_cast<int8_t>},
    {aipubt::Tensor::UINT16, groupnorm_forward_cast<uint16_t>},
    {aipubt::Tensor::INT16, groupnorm_forward_cast<int16_t>},
    {aipubt::Tensor::UINT32, groupnorm_forward_cast<uint32_t>},
    {aipubt::Tensor::INT32, groupnorm_forward_cast<int32_t>},
    {aipubt::Tensor::FP32, groupnorm_forward_cast<float_t>},
};

void groupnorm_forward(aipubt::NodePtr node)
{
    // aipubt::Tensor::Dtype dtype = node->inputs()[0]->type();
    auto dtype = aipubt::Tensor::FP32;
    DTYPE_TO_FORWARD_MAP[dtype](node);
}

void groupnorm_quantize(aipubt::NodePtr node)
{
    int q_out_bits = node->outputs()[0]->quantization().bits();

    auto aipu_ngamma_scale = node->placeholders()[0]->quantization().scale();
    auto aipu_normalized_scale = node->placeholders()[1]->quantization().scale();
    auto aipu_input_scale = node->inputs()[0]->quantization().scale();
    auto aipu_output_scale = node->outputs()[0]->quantization().scale();

    int ngamma_multiplier_bits = std::max(16, q_out_bits);

    float ngamma_do_scale;
    int8_t ngamma_do_shift;
    float ts0 = aipu_input_scale * aipu_ngamma_scale;

    quant_utils::get_scale_approximation_params(ts0, ngamma_do_scale, ngamma_do_shift, ngamma_multiplier_bits);
    node->params()["ngamma_scale_value"] = static_cast<uint32_t>(ngamma_do_scale);
    node->params()["ngamma_scale_type"] = dtype_utils::bits_to_dtype(ngamma_multiplier_bits, false);
    node->params()["ngamma_shift_value"] = ngamma_do_shift;
    node->params()["ngamma_shift_type"] = aipubt::Tensor::INT8;
    node->params()["ngamma_zp_value"] = static_cast<int8_t>(node->placeholders()[0]->quantization().offset());
    node->params()["ngamma_zp_type"] = dtype_utils::bits_to_dtype(q_out_bits, false);

    float norm_do_scale;
    int8_t norm_do_shift;
    float ts1 = aipu_normalized_scale / (aipu_input_scale * aipu_ngamma_scale);

    quant_utils::get_scale_approximation_params(ts1, norm_do_scale, norm_do_shift, q_out_bits);
    node->params()["norm_scale_value"] = static_cast<uint32_t>(norm_do_scale);
    node->params()["norm_scale_type"] = dtype_utils::bits_to_dtype(q_out_bits, false);
    node->params()["norm_shift_value"] = norm_do_shift;
    node->params()["norm_shift_type"] = aipubt::Tensor::INT8;

    float w_scale = 1.0;
    //TODO: if weights in constant
    float do_scale;
    int8_t do_shift;
    float ts2 = aipu_output_scale / (aipu_normalized_scale * w_scale);
    quant_utils::get_scale_approximation_params(ts2, do_scale, do_shift, q_out_bits);
    node->params()["scale_value"] = static_cast<uint32_t>(do_scale);
    node->params()["scale_type"] = dtype_utils::bits_to_dtype(q_out_bits, false);
    node->params()["shift_value"] = do_shift;
    node->params()["shift_type"] = aipubt::Tensor::INT8;

    //lut
    uint32_t lut_len = 49;
    int16_t *layernorm_lut = new int16_t[lut_len];
    const uint32_t k_lut_len = 49;
    int16_t D_ROM_isqrt[k_lut_len] = {32767, 31790, 30894, 30070, 29309, 28602, 27945, 27330, 26755, 26214, 25705, 25225, 24770, 24339, 23930,
                                      23541, 23170, 22817, 22479, 22155, 21845, 21548, 21263, 20988, 20724, 20470, 20225, 19988, 19760, 19539,
                                      19326, 19119, 18919, 18725, 18536, 18354, 18176, 18004, 17837, 17674, 17515, 17361, 17211, 17064, 16921,
                                      16782, 16646, 16514, 16384};
    memcpy(layernorm_lut, D_ROM_isqrt, lut_len * sizeof(int16_t));
    auto lut_constant = std::make_shared<aipubt::Tensor>(node->name() + "/isqrt_lut", (void *)layernorm_lut, lut_len * sizeof(int16_t));
    aipubt::TensorShape lut_shape(lut_len);
    lut_constant->shape() = lut_shape;
    lut_constant->set_type(aipubt::Tensor::INT16);
    node->constants()["lut"] = lut_constant;
}
}; // namespace groupnorm_utils