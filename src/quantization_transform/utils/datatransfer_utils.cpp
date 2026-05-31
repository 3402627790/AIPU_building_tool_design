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
 * @Date: 2022-04-19 10:43:10
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-09-01 21:01:51
 */

#include "quantization_transform/utils/datatransfer_utils.h"

namespace datatransfer_utils
{
static void _update_tensor_data(aipubt::TensorPtr tensor, float data, int32_t num_output)
{
    uint32_t num = tensor->shape().size();
    float *data_t = new float[num];
    for (uint32_t i = 0; i < num; i++)
    {
        data_t[i] = data;
    }
    std::vector<float> scales(num_output, 1.0);
    std::vector<int32_t> offsets(num_output, 0);
    std::vector<float> mins(num_output, data);
    std::vector<float> maxs(num_output, data);
    tensor->set_data(data_t, num * sizeof(float));
    tensor->quantization().set_scales(scales);
    tensor->quantization().set_offsets(offsets);
    tensor->quantization().set_mins(mins);
    tensor->quantization().set_maxs(maxs);
    tensor->set_type(aipubt::Tensor::FP32);
    // tensor->quantization().set_primitive_dtype(aipubt::Tensor::FP32);
    dtype_utils::set_aipu_dtype(tensor, true, false);
}

static void _data_transfer_output(aipubt::NodePtr node)
{
    aipubt::NodePtr node_i = node->parents()[0];
    auto primitive_scale = node_i->get_attr<std::vector<aipubt::NodeParamValue>>(Constants::original_scale);
    auto primitive_offset = node_i->get_attr<std::vector<aipubt::NodeParamValue>>(Constants::original_zp);
    auto aipu_scale = node->inputs()[0]->quantization().scales();
    auto aipu_offset = node->inputs()[0]->quantization().offsets();
    //default one output
    //(pri_q - pre_zp) * pre_s = f = (aipu_q + aipu_zp) /aipu_s
    int32_t num_output = node->params()["num_output"];
    float weight = 1.0 / ((float)aipu_scale[0] * (float)primitive_scale[0]);
    float bias = (float)aipu_offset[0] * weight + (float)primitive_offset[0];
    node->inputs()[0]->quantization().set_scale(1.0);
    aipubt::TensorPtr weight_t = node->constants()["weights"];
    aipubt::TensorPtr bias_t = node->constants()["biases"];
    _update_tensor_data(weight_t, weight, num_output);
    _update_tensor_data(bias_t, bias, num_output);
}

static void _data_transfer_input(aipubt::NodePtr node)
{
    auto primitive_scale = node->get_attr<std::vector<aipubt::NodeParamValue>>(Constants::original_scale);
    auto primitive_offset = node->get_attr<std::vector<aipubt::NodeParamValue>>(Constants::original_zp);
    auto aipu_scale = node->outputs()[0]->quantization().scales();
    auto aipu_offset = node->outputs()[0]->quantization().offsets();

    int32_t num_output = node->params()["num_output"]; //output_num = scale_size
    //default one output
    //(pri_q - pre_zp) * pre_s = f = (aipu_q + aipu_zp) /aipu_s
    float weight = (float)aipu_scale[0] * (float)primitive_scale[0];
    float bias = 0.0 - (float)primitive_offset[0] * weight - (float)aipu_offset[0];
    node->outputs()[0]->quantization().set_scale(1.0);
    aipubt::TensorPtr weight_t = node->constants()["weights"];
    aipubt::TensorPtr bias_t = node->constants()["biases"];
    _update_tensor_data(weight_t, weight, num_output);
    _update_tensor_data(bias_t, bias, num_output);
}

static std::map<std::string, std::function<void(aipubt::NodePtr)>> _DATA_TRANSFER = {
    {"input", _data_transfer_input},
    {"output", _data_transfer_output},
};
void data_transfer(std::string mode, aipubt::NodePtr node)
{
    _DATA_TRANSFER[mode](node);
}
}; // namespace datatransfer_utils