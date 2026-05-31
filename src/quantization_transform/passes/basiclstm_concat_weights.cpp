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
 * @Date: 2020-12-01 14:24:51
 * @Last Modified by: Ruodi Lu
 * @Last Modified time: 2022-6-15 15:12:34
 */

#include "quantization_transform/passes/passes.h"

namespace qtlibn
{
template <typename new_dtype>
float *get_float_data(aipubt::TensorPtr tensor)
{
    new_dtype *tensor_data = tensor->data<new_dtype>();
    int32_t tensor_zp = tensor->quantization().offset();
    float tensor_scale = tensor->quantization().scale();
    uint32_t num = tensor->mem_size() / sizeof(new_dtype);
    float *new_data = new float[num];
    for (uint32_t i = 0; i < num; i++)
    {
        new_data[i] = (tensor_data[i] - tensor_zp) * tensor_scale;
    }
    return new_data;
}
static std::map<aipubt::Tensor::Dtype, std::function<float *(aipubt::TensorPtr)>> GET_FLOAT_DATA_MAP{
    {aipubt::Tensor::INT8, get_float_data<int8_t>},
    {aipubt::Tensor::UINT8, get_float_data<uint8_t>},
    {aipubt::Tensor::INT16, get_float_data<int16_t>},
    {aipubt::Tensor::UINT16, get_float_data<uint16_t>}};
void concat_weights(aipubt::NodePtr node)
{
    aipubt::TensorPtr input_weights = node->constants()["input_weights"];
    aipubt::TensorPtr recurrence_weights = node->constants()["recurrence_weights"];

    aipubt::Tensor::Dtype data_type = input_weights->type();
    float *f_input_weights_data = GET_FLOAT_DATA_MAP[data_type](input_weights);           //h*x
    float *f_recurrence_weights_data = GET_FLOAT_DATA_MAP[data_type](recurrence_weights); //h*h

    uint32_t cell_size = input_weights->shape()[0] / 4;                    //h
    size_t data_size = input_weights->size() + recurrence_weights->size(); // 4*h*(h+x)
    uint32_t x = data_size / cell_size / 4 - cell_size;                    //x
    std::vector<int> ts;
    ts.push_back(4 * cell_size);
    ts.push_back(cell_size + x);
    aipubt::TensorShape t_shape(ts);
    aipubt::TensorPtr weights_tensor = std::make_shared<aipubt::Tensor>(node->name() + "weights", t_shape, aipubt::Tensor::FP32);
    float *weights_data = new float[data_size]; //h* (x+h)
    for (uint32_t i = 0; i < 4 * cell_size; i++)
    {
        for (uint32_t xc = 0; xc < x; xc++)
        {
            *(weights_data + i * (cell_size + x) + xc) = *(f_input_weights_data + i * x + xc);
        }
        for (uint32_t hc = 0; hc < cell_size; hc++)
        {
            *(weights_data + i * (cell_size + x) + hc) = *(f_recurrence_weights_data + i * cell_size + hc);
        }
    }
    weights_tensor->set_data(weights_data, data_size * sizeof(float_t));
    weights_tensor->set_type(aipubt::Tensor::FP32);
    node->constants()["weights"] = weights_tensor;
    weights_tensor->quantization() = node->constants()["input_weights"]->quantization();
    node->constants()["weights"]->quantization().set_scale(1.0);
    node->constants()["weights"]->quantization().set_offset(0);
    node->constants()["weights"]->quantization().set_bits(dtype_utils::dtype_size(data_type) * 8);
    dtype_utils::set_aipu_dtype(node->constants()["weights"], true, false);
    delete[] f_input_weights_data;
    delete[] f_recurrence_weights_data;
}
void basiclstm_concat_weights(aipubt::GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto &node : nodes)
    {
        if (node->type() == aipubt::OpType_BasicLSTM)
        {
            if (node->constants().count("input_weights") && node->constants().count("recurrence_weights"))
            {
                concat_weights(node);
                node->constants().erase("input_weights");
                node->constants().erase("recurrence_weights");
            }
        }
    }
}

} //namespace qtlibn
