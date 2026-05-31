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
 * @Date: 2022-03-07 18:57:23
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2024-01-15 15:19:57
 */

#include "quantization_transform/passes/passes.h"

namespace qtlibn
{

// resume weights for per-channel quantization
template <typename dtype>
static void resume_weights(aipubt::TensorPtr weight)
{
    dtype *wo_data = new dtype[weight->mem_size() / sizeof(dtype)];
    dtype *w_data = weight->data<dtype>();
    memcpy(wo_data, w_data, weight->mem_size());

    weight->set_data(wo_data, weight->mem_size());
}
static std::map<aipubt::Tensor::Dtype, std::function<void(aipubt::TensorPtr)>> RESUME_WEIGHTS_MAP{
    {aipubt::Tensor::INT8, resume_weights<int8_t>},
    {aipubt::Tensor::UINT8, resume_weights<uint8_t>},
    {aipubt::Tensor::INT16, resume_weights<int16_t>},
    {aipubt::Tensor::UINT16, resume_weights<uint16_t>},
    {aipubt::Tensor::INT32, resume_weights<int32_t>},
    {aipubt::Tensor::UINT32, resume_weights<uint32_t>},
};

void convtranspose_postpass(aipubt::GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto &node : nodes)
    {
        if (node->type() == aipubt::OpType_ConvTranspose || node->type() == aipubt::OpType_ConvTranspose3D)
        {
            aipubt::TensorShape ot_shape = node->outputs()[0]->shape();
            aipubt::TensorPtr weights = node->constants()["weights"];
            aipubt::TensorShape w_shape = weights->shape();
            if (node->attrs().count("repeat_weights") && node->get_attr<bool>("repeat_weights") == true)
            {
                int group = node->params()["group"];
                aipubt::Tensor::Dtype w_dtype = weights->quantization().aipu_dtype();
                w_shape[0] = w_shape[0] / group;
                weights->shape() = w_shape;
                RESUME_WEIGHTS_MAP[w_dtype](weights);
                node->erase_attr("repeat_weights");
            }
        }
    }
}
} // namespace qtlibn
