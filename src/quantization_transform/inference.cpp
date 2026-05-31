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
 * @Date: 2022-11-22 20:11:40
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-04-07 11:13:26
 */

/*
    1. input tensor already has input data, like:
    input1=Tensor("input1",np.fromfile("./input.bin", dtype="uint8").reshape([2, 2, 2, 8]))
    2. input tensor has int8/uint8/int16/uint16/ datatype, needs to dequantize
*/

#include "quantization_transform/inference.h"
#include "engine/engine.h"
#include "quantization_transform/observer.h"
#include "quantization_transform/utils/data_type.h"
#include "quantization_transform/utils/quantize_utils.h"

namespace qtlibn
{

void Inferencer::prepare(aipubt::NodePtr node)
{
    std::function<float(float, float, int32_t)> dequantize_formula = quant_utils::dequantize_formula_aipu;
    auto _dequantize = [](aipubt::TensorPtr t, std::function<float(float, float, int32_t)> func) -> void {
        if (!aipubt::Tensor::is_float(t->type()) && t->data<int8_t>())
        {
            quant_utils::primitive_dequantize(t, func);
        }
    };
    for (auto &i_t : node->inputs())
    {
        if (i_t->data<float>())
        {
            _dequantize(i_t, dequantize_formula);
        }
        else
        {
            // generate the float input dataset for node
            auto dataset = new Dataset(i_t);
            dataset->generate_data();
            auto inp_datas = dataset->get_item();
            auto inp_data = inp_datas[0];
            auto inp_shape = i_t->shape();
            i_t->set_type(aipubt::Tensor::FP32);
            i_t->set_data(inp_data, inp_shape.size() * sizeof(float_t));
        }
    }
    for (auto &c_t : node->constants())
    {
        _dequantize(c_t.second, dequantize_formula);
    }

    for (auto &o_t : node->outputs())
    {
        o_t->set_type(aipubt::Tensor::FP32);
    }
}

void Inferencer::update(aipubt::NodePtr node, aipubt::NodePtr from_node)
{
    for (auto &plh_t : from_node->placeholders())
    {
        node->placeholders().push_back(std::move(plh_t));
    }
    auto from_outs = from_node->outputs();
    // how to deal with the tensor->type()
    for (int32_t idx = 0; idx < (int32_t)from_outs.size(); idx++)
    {
        node->outputs()[idx]->share_data_from(*from_outs[idx]);
    }
}

void Inferencer::inference(aipubt::NodePtr node, std::string forward_mode)
{
    /* if forward_mode == float, the input/constant data should be float(but this also has a issue todo, like gather, the one input is fixed
     * to integer, should not float dtype). and defaultly forward_mode == float, so opsapi can directly to call the float forward for quantization
     * if forward_mode != float, we assume all data dtype of tensor is ok, and egine will check this using criteria function.
     * and in quantized forward, we will not statistic and update.
     */
    std::transform(forward_mode.begin(), forward_mode.end(), forward_mode.begin(), ::toupper);

    auto forward_node = node->clone();
    if (forward_mode == "FLOAT")
    {
        // 0. mainly check whether the node has quantized data or data type.
        // 1. get inference data if needed
        prepare(forward_node);
    }

    aipubt::engine::run_node(forward_node);

    if (forward_mode == "FLOAT")
    {
        statistic(forward_node);
        update(node, forward_node);
    }
}

} // namespace qtlibn
