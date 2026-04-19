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
 * @Author: Aaron Gao
 * @Date: 2022-04-27 11:19:03
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-08-13 11:33:43
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    TensorPtr crop_and_resize(TensorPtr input,
                              TensorPtr box,
                              TensorPtr box_indices,
                              const std::vector<uint32_t> &crop_size,
                              ResizeMethod method,
                              NodeParamValue extrapolation_value,
                              bool is_perf_mode)
    {
        if (crop_size.size() != 2)
        {
            ERROR() << "CropAndResize crop_size's size should be 2, instead of: " << crop_size.size();
            return nullptr;
        }
        if (input->shape().dim() != 4 || box->shape().dim() != 2 || box_indices->shape().dim() != 1)
        {
            ERROR() << "CropAndResize input, box, box_indices rank should be 4, 2, 1, instead of "
                    << input->shape().dim() << ", "
                    << box->shape().dim() << ", "
                    << box_indices->shape().dim();
            return nullptr;
        }
        if (box->shape()[0] != box_indices->shape()[0] || box->shape()[1] != 4)
        {
            ERROR() << "CropAndResize box[0] should be equal with box_indices[0], and box[1] should be 4";
            return nullptr;
        }

        auto g = input->node()->graph();
        auto node = Node::create(g->get_valid_node_name("crop_and_resize"), OpType_CropAndResize);
        node->params()["crop_size"] = crop_size;
        node->params()["method"] = method_to_string(method);

        node->params()["extrapolation_value"] = extrapolation_value;
        node->params()["is_perf_mode"] = is_perf_mode;
        bool quantized = false;
        if (extrapolation_value.t != NodeParamValue::NodeParamType_float)
        {
            quantized = true;
        }
        node->set_attr("params_quantized", NodeParamValue(quantized));

        TensorShape shape({box->shape()[0],
                           static_cast<int32_t>(crop_size[0]),
                           static_cast<int32_t>(crop_size[1]),
                           input->shape().c()},
                          input->shape().layout());
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("crop_and_resize"), shape);
        add_tensor_info(input, output);

        node->add_input(input);
        node->add_input(box);
        node->add_input(box_indices);
        node->add_output(output);
        g->add_node(node);
        quantize(node);

        return output;
    }
}; // namespace ops
}; // namespace aipubt