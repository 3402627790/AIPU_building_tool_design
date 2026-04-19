
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
 * @Author: ALex Zheng
 * @Date: 2021-07-07 16:26:26
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-08-13 11:33:19
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    TensorPtr data_stride(TensorPtr input,
                          const std::vector<uint32_t> &kernel_,
                          const std::vector<uint32_t> &stride_)
    {
        if (input->shape().dim() != 4)
        {
            ERROR() << "input rank should be 4, which makes h/w stride meaningful!";
            return nullptr;
        }
        uint32_t in_h = input->shape().h();
        uint32_t in_w = input->shape().w();
        TensorShape output_shape(input->shape());
        auto kernel = kernel_;
        auto stride = stride_;
        if (kernel.size() == 1)
        {
            kernel.push_back(kernel[0]);
        }
        if (stride.size() == 1)
        {
            stride.push_back(stride[0]);
        }
        int32_t new_h = in_h / stride[0] * kernel[0];
        if (in_h % stride[0] >= kernel[0])
        {
            new_h += kernel[0];
        }
        output_shape.h() = new_h;
        int32_t new_w = in_w / stride[1] * kernel[1];
        if (in_w % stride[1] >= kernel[1])
        {
            new_w += kernel[1];
        }
        output_shape.w() = new_w;
        auto g = input->node()->graph();
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("data_stride"), output_shape);
        add_tensor_info(input, output);
        output->set_type(input->type());
        output->quantization() = input->quantization();

        auto node = Node::create(g->get_valid_node_name("data_stride"), OpType_DataStride);
        node->add_input(input);
        node->add_output(output);
        g->add_node(node);
        auto &params = node->params();
        params["kernel_y"] = kernel[0];
        params["kernel_x"] = kernel[1];
        params["stride_y"] = stride[0];
        params["stride_x"] = stride[1];
        quantize(node);
        return output;
    }
}; // namespace ops
}; // namespace aipubt
