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
 * @Author: Alex Zheng
 * @Date: 2022-12-02 11:04:33
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2022-12-02 11:32:47
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
#include "ops_internal.h"
namespace aipubt
{
namespace ops
{
static TensorPtr grayscale (OpType op_type,
                            TensorPtr input,
                            TensorPtr weights,
                            const std::vector<int32_t> &strides,
                            Padding padding,
                            const std::vector<int32_t> &dilations,
                            const Quantization &quant)
{
    GraphPtr g = input->node()->graph();
    // 1. paramter check
    if (input->shape().c() != weights->shape()[0])
    {
        ERROR() << "Invalid input channel  input c=" << input->shape().c()
        << " weight depth = " <<  weights->shape()[0];
        return nullptr;
    }
    auto in_dim = input->shape().dim();
    if (in_dim < 4 || in_dim > 5)
    {
        ERROR() << "Invalid inputs dimension " << in_dim << ", expected 4 or 5";
        return nullptr;
    }
    if ((weights->shape().dim() != 3)
       &&(weights->shape().dim() != 4))
    {
        ERROR() << "Invalid weights dimension " << weights->shape().dim() << ", expected 3,4";
        return nullptr;
    }
    if (weights->shape().dim() == 3)
    {
        weights->shape().reshape(weights->shape().n(),weights->shape().h(),weights->shape().w(),1);
    }

    if ((weights->shape().dim() == 4) && (weights->shape()[3] != 1))
    {
        ERROR() << "Invalid weights shape " << weights->shape() << ", expected [C,H,W,1]";
        return nullptr;
    }
    // 2. calculate pad value
    int32_t kernel_y = weights->shape().h();
    int32_t kernel_x = weights->shape().w();
    if (padding.size() == 0)
    {
        padding.values.resize(4, 0);
        if(padding == "SAME")
        {
            cal_explicit_padding(input->shape().h(), strides[0], kernel_y, padding, padding[0], padding[1]);
            cal_explicit_padding(input->shape().w(), strides[1], kernel_x, padding, padding[2], padding[3]);
        }
    }
    int32_t dilation_y = 1;
    int32_t dilation_x = 1;
    if(dilations.size()==2)
    {
        dilation_y = dilations[0];
        dilation_x = dilations[1];
    }
    else if (dilations.size()==1)
    {
        dilation_y = dilations[0];
        dilation_x = dilations[0];
    }

    std::vector<int32_t> valid_strides = {1, 1};
    if (extract_valid_stride(strides, valid_strides) < 0)
    {
        ERROR() << "Unsupport strides size";
        return nullptr;
    }
    std::map<OpType, std::string> optype_to_string = 
    {
        {OpType_Dilation, "dilation"},
        {OpType_Erosion, "erosion"},
    };
    // 3. calculate output shape
    int32_t output_n = input->shape().n();
    int32_t output_h = compute_out_size(input->shape().h(), kernel_y, valid_strides[0],
                                        dilation_y, padding[0], padding[1]);
    int32_t output_w = compute_out_size(input->shape().w(), kernel_x, valid_strides[1],
                                        dilation_x, padding[2], padding[3]);
    int32_t output_c = input->shape().c();
    TensorShape new_shape(output_n, output_h, output_w, output_c);
    new_shape.set_layout(input->shape().layout());
    auto output = std::make_shared<Tensor>(g->get_valid_tensor_name(optype_to_string[op_type]), new_shape);
    add_tensor_info(input, output, quant);

    auto node = Node::create(g->get_valid_node_name(optype_to_string[op_type]), op_type);
    node->add_input(input);
    node->add_output(output);

    auto &params = node->params();
    params["kernel_y"] = kernel_y;
    params["kernel_x"] = kernel_x;
    params["stride_y"] = valid_strides[0];
    params["stride_x"] = valid_strides[1];
    params["pad_top"] = padding[0];
    params["pad_bottom"] = padding[1];
    params["pad_left"] = padding[2];
    params["pad_right"] = padding[3];
    params["dilation_y"] = dilation_y;
    params["dilation_x"] = dilation_x;

    node->constants()["weights"] = weights;
    g->add_node(node);
    quantize(node);
    DEBUG() << "output shape :"<< new_shape;
    return output;
}

TensorPtr dilation (TensorPtr input,
                    TensorPtr weights,
                    const std::vector<int32_t> &strides,
                    Padding padding,
                    const std::vector<int32_t> &dilations,
                    const Quantization &quant)
{
    return grayscale(OpType_Dilation, input, weights, strides, padding, dilations, quant);
}

TensorPtr erosion (TensorPtr input,
                   TensorPtr weights,
                   const std::vector<int32_t> &strides,
                   Padding padding,
                   const std::vector<int32_t> &dilations,
                   const Quantization &quant)
{
    return grayscale(OpType_Erosion, input, weights, strides, padding, dilations, quant);
}

}; //namespace ops
}; // namespace aipubt