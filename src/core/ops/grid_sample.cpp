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
 * @Date: 2022-08-13 11:32:34
 * @Last Modified by:   Aaron Gao
 * @Last Modified time: 2022-08-13 11:32:34
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    TensorPtr grid_sample(TensorPtr input,
                          TensorPtr grid,
                          bool align_corners,
                          GridSampleMethod method,
                          GridSamplePaddingMode padding_mode,
                          const Quantization &quantization)
    {
        bool is_5_dims = input->shape().dim() == 5;
        if ((input->shape().dim() != 4 && input->shape().dim() != 5) || (grid->shape().dim() != 4 && grid->shape().dim() != 5))
        {
            ERROR() << "GridSample's input and grid must be rank 4 or 5 but got rank ("
                    << input->shape().dim() << ", " << grid->shape().dim() << ")!";
            return nullptr;
        }
        if (input->shape()[0] != grid->shape()[0])
        {
            ERROR() << "GridSample's input and grid must be with same size at batch dim but got "
                    << input->shape()[0] << " and " << grid->shape()[0] << "!";
            return nullptr;
        }

        int grid_c_dim = is_5_dims ? 4 : 3;
        int grid_ref = is_5_dims ? 3 : 2;
        if (grid->shape()[grid_c_dim] != grid_ref)
        {
            ERROR() << "GridSample's grid must be with size "
                    << grid_ref
                    << "at channel dim but got "
                    << grid->shape()[grid_c_dim] << "!";
            return nullptr;
        }
        if (is_5_dims && method == GridSampleMethod::Bilinear)
        {
            ERROR() << "5-D GridSample unsuport method 'Bilinear',"
                    << "but suport method 'Linear'";
            return nullptr;
        }
        else if (!is_5_dims && method == GridSampleMethod::Linear)
        {
            ERROR() << "4-D GridSample unsuport method 'Linear',"
                    << "but suport method 'Bilinear'";
            return nullptr;
        }

        std::map<GridSampleMethod, std::string> method_table = {
            {GridSampleMethod::Bicubic, "BICUBIC"},
            {GridSampleMethod::Bilinear, "BILINEAR"},
            {GridSampleMethod::Nearest, "NEAREST"},
            {GridSampleMethod::Linear, "LINEAR"},
        };
        std::string method_str = enum_to_string(method_table, method);

        std::map<GridSamplePaddingMode, std::string> padding_mode_table = {
            {GridSamplePaddingMode::Border, "BORDER"},
            {GridSamplePaddingMode::Reflection, "REFLECTION"},
            {GridSamplePaddingMode::Zeros, "ZEROS"},
        };
        std::string padding_mode_str = enum_to_string(padding_mode_table, padding_mode);

        auto g = input->node()->graph();
        TensorShape new_shape(input->shape());
        new_shape[1] = grid->shape()[1];
        new_shape[2] = grid->shape()[2];
        if (is_5_dims)
        {
            new_shape[3] = grid->shape()[3];
        }

        new_shape.set_layout(input->shape().layout());
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("grid_sample"), new_shape);
        add_tensor_info(input, output, quantization);

        auto node = Node::create(g->get_valid_node_name("grid_sample"), OpType_GridSample);
        node->add_input(input);
        node->add_input(grid);
        node->add_output(output);
        g->add_node(node);

        auto &params = node->params();
        params["align_corners"] = align_corners;
        params["method"] = method_str;
        params["padding_mode"] = padding_mode_str;
        quantize(node);

        return output;
    }

}; // namespace ops
}; // namespace aipubt