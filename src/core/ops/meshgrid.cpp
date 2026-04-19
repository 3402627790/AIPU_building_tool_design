
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
 * @Date: 2023-10-20 11:33:19
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2023-10-20 11:33:19
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    std::vector<TensorPtr> meshgrid(std::vector<TensorPtr> inputs,
                                    const std::string indexing,
                                    const bool sparse,
                                    const bool copy,
                                    const std::vector<Quantization> &quants)
    {
        uint32_t input_num = inputs.size();
        std::string indexing_loc = indexing;
        std::vector<uint32_t> input_lens;
        if (input_num < 2)
        {
            ERROR() << "At least 2 inputs are required";
            return {};
        }
        for (uint32_t i=0; i<input_num; ++i)
        {
            if (inputs[i]->shape().dim() != 1)
            {
                ERROR() << "Input Only support 1-dimension." \
                    << "But the " << i << " input dimension is " << inputs[i]->shape().dim() << ".";
                return {};
            }
            input_lens.push_back(inputs[i]->shape()[0]);
        }
        bool copy_loc = copy;
        // Lib only support 'true' now.
        copy_loc = true;

        if(indexing_loc != "xy" && indexing_loc !="ij") {
            indexing_loc = "xy";
            WARN() << "Paramter 'indexing' only support 'xy','ij'. But now it is \"" << indexing \
                   << "\". Set to default value 'xy'";
        }

        std::vector<std::vector<int32_t>> output_dims;
        if (sparse)
        {
            for (uint32_t i =0; i< input_num; ++i)
            {
                std::vector<int32_t> output_dim;
                for(uint32_t j=0; j< input_num; ++j)
                {
                    output_dim.push_back(1);
                }
                uint32_t index = i;
                if(indexing_loc == "xy")
                {
                    if (index == 0)
                    {
                        index =1;
                    }
                    else if (index == 1)
                    {
                        index =0;
                    }
                }
                output_dim[index] = input_lens[i];
                output_dims.push_back(output_dim);
            }
        }
        else
        {
            std::vector<int32_t> output_dim;
            if(indexing_loc == "xy")
            {
                output_dim.push_back(input_lens[1]);
                output_dim.push_back(input_lens[0]);
            }
            else
            {
                output_dim.push_back(input_lens[0]);
                output_dim.push_back(input_lens[1]);
            }
            for(uint32_t i =2; i< input_num; ++i)
            {
                output_dim.push_back(input_lens[i]);
            }
            for(uint32_t i =0; i< input_num; ++i)
            {
                output_dims.push_back(output_dim);
            }
        }

        auto g = inputs[0]->node()->graph();
        std::vector<TensorPtr> outputs;
        for(uint32_t i =0; i< input_num; ++i)
        {
            TensorShape output_shape(output_dims[i]);
            output_shape.set_layout(inputs[i]->shape().layout());
            auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("meshgrid/" + std::to_string(i)), output_shape);
            if ((quants.size() > i) && !quants[i].empty())
            {
                add_tensor_info(inputs[i], output, quants[i]);
            }
            else if ((quants.size() > 0) && !quants[0].empty())
            {
                add_tensor_info(inputs[i], output, quants[0]);
            }
            else
            {
                add_tensor_info(inputs[i], output);
            }
            outputs.push_back(output);
        }

        auto node = Node::create(g->get_valid_node_name("meshgrid"), OpType_Meshgrid);
        for(uint32_t i=0;i<input_num;++i)
        {
            node->add_input(inputs[i]);
        }

        for(uint32_t i=0;i<input_num;++i)
        {
            node->add_output(outputs[i]);
        }
        g->add_node(node);
        auto &params = node->params();
        params["sparse"] = NodeParamValue(sparse);
        params["indexing"] = indexing_loc;    
        params["copy"] = NodeParamValue(copy_loc);
        quantize(node);
        return outputs;
    }
}; //namespace ops
}; // namespace aipubt