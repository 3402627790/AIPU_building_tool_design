//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020-2026 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Barney Zhao
 * @Date: 2023-06-19 10:05:00
 * @Last Modified by: Barney Zhao
 * @Last Modified time: 2023-06-19 10:05:00
 */

#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/internal_optype.h"
#include "core/ops/ops.h"
#include <cmath>
#include <vector>

namespace aipubt
{
void decompose_groupnorm(GraphPtr graph)
{
    auto criterion = [](NodePtr node) -> bool
    {
        auto arch = aipubt::get_target()->isainfo().type;
        if (arch > aipubt::ISA_ZYv5)
        {
            return false;
        }
        
        if (node->type() != OpType_GroupNorm)
        {
            return false;
        }
        if ((node->inputs()[0]->type() == Tensor::FP16) || (node->inputs()[0]->type() == Tensor::BFP16))
        {
            return false;
        }
        int in_dim = node->inputs()[0]->shape().dim();
        int out_dim = node->outputs()[0]->shape().dim();
        if (in_dim < 2 || in_dim > 5)
        {
            return false;
        }
        if(in_dim != out_dim)
        {
            return false;
        }
        int axis = (node->params()["axis"] + in_dim)%in_dim;
        int axis_size = node->inputs()[0]->shape()[axis];
        int group = node->params()["group"];
        if (axis == 0)
        {
            return false;
        }
        if (axis_size % group != 0)
        {
            return false;
        }

        return true;
    };

    auto nodes = graph->nodes();
    for (auto node : nodes)
    {
        if (!criterion(node))
        {
            continue;
        }

        auto input = node->inputs()[0];
        auto input_original = node->inputs()[0];
        auto input_shape = input->shape();
        auto input_shape_original = input->shape();
        int dims = input_shape.dim();
        int group = node->params()["group"];
        int axis = (node->params()["axis"] + dims)%dims;
        auto params = node->params();
        auto consts = node->constants();

        if(dims != 4)
        {
            std::vector<int> tmp_shape = input_shape_original;
            std::vector<int> new_shape;
            if(dims > 4)
            {
                int new_w = 1;
                new_shape.push_back(tmp_shape[0]);
                if(axis == 1)
                {
                    new_shape.push_back(tmp_shape[1]);
                    for(int i = 2; i < dims-1; i++)
                    {
                        new_w *= tmp_shape[i];
                    }
                    new_shape.push_back(new_w);
                    new_shape.push_back(tmp_shape[dims-1]);
                }
                else if(axis == dims -1)
                {
                    for(int i = 1; i < dims-2; i++)
                    {
                        new_w *= tmp_shape[i];
                    }
                    new_shape.push_back(new_w);
                    new_shape.push_back(tmp_shape[dims-2]);
                    new_shape.push_back(tmp_shape[dims-1]);
                    axis = 3;           
                }
                else
                {
                    for(int i = 1; i < axis; i++)
                    {
                        new_w *= tmp_shape[i];
                    }
                    new_shape.push_back(new_w);
                    new_shape.push_back(tmp_shape[axis]);
                    new_w = 1;
                    for(int i = axis+1; i < dims; i++)
                    {
                        new_w *= tmp_shape[i];
                    }
                    new_shape.push_back(new_w);
                    axis = 2;
                }
                
            }
            else
            {
                for(int i = 0; i < dims; i++)
                {
                    new_shape.push_back(tmp_shape[i]);
                }
                for(int i = dims; i < 4; i++)
                {
                    new_shape.push_back(1);
                }
            }
            input_shape.reshape(new_shape);
            auto input_reshape_node = ops::reshape(node->inputs()[0], input_shape);
            auto reshape_node = input_reshape_node->node();
            graph->add_node(reshape_node);
            ADD_LOCATION_BY_NODE(reshape_node, node);
            input = input_reshape_node;
        }

        auto mvn_node = Node::create(node->name() + "/mvn", OpType_MVN);
        if(group == 1)
        {
            mvn_node->add_input(input);
        }
        else
        {
            std::vector<uint32_t> perm;
            int32_t tmp_h = input_shape[2];
            int32_t tmp_w = input_shape[3];
            if(axis == 2)
            {
                perm = {0, 2, 1, 3};
                tmp_h = input_shape[1];
                tmp_w = input_shape[3];
            }
            if(axis == 3)
            {
                perm = {0, 3, 1, 2};
                tmp_h = input_shape[1];
                tmp_w = input_shape[2];
            }
            if(axis != 1)
            {
                auto transpose_0_output = ops::transpose(input, perm);
                auto transpose_0_node = transpose_0_output->node();
                graph->add_node(transpose_0_node);
                ADD_LOCATION_BY_NODE(transpose_0_node, node);
                input = transpose_0_output;
            }
            
            TensorShape reshape_0_shape{input_shape[0] * group, input_shape[axis]/group, tmp_h,tmp_w};
            auto reshape_0_output = ops::reshape(input, reshape_0_shape);
            auto reshape_0_node = reshape_0_output->node();
            graph->add_node(reshape_0_node);
            ADD_LOCATION_BY_NODE(reshape_0_node, node);
            input = reshape_0_output;
            mvn_node->add_input(input);
        }
  
        // mvn
        std::vector<int32_t> mvn_axis = {1, 2, 3};
        mvn_node->params()["axis"] = mvn_axis;
        if ((node->inputs()[0]->type() == Tensor::INT8) || (node->inputs()[0]->type() == Tensor::UINT8))
        {
            mvn_node->params()["shift_value"] = params["norm_shift_value"];
            mvn_node->params()["shift_type"] = params["norm_shift_type"];
            mvn_node->params()["var_shift_value"] = params["var_shift_value"];
            mvn_node->params()["var_shift_type"] = params["var_shift_type"];
            mvn_node->params()["scale_value"] = params["norm_scale_value"];
            mvn_node->params()["scale_type"] = params["norm_scale_type"];
            mvn_node->constants()["lut"] = consts["lut"];
        }
        else if ((node->inputs()[0]->type() == Tensor::INT16) || (node->inputs()[0]->type() == Tensor::UINT16))
        {
            mvn_node->params()["shift_value"] = params["norm_shift_value"];
            mvn_node->params()["shift_type"] = params["norm_shift_type"];
            mvn_node->params()["scale_value"] = params["norm_scale_value"];
            mvn_node->params()["scale_type"] = params["norm_scale_type"];
            mvn_node->constants()["lut"] = consts["lut"];
        }
        TensorPtr mvn_output;
        
        if ((node->inputs()[0]->type() == Tensor::INT8) || (node->inputs()[0]->type() == Tensor::UINT8))
        {
            mvn_output = std::make_shared<Tensor>(mvn_node->name(), mvn_node->inputs()[0]->shape(), Tensor::INT8);
            auto mvn_out_mem_size = mvn_output->mem_size();
            uint8_t *mvn_data = new uint8_t[mvn_out_mem_size];
            mvn_output->set_data(mvn_data, mvn_out_mem_size);
        }
        else if ((node->inputs()[0]->type() == Tensor::INT16) || (node->inputs()[0]->type() == Tensor::UINT16))
        {
            mvn_output = std::make_shared<Tensor>(mvn_node->name(), mvn_node->inputs()[0]->shape(), Tensor::INT16);
            auto mvn_out_mem_size = mvn_output->mem_size();
            uint16_t *mvn_data = new uint16_t[mvn_out_mem_size];
            mvn_output->set_data(mvn_data, mvn_out_mem_size);
        }
        mvn_node->add_output(mvn_output);
        graph->add_node(mvn_node);
        ADD_LOCATION_BY_NODE(mvn_node, node);
        input = mvn_output;

        auto bn_node = Node::create(node->name() + "/bn", OpType_BatchNorm);
        if (group==1)
        {
            bn_node->add_input(input);
        }
        else
        {
            std::vector<uint32_t> perm;
            int32_t tmp_h = input_shape[2];
            int32_t tmp_w = input_shape[3];
            if(axis == 2)
            {
                perm = {0, 2, 1, 3};
                tmp_h = input_shape[1];
                tmp_w = input_shape[3];
            }
            if(axis == 3)
            {
                perm = {0, 2,3,1};
                tmp_h = input_shape[1];
                tmp_w = input_shape[2];
            }
            TensorShape reshape_3_shape{input_shape[0], input_shape[axis], tmp_h,tmp_w};
            auto reshape_3_output = ops::reshape(input, reshape_3_shape);
            auto reshape_3_node = reshape_3_output->node();
            graph->add_node(reshape_3_node);
            ADD_LOCATION_BY_NODE(reshape_3_node, node);
            input = reshape_3_output;
            
            if(axis != 1)
            {
                auto transpose_1_output = ops::transpose(input, perm);
                auto transpose_1_node = transpose_1_output->node();
                graph->add_node(transpose_1_node);
                ADD_LOCATION_BY_NODE(transpose_1_node, node);
                input = transpose_1_output;
            }
            bn_node->add_input(input);
        }
        
        bn_node->params()["axis"] = axis;
        if (Tensor::dtype_size(consts["weights"]->type()) == 1 && Tensor::dtype_size(bn_node->inputs()[0]->type()) == 2)
        {
            TensorShape bn_weights_shape = consts["weights"]->shape();
            TensorPtr bn_weights16_ptr = std::make_shared<Tensor>("bn_weights16", bn_weights_shape, Tensor::INT16);
            int16_t *bn_weights16_data = new int16_t[bn_weights_shape.size()];
            auto p = consts["weights"]->data<int8_t>();
            for (size_t i = 0; i < bn_weights_shape.size(); i++)
            {
                bn_weights16_data[i] = int16_t(*(p + i));
            }
            bn_weights16_ptr->set_data(bn_weights16_data, bn_weights_shape.size() * sizeof(int16_t));
            bn_node->constants()["weights"] = bn_weights16_ptr;
        }
        else
        {
            bn_node->constants()["weights"] = consts["weights"];
        }
        if (Tensor::dtype_size(bn_node->constants()["weights"]->type()) == 1)
        {
            bn_node->params()["use_8bit_weight"] = 1;
        }
        bn_node->constants()["biases"] = consts["biases"];
        if ((params.count("scale_value")) && (params.count("shift_value")))
        {
            bn_node->params()["scale_value"] = params["scale_value"];
            bn_node->params()["shift_value"] = params["shift_value"];
            bn_node->params()["scale_type"] = params["scale_type"];
            bn_node->params()["shift_type"] = params["shift_type"];
        }
        else if (consts.count("scale") && consts.count("shift"))
        {
            bn_node->constants()["scale"] = consts["scale"];
            bn_node->constants()["shift"] = consts["shift"];
        }

        if(dims!=4)
        {
            auto origin_output_tensor = node->outputs()[0];
            auto bn_output= std::make_shared<Tensor>(origin_output_tensor->name() + "/input", input_shape, origin_output_tensor->type());
            bn_node->add_output(bn_output);
            graph->add_node(bn_node);
            ADD_LOCATION_BY_NODE(bn_node, node);
            input = bn_output;
            auto reshape_4_output = ops::reshape(input, input_shape_original);
            auto reshape_4_node = reshape_4_output->node();
            reshape_4_node->remove_output(0);
            reshape_4_node->add_output(origin_output_tensor);
            graph->add_node(reshape_4_node);
            ADD_LOCATION_BY_NODE(reshape_4_node, node);
        }
        else
        {
            auto origin_output_tensor = node->outputs()[0];
            bn_node->remove_output(0);
            bn_node->add_output(origin_output_tensor);
            graph->add_node(bn_node);
            ADD_LOCATION_BY_NODE(bn_node, node);
        }
        
        graph->remove_node(node);
    }
}
} // namespace aipubt
