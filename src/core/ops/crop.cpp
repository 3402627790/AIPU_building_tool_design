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
 * @Date: 2021-07-16 16:13:51
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-08-13 11:33:39
 */

#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    TensorPtr crop(TensorPtr input,
                   const std::vector<std::pair<int32_t, int32_t>> &crops,
                   const std::vector<std::pair<tongue::Expr, tongue::Expr>> &ds_crops)
    {
        auto g = input->node()->graph();
        std::vector<int32_t> output_dim;
        std::vector<std::pair<int32_t, int32_t>> loc_crops = crops;
        std::vector<std::pair<tongue::Expr, tongue::Expr>> loc_ds_crops = ds_crops;
        int32_t dim = input->shape().dim();
        for (int32_t i = loc_crops.size(); i < dim; ++i)
        {
            loc_crops.push_back({0, input->shape()[i]});
        }
        for (int32_t i = loc_ds_crops.size(); i < dim; ++i)
        {
            loc_ds_crops.push_back({loc_crops[i].first, loc_crops[i].second});
        }

        bool has_expr = false;
        auto crop_value_process = [&](tongue::Expr &expr, tongue::Expr expr_i, int32_t max_dim)
        {
            if (expr->is_constant() &&
                (int32_t)expr->value() < 0 &&
                !expr_i->is_constant())
            {
                expr = expr + expr_i;
            }
            if (!expr->is_constant())
            {
                std::vector<Symbol> vars = expr->variables();
                std::map<Symbol, Symbol::Value> args;
                for (auto var : vars)
                {
                    args[var] = var.value();
                }
                expr->eval(args);
                has_expr = true;
            }
            int32_t expr_val = (int32_t)expr->value();
            expr_val = std::min(std::max(expr_val, -max_dim), max_dim);
            return expr_val;
        };

        NodeParamValue params_crops;
        NodeParamValue params_ds_crops;
        for (size_t i = 0; i < loc_ds_crops.size(); ++i)
        {
            auto expr_begin = loc_ds_crops[i].first;
            auto expr_end = loc_ds_crops[i].second;
            int32_t crop_begin_val = crop_value_process(expr_begin, input->shape().symbol()[i], input->shape()[i]);
            int32_t crop_end_val = crop_value_process(expr_end, input->shape().symbol()[i], input->shape()[i]);

            params_crops.push_back({crop_begin_val, crop_end_val});
            params_ds_crops.push_back({expr_begin, expr_end});

            if (crop_end_val < 0)
            {
                crop_end_val = crop_end_val + input->shape()[i];
            }
            if (crop_begin_val < 0)
            {
                crop_begin_val = crop_begin_val + input->shape()[i];
            }
            output_dim.push_back(std::abs(crop_end_val - crop_begin_val));
        }
        TensorShape output_shape(output_dim, input->shape().layout());
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("crop"), output_shape);
        add_tensor_info(input, output);
        auto node = Node::create(g->get_valid_node_name("crop"), OpType_Crop);
        node->add_input(input);
        node->add_output(output);
        node->params()["crops"] = params_crops;
        if (has_expr)
        {
            node->params()["ds_crops"] = params_ds_crops;
        }
        node->infer_shape(true);
        g->add_node(node);
        quantize(node);
        return output;
    }
} // namespace ops
} // namespace aipubt
