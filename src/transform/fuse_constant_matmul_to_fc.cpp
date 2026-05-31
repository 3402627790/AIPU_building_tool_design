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
 * =====================================================================================
 *
 *       Filename:  fuse_constant_matmul_to_fc.cpp
 *
 *    Description:  fuse constant + single batch matmul to equivalent fc
 *
 *        Version:  1.0
 *        Created:  09/01/23 15:44:25
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Shuai Yuan
 *   Organization:  Arm China
 *
 * =====================================================================================
 */
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include <array>

namespace aipubt
{

static void set_fc_bias(ComputeTensor &fc_bias, ComputeTensor &fc_weight, int input_zp)
{
    if (input_zp == 0)
    {
        return;
    }

    fc_bias = fc_weight.sum({1}) * input_zp;
}

bool fuse_constant_matmul_to_fc(GraphPtr graph)
{
    static GraphMatcherPattern pattern = {
        {{"0", OpType_Constant}, {"1", OpType_MatMul}},
        {{"0", "1"}},
    };
    static GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);

    auto remove_node = [graph](NodePtr node)
    {
        for (auto output : node->outputs())
        {
            if (output->consumers().size() > 1)
            {
                return;
            }
        }
        graph->remove_node(node);
    };

    auto matched = matcher.match(graph);
    bool is_changed = false;
    while (matched.size())
    {
        bool is_fused = false;
        for (auto &&m : matched)
        {
            auto constant_node = m[0];
            auto matmul_node = m[1];

            if (matmul_node->inputs()[1] != constant_node->outputs()[0])
            {
                continue;
            }

            bool is_valid_matmul_dim = true;
            for (auto input : matmul_node->inputs())
            {
                auto shape = input->shape();
                if ((shape.dim() > 4) || (shape.dim() < 2))
                {
                    is_valid_matmul_dim = false;
                    break;
                }
            }
            auto weight_ptr = matmul_node->inputs()[1];
            auto weight_shape = weight_ptr->shape();
            auto weight_batch = 1;
            for (auto i = 0; i < weight_shape.dim() - 2; i++)
            {
                weight_batch *= weight_shape[i];
            }
            bool is_weight_not_quant = Tensor::is_float(weight_ptr->type());
            bool is_weight_asym_quant = !is_weight_not_quant && (weight_ptr->quantization().offset() != 0);
            bool is_weight_single_batch = (weight_batch == 1);
            if (!is_valid_matmul_dim || is_weight_asym_quant || !is_weight_single_batch)
            {
                continue;
            }

            DEBUG() << "Fuse constant: " << constant_node->name()
                    << " matmul: " << matmul_node->name();

            auto origin_input = matmul_node->inputs()[0];
            auto input_zp = origin_input->quantization().offset();
            auto matmul_input = origin_input;
            bool trans_a = matmul_node->params()["trans_a"];
            if (trans_a)
            {
                std::vector<uint32_t> perm;
                for (auto i = 0; i < origin_input->shape().dim(); i++)
                {
                    perm.push_back(i);
                }
                std::swap(perm[perm.size() - 2], perm[perm.size() - 1]);
                auto in_transpose_output = ops::transpose(origin_input, perm);
                matmul_input = in_transpose_output;
                auto in_transpose_node = in_transpose_output->node();
                graph->add_node(in_transpose_node);
                ADD_LOCATION_BY_NODE(in_transpose_node, matmul_node);
            }

            auto matmul_input_shape = matmul_input->shape();
            int fc_batch = 1;
            for (auto i = 0; i < matmul_input_shape.dim() - 1; i++)
            {
                fc_batch *= matmul_input_shape[i];
            }
            auto matmul_output = matmul_node->outputs()[0];
            auto matmul_output_shape = matmul_output->shape();
            TensorShape fc_input_shape(fc_batch, matmul_input_shape[matmul_input_shape.dim() - 1]);
            std::vector<tongue::Expr> fc_input_exprs;
            if (matmul_node->is_dynamic())
            {
                tongue::Expr fc_batch_expr = 1;
                for (auto i = 0; i < matmul_input_shape.symbol().dim() - 1; i++)
                {
                    fc_batch_expr *= matmul_input_shape.symbol()[i];
                }
                fc_input_exprs = {fc_batch_expr, matmul_input_shape.symbol()[matmul_input_shape.symbol().dim() - 1]};
            }
            auto fc_in_reshape = ops::reshape(matmul_input, fc_input_shape, fc_input_exprs);
            auto fc_in_reshape_node = fc_in_reshape->node();
            ADD_LOCATION_BY_NODE(fc_in_reshape->node(), matmul_node);

            int fc_output_dim_1 = matmul_output_shape[matmul_output_shape.dim() - 1];
            auto weight = constant_node->constants()["weights"];
            auto weight_quant = weight->quantization();
            ComputeTensor fc_weight = weight;
            bool trans_b = matmul_node->params()["trans_b"];
            if (!trans_b)
            {
                fc_weight = fc_weight.reshape(fc_input_shape[1], fc_output_dim_1).permute({1, 0});
            }
            TensorShape fc_weight_shape(fc_output_dim_1, fc_input_shape[1]);
            fc_weight_shape.set_layout(DataLayout_Flat);
            fc_weight->shape() = fc_weight_shape;
            weight = fc_weight;
            weight->quantization() = weight_quant;

            // Still need to create fake biases for fc even if the input is sym quant.
            // TODO: deprecated once fc and conv supports empty biases
            TensorShape fc_bias_shape(fc_output_dim_1);
            auto bias_type = Tensor::INT32;
            if (is_weight_not_quant)
            {
                bias_type = Tensor::FP32;
            }
            auto fc_bias = ComputeTensor::zeros(fc_bias_shape, bias_type);
            set_fc_bias(fc_bias, fc_weight, input_zp);
            TensorPtr bias = fc_bias;
            bias->quantization() = Quantization(1.0, 0);

            auto fc_out = ops::fully_connected(fc_in_reshape, weight, bias, "NONE",
                                               matmul_output->quantization());
            ADD_LOCATION_BY_NODE(fc_out->node(), matmul_node);

            auto fc_out_reshape_node = ops::reshape(fc_out, matmul_output_shape)->node();
            ADD_LOCATION_BY_NODE(fc_out_reshape_node, matmul_node);
            fc_out_reshape_node->remove_output(0);
            fc_out_reshape_node->add_output(matmul_output, 0);

            remove_node(constant_node);
            graph->remove_node(matmul_node);

            is_changed = true;
            is_fused = true;
        }
        if (!is_fused)
        {
            break;
        }
        matched = matcher.match(graph);
    }

    return is_changed;
}
} // namespace aipubt
