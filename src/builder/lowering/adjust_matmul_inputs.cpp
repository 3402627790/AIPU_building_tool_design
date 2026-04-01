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
 *       Filename:  adjust_matmul_inputs.cpp
 *
 *    Description:  adjust input node for matmul
 * For C = A x B:
 * - convert matrix A to conv inputs format
 * - convert matrix B to conv weights format
 * - transpose A/B according to trans_a/trans_b params
 * - set the trans_a and trans_b params
 *
 * The LayerLib implementation in Zhouyi v1 accepts C = A x transposed(B)
 * Input matrix A:
 * - NHWC
 * Input matrix B:
 * - KHWC (C is the input channel number and K is the output channel number)
 *
 * The LayerLib implementation in Zhouyi v2 accepts C = A x B
 * Input matrix A:
 * - NC'HWC32 (8-bit)
 * - NC'HWC16 (16-bit / 8-bit asymmetric quantization / uint8 x uint8)
 * Input matrix B:
 * - NCi'Co'Ci32Co32 (8-bit)
 * - NCi'Co'Ci16Co16 (16-bit / 8-bit asymmetric quantization)
 *
 * The LayerLib implementation in Zhouyi v3 accepts C = A x B or C = A x transposed(B)
 * Input matrix A:
 * - NC'HWC32T8 (8-bit)
 * - NC'HWC16T8 (16-bit / 8-bit asymmetric quantization / uint8 x uint8)
 * Input matrix B:
 * - NC'HWC32T8 (8-bit)
 * - NC'HWC16T8 (16-bit / 8-bit asymmetric quantization / uint8 x uint8)
 *
 *        Version:  1.0
 *        Created:  10/19/20 14:31:36
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
#include "core/config.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include <algorithm>
#include <functional>

namespace aipubt
{
void adjust_matmul_inputs(GraphPtr graph)
{
    auto criterion = [](NodePtr node) -> bool {
        return (node->type() == OpType_MatMul) && (node->inputs().size() == 2);
    };

    auto gen_transpose_node = [](TensorPtr input_tensor, NodePtr origin_node) {
        auto node_name = input_tensor->name() + "_transpose";
        // NOTE: OpType_Transpose4D is deprecated
        auto transpose_node = Node::create(node_name, OpType_Transpose);
        transpose_node->add_input(input_tensor);

        TensorShape output_shape(input_tensor->shape());
        std::swap(output_shape.w(), output_shape.c());
        auto output_type = input_tensor->type();
        output_shape.set_layout(DataLayout_NHWC);
        auto output_tensor = std::make_shared<Tensor>(node_name + "/output", output_shape, output_type);
        output_tensor->quantization() = input_tensor->quantization();
        transpose_node->add_output(output_tensor);

        transpose_node->params()["perm"].push_back(0);
        transpose_node->params()["perm"].push_back(1);
        transpose_node->params()["perm"].push_back(3);
        transpose_node->params()["perm"].push_back(2);
        return transpose_node;
    };

    auto need_arrange_input = [](TensorShape &shape, int alignment, bool is_transposed = false) {
        if (is_transposed)
        {
            return shape.c_total() > 1;
        }
        else
        {
            return shape.w() > 1 && shape.c_total() != alignment;
        }
    };

    auto reshape_input = [](TensorPtr input_tensor, int alignment, bool is_transposed = false, bool is_input_asym_quant = false) {
        auto input_shape = input_tensor->shape();
        if (is_transposed)
        {
            // shape.c_total() == 1
            if (input_shape[4] == 1)
            {
                input_tensor->shape() = TensorShape(
                    input_shape[0],
                    input_shape[2],
                    1,
                    1,
                    alignment);
            }
            // shape.w() == alignment
            else if (input_shape[2] == 1)
            {
                input_tensor->shape() = TensorShape(
                    input_shape[0],
                    1,
                    1,
                    input_shape[4],
                    alignment);
            }
            else
            {
                aipu_assert(false);
            }
        }
        else
        {
            // shape.w() == 1
            if (input_shape[2] == 1)
            {
                input_tensor->shape() = TensorShape(
                    input_shape[0],
                    input_shape[3],
                    1,
                    1,
                    alignment);
            }
            // shape.c_total() == alignment
            else if (input_shape[3] == 1)
            {
                input_tensor->shape() = TensorShape(
                    input_shape[0],
                    1,
                    1,
                    input_shape[2],
                    alignment);
            }
            else
            {
                aipu_assert(false);
            }
        }
        input_tensor->shape().set_layout(DataLayout_Flat);
    };

    auto gen_input_reshape_node = [](TensorPtr input_tensor, NodePtr origin_node, int alignment, bool is_transposed = false, bool is_input_asym_quant = false) {
        auto node_name = input_tensor->name() + "_reshape";
        auto reshape_node = Node::create(node_name, OpType_Reshape);
        reshape_node->add_input(input_tensor);

        int batch_num = input_tensor->shape().n() * input_tensor->shape().h();
        // Input shape:
        // - NHWC==NHWC'C32 (8-bit matrix)
        // - NHCW==NHC'C32W (8-bit transposed matrix)
        // - NHWC==NHWC'C16 (16-bit matrix)
        // - NHCW==NHC'C16W (16-bit transposed matrix)
        auto input_shape = input_tensor->shape();
        TensorShape output_shape;
        if (is_transposed)
        {
            // Channel alignment must be satisfied after matmul_c_adaptor
            aipu_assert(input_shape.w() % alignment == 0);
            output_shape = TensorShape(
                batch_num,
                1,
                input_shape.w() / alignment,
                alignment,
                input_shape.c());
        }
        else
        {
            // Channel alignment must be satisfied after matmul_c_adaptor
            aipu_assert(input_shape.c() % alignment == 0);
            output_shape = TensorShape(
                batch_num,
                1,
                input_shape.w(),
                input_shape.c() / alignment,
                alignment);
        }
        auto output_type = input_tensor->type();
        output_shape.set_layout(DataLayout_Flat);
        auto output_tensor = std::make_shared<Tensor>(node_name + "/output", output_shape, output_type);
        output_tensor->quantization() = input_tensor->quantization();
        reshape_node->add_output(output_tensor);
        reshape_node->params()["shape"] = reshape_node->outputs()[0]->shape();

        return reshape_node;
    };

    auto gen_input_arrange_node = [](TensorPtr input_tensor, NodePtr origin_node, int alignment, bool is_transposed = false, bool is_input_asym_quant = false) {
        int batch_num = input_tensor->shape()[0];
        // width
        int32_t w = input_tensor->shape()[2];
        // input channel / alignment
        int32_t c_aligned = input_tensor->shape()[3];
        if (is_transposed)
        {
            w = input_tensor->shape()[4];
            c_aligned = input_tensor->shape()[2];
        }

        auto node_name = input_tensor->name() + "_input_arrange";
        // NOTE: OpType_Transpose4D is deprecated
        auto arrange_node = Node::create(node_name, OpType_Transpose);
        arrange_node->add_input(input_tensor);

        // Output shape:
        // - NC'HWC32 (8-bit matrix, 8-bit transposed matrix)
        // - NC'HWC16 (16-bit matrix, 16-bit transposed matrix)
        TensorShape output_shape(
            batch_num,
            c_aligned,
            1,
            w,
            alignment);
        auto output_type = input_tensor->type();
        output_shape.set_layout(DataLayout_Flat);
        auto output_tensor = std::make_shared<Tensor>(node_name + "/output", output_shape, output_type);
        output_tensor->quantization() = input_tensor->quantization();
        arrange_node->add_output(output_tensor);

        if (is_transposed)
        {
            arrange_node->params()["perm"].push_back(0);
            arrange_node->params()["perm"].push_back(2);
            arrange_node->params()["perm"].push_back(1);
            arrange_node->params()["perm"].push_back(4);
            arrange_node->params()["perm"].push_back(3);
        }
        else
        {
            arrange_node->params()["perm"].push_back(0);
            arrange_node->params()["perm"].push_back(3);
            arrange_node->params()["perm"].push_back(1);
            arrange_node->params()["perm"].push_back(2);
            arrange_node->params()["perm"].push_back(4);
        }

        return arrange_node;
    };

    auto gen_weight_reshape_node = [](TensorPtr input_tensor, NodePtr origin_node, int alignment, bool is_input_asym_quant) {
        // Channel alignment must be satisfied after matmul_c_adaptor
        aipu_assert(input_tensor->shape().w() % alignment == 0);
        aipu_assert(input_tensor->shape().c() % alignment == 0);

        auto node_name = input_tensor->name() + "_reshape";
        auto reshape_node = Node::create(node_name, OpType_Reshape);
        reshape_node->add_input(input_tensor);

        int batch_num = input_tensor->shape().n() * input_tensor->shape().h();
        // Weight shape:
        // - NCiCo==NCi'Ci32Co'Co32 (8-bit matrix)
        // - NCoCi==NCi'Ci32Co'Co32 (8-bit transposed matrix)
        // - NCiCo==NCi'Ci16Co'Co16 (16-bit matrix)
        // - NCoCi==NCi'Ci16Co'Co16 (16-bit transposed matrix)
        TensorShape output_shape(
            batch_num,
            input_tensor->shape().w() / alignment,
            alignment,
            input_tensor->shape().c() / alignment,
            alignment);
        auto output_type = input_tensor->type();
        output_shape.set_layout(DataLayout_Flat);
        auto output_tensor = std::make_shared<Tensor>(node_name + "/output", output_shape, output_type);
        output_tensor->quantization() = input_tensor->quantization();
        reshape_node->add_output(output_tensor);
        reshape_node->params()["shape"] = reshape_node->outputs()[0]->shape();

        return reshape_node;
    };

    auto gen_weight_arrange_node = [](TensorPtr input_tensor, NodePtr origin_node, int alignment, bool is_transposed, bool is_input_asym_quant) {
        int batch_num = input_tensor->shape()[0];
        // input channel / alignment
        int32_t in_c_aligned = input_tensor->shape()[1];
        // output channel / alignment
        int32_t out_c_aligned = input_tensor->shape()[3];
        if (is_transposed)
        {
            std::swap(in_c_aligned, out_c_aligned);
        }

        auto node_name = input_tensor->name() + "_weight_arrange";
        // NOTE: OpType_Transpose4D is deprecated
        auto arrange_node = Node::create(node_name, OpType_Transpose);
        arrange_node->add_input(input_tensor);

        // Output shape:
        // - NCi'Co'Ci32Co32 (8-bit matrix, 8-bit transposed matrix)
        // - NCi'Co'Ci16Co16 (16-bit matrix, 16-bit transposed matrix)
        TensorShape output_shape(
            batch_num,
            in_c_aligned,
            out_c_aligned,
            alignment,
            alignment);
        auto output_type = input_tensor->type();
        output_shape.set_layout(DataLayout_Flat);
        auto output_tensor = std::make_shared<Tensor>(node_name + "/output", output_shape, output_type);
        output_tensor->quantization() = input_tensor->quantization();
        arrange_node->add_output(output_tensor);

        if (is_transposed)
        {
            arrange_node->params()["perm"].push_back(0);
            arrange_node->params()["perm"].push_back(3);
            arrange_node->params()["perm"].push_back(1);
            arrange_node->params()["perm"].push_back(4);
            arrange_node->params()["perm"].push_back(2);
        }
        else
        {
            arrange_node->params()["perm"].push_back(0);
            arrange_node->params()["perm"].push_back(1);
            arrange_node->params()["perm"].push_back(3);
            arrange_node->params()["perm"].push_back(2);
            arrange_node->params()["perm"].push_back(4);
        }

        return arrange_node;
    };

    /*
    auto find_parent = [](const std::vector<NodePtr> &parents, TensorPtr input) {
        auto it = std::find_if(parents.begin(), parents.end(), [input](NodePtr parent) {
            auto outputs = parent->outputs();
            return std::find(outputs.begin(), outputs.end(), input) != outputs.end();
        });
        return *it;
    };
    */

    auto insert_transpose_node = [gen_transpose_node](GraphPtr graph, NodePtr node, size_t input_idx) {
        auto transpose_node = gen_transpose_node(node->inputs()[input_idx], node);
        graph->add_node(transpose_node);
        ADD_LOCATION_BY_NODE_WITH_NAME(adjust_matmul_inputs, transpose_node, node);
        node->remove_input(input_idx);
        node->add_input(transpose_node->outputs()[0], input_idx);
    };

    auto adjust_v1_inputs = [insert_transpose_node](GraphPtr graph, NodePtr node) {
        // Adjust input matrix 0 if it's transposed
        if (node->params()["trans_a"])
        {
            insert_transpose_node(graph, node, 0);
        }

        // Adjust input matrix 1 if it's NOT transposed
        if (!node->params()["trans_b"])
        {
            insert_transpose_node(graph, node, 1);
        }

        node->params()["trans_a"] = false;
        node->params()["trans_b"] = true;
    };

    auto adjust_v2_inputs = [gen_input_reshape_node, need_arrange_input, gen_input_arrange_node, reshape_input,
                             gen_weight_reshape_node, gen_weight_arrange_node](GraphPtr graph, NodePtr node) {
        auto alignment = get_input_channel_base(node);
        bool is_input_asym_quant = false;
        for (auto input : node->inputs())
        {
            if (input->quantization().offset() != 0)
            {
                is_input_asym_quant = true;
            }
        }
        if (node->inputs()[0]->type() == Tensor::UINT8 && node->inputs()[1]->type() == Tensor::UINT8)
        {
            is_input_asym_quant = true;
        }
        if (alignment == 32 && is_input_asym_quant)
        {
            alignment = 16;
        }

        // Adjust input matrix 0 if it's transposed
        // Convert the input matrix 0 to NC'HWC32 / NC'HWC16
        if (node->inputs()[0]->shape().dim() == 4)
        {
            auto is_transposed_x = node->params()["trans_a"];
            auto reshape_node = gen_input_reshape_node(node->inputs()[0], node, alignment, is_transposed_x, is_input_asym_quant);

            graph->add_node(reshape_node);
            ADD_LOCATION_BY_NODE_WITH_NAME(adjust_matmul_inputs, reshape_node, node);

            auto input_tensor_shape = node->inputs()[0]->shape();
            if (need_arrange_input(input_tensor_shape, alignment, is_transposed_x))
            {
                auto arrange_node = gen_input_arrange_node(reshape_node->outputs()[0], node, alignment, is_transposed_x, is_input_asym_quant);
                graph->add_node(arrange_node);
                ADD_LOCATION_BY_NODE_WITH_NAME(adjust_matmul_inputs, arrange_node, node);
                node->remove_input(0);
                node->add_input(arrange_node->outputs()[0], 0);
            }
            else
            {
                // Don't need to add arrange node if the input shape is:
                // - N x H x 1 x C for matrix
                // - N x H x W x 32 for matrix
                // - N x H x W x 1 for transposed matrix
                // - N x H x 32 x C for transposed matrix
                node->remove_input(0);
                node->add_input(reshape_node->outputs()[0], 0);
                reshape_input(node->inputs()[0], alignment, is_transposed_x, is_input_asym_quant);
                reshape_node->params()["shape"] = reshape_node->outputs()[0]->shape();
            }
        }

        // Adjust input matrix 1 if it's transposed
        // Convert the input matrix 1 to NCi'Co'Ci32Co32 / NCi'Co'Ci16Co16
        // which will be treated as convolution weights in the implementation of LayerLib
        // if (node->inputs()[1]->shape().dim() == 4)
        {
            auto reshape_node = gen_weight_reshape_node(node->inputs()[1], node, alignment, is_input_asym_quant);
            auto arrange_node = gen_weight_arrange_node(reshape_node->outputs()[0], node, alignment, node->params()["trans_b"], is_input_asym_quant);
            graph->add_node(reshape_node);
            graph->add_node(arrange_node);
            ADD_LOCATION_BY_NODE_WITH_NAME(adjust_matmul_inputs, reshape_node, node);
            ADD_LOCATION_BY_NODE_WITH_NAME(adjust_matmul_inputs, arrange_node, node);
            node->remove_input(1);
            node->add_input(arrange_node->outputs()[0], 1);
        }

        node->params()["trans_a"] = false;
        node->params()["trans_b"] = false;
    };

    auto need_reshape_batch_num = [](TensorPtr input_tensor) {
        return (input_tensor->shape().h() > 1);
    };

    auto gen_batch_num_reshape_node = [](TensorPtr input_tensor, NodePtr origin_node) {
        auto node_name = input_tensor->name() + "_reshape";
        auto reshape_node = Node::create(node_name, OpType_Reshape);
        reshape_node->add_input(input_tensor);

        auto input_shape = input_tensor->shape();
        auto output_shape = input_shape;
        output_shape.n() = input_shape.n() * input_shape.h();
        output_shape.h() = 1;
        auto output_type = input_tensor->type();
        auto output_tensor = std::make_shared<Tensor>(node_name + "/output", output_shape, output_type);
        output_tensor->quantization() = input_tensor->quantization();
        reshape_node->add_output(output_tensor);
        reshape_node->params()["shape"] = reshape_node->outputs()[0]->shape();

        if (BuilderConfig::get<bool>(Constants::enable_ds))
        {
            std::vector<Expr> ds_shape = {
                input_shape.symbol().n() * input_shape.symbol().h(),
                1,
                input_shape.symbol().w(),
                input_shape.symbol().c(),
            };
            reshape_node->params()["ds_shape"] = ds_shape;
        }
        reshape_node->infer_shape(true);
        return reshape_node;
    };

    auto adjust_v3_inputs = [need_reshape_batch_num, gen_batch_num_reshape_node, insert_transpose_node](GraphPtr graph, NodePtr node) {
        for (auto i = 0; i < 2; i++)
        {
            auto input_tensor = node->inputs()[i];
            if (need_reshape_batch_num(input_tensor))
            {
                auto consumers = input_tensor->consumers();
                auto reshape_node = gen_batch_num_reshape_node(input_tensor, node);
                graph->add_node(reshape_node);
                ADD_LOCATION_BY_NODE_WITH_NAME(adjust_matmul_inputs, reshape_node, node);
                for (auto n : consumers)
                {
                    if (n->type() == OpType_MatMul)
                    {
                        n->add_input(reshape_node->outputs()[0], n->remove_input(input_tensor));
                    }
                }
            }
        }

        // Adjust input matrix 0 if it's transposed
        auto is_transposed_x = node->params()["trans_a"];
        if (is_transposed_x)
        {
            insert_transpose_node(graph, node, 0);
        }

        // NOTE: don't change layouts of inputs
        // datalayout_schedule will insert necessary LayoutConvertor
        // for the accepted layouts of Matmul
        // Refer to: https://jira.armchina.com/browse/CAL-1088

        node->params()["trans_a"] = false;
    };

    auto nodes = graph->nodes();
    for (auto node : nodes)
    {
        if (!criterion(node))
        {
            continue;
        }

        if (is_matmul_to_fc_enabled() && need_matmul_to_fc(node))
        {
            WARN() << "Use matmul_to_fc instead of adjust_matmul_inputs pass.";
            continue;
        }

        // Only use the pass for HWA plugin
        if (!check_hwa_matmul(node))
        {
            WARN() << "Matmul not supported by HWA, skip adjust_matmul_inputs pass and use TPC plugin";
            continue;
        }

        auto arch = aipubt::get_target()->isainfo().type;
        if (arch == aipubt::ISA_ZY)
        {
            adjust_v1_inputs(graph, node);
        }
        else if (arch == aipubt::ISA_ZYv2)
        {
            adjust_v2_inputs(graph, node);
        }
        else if (arch == aipubt::ISA_ZYv3)
        {
            // NOTE: work around the hardware issue that Z3 AIFF batch-matmul only supports 8bit
            // Refer to: https://jira.armchina.com/browse/CB-1657
            bool is_8bit = true;
            bool is_input_asym_quant = false;
            for (auto input : node->inputs())
            {
                is_8bit &= (Tensor::dtype_bits(input->type()) == 8);
                is_input_asym_quant |= (input->quantization().offset() != 0);
            }
            if (node->inputs()[0]->type() == Tensor::UINT8 && node->inputs()[1]->type() == Tensor::UINT8)
            {
                is_input_asym_quant = true;
            }
            if (is_input_asym_quant)
            {
                is_8bit = false;
            }
            if (is_8bit)
            {
                adjust_v3_inputs(graph, node);
            }
            else
            {
                adjust_v2_inputs(graph, node);
            }
        }
        else if (arch >= aipubt::ISA_ZYv5)
        {
            adjust_v3_inputs(graph, node);
        }

        auto output = node->outputs()[0];
        auto consumers = output->consumers();
        if (arch >= aipubt::ISA_ZYv2 && need_reshape_batch_num(output))
        {
            std::vector<int> indexs;
            for (auto c : consumers)
            {
                indexs.push_back(c->remove_input(output));
            }
            auto old_output_shape = output->shape();
            auto new_output = std::make_shared<Tensor>(*output);
            new_output->shape().n() = old_output_shape.n() * old_output_shape.h();
            new_output->shape().h() = 1;
            new_output->set_name(graph->get_valid_tensor_name(output->name()));
            new_output->set_io_tensor(false);

            node->clear_outputs();
            node->add_output(new_output);
            node->infer_shape(true);

            auto reshape_output = ops::reshape(new_output, old_output_shape);
            auto reshape_node = reshape_output->node();
            reshape_node->clear_outputs();
            reshape_node->add_output(output);
            if (BuilderConfig::get<bool>(Constants::enable_ds))
            {
                std::vector<Expr> ds_shape = old_output_shape.symbol();
                reshape_node->params()["ds_shape"] = ds_shape;
            }
            reshape_node->infer_shape(true);

            graph->add_node(reshape_node);
            ADD_LOCATION_BY_NODE(reshape_node, node);
            for (uint32_t i = 0; i < consumers.size(); ++i)
            {
                consumers[i]->add_input(output, indexs[i]);
            }
        }
    }
}
} // namespace aipubt
