//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2023 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Ryan Lai
 * @Date: 2023-11-18  13:15:32
 * @Last Modified by: Ryan Lai
 * @Last Modified time: 2023-11-30 15:34:00
 */
#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/ops/ops.h"

/***
 *
 * A special pass for codeformer
 *
 * It REVERSE some transpose of the "lower_transpose" pass
 * because in codeformer:
 * "lower transpose" results in more unnecessary LayoutConvertor and Concat(Multi core)
 * The "origin" transpose position is better
 *
 *      conv/eltwise                                  conv/eltwise
 *
 *                                                      parent node
 *                                                          |
 *      parent node                                     (help node)*
 *          |                                               |
 *       reshape                                        transpose
 * [1,a,b,c*d]->[a,b,c,d]     -this-pass->      [1,a,b,c*d]->[1,c*d,a,b]
 *          |  (*)                                       (0,3,1,2)
 *      transpose                                           |
 *  [a,b,c,d]->[c,d,a,b]                                 reshape
 *      (2,3,0,1)                               [1,c*d,a,b]->[c,d,a,b]
 *          |             <- lower_transpose-               |
 *       child node                                     child node
 *
 *
 * main idea: remove the "Layout Converter" or "Concat" between parent node <-> reshape
 *
 * The following conditions are meet to make a profit:
 *    - input tensor type is NCHWC32, like concat/eltwise
 *    - child node is NHWC, like layernorm
 *
 * (*) the help node:
 *      because speed of transpose read from ddr is slow
 *      insert a help node to move data from ddr to gm
 *      let transpose read from gm
 *
 */

namespace aipubt
{
/**
 * @brief creteria of graph match pattern result, check the fit shape
 * @param reshape_in_shape
 * @param reshape_out_shape
 * @param perm
 * @return
 */
static inline bool creteria(
    const std::vector<int> &reshape_in_shape,
    const std::vector<int> &reshape_out_shape,
    const std::vector<int> &perm)
{
    if (reshape_in_shape.size() != 4)
        return false;

    // check reshape
    int a = reshape_out_shape[0];
    int b = reshape_out_shape[1];
    int c = reshape_out_shape[2];
    int d = reshape_out_shape[3];

    if (1 != reshape_in_shape[0])
        return false;
    if (a != reshape_in_shape[1])
        return false;
    if (b != reshape_in_shape[2])
        return false;
    if (c * d != reshape_in_shape[3])
        return false;

    // check transpose
    std::vector<int> const_perm = {2, 3, 0, 1};
    if (perm != const_perm)
        return false;

    return true;
}

/**
 * @brief createria_read_from_ddr
 * @param parent_node
 * @return
 *
 * !! only work on codeformer !!
 */
static inline bool createria_read_from_ddr(NodePtr parent_node)
{
    if (OpType_Concat == parent_node->type() ||
        OpType_InternalConcatenated == parent_node->type())
    {
        // TODO: refine the hard code of X2_1204MP3:
        if (parent_node->outputs()[0]->mem_size() <= 4 * 1024 * 1024)
        {
            DEBUG() << "insert help node after node: " << parent_node->name();
            return true;
        }
    }
    // TODO: refine me, only work on codeformer first layer:
    if (OpType_Convolution == parent_node->type())
    {
        if (parent_node->outputs()[0]->mem_size() == 4 * 1024 * 1024)
        {
            DEBUG() << "insert help node after node: " << parent_node->name();
            return true;
        }
    }
    return false;
}

/**
 * @brief insert_dma_to_gm_help_node
 * @param ori_tensor
 * @return help node's output tensor
 *
 * Use a activation(leakyrelu) to move data
 * TODO: build a ddr->gm op?
 */
TensorPtr insert_dma_to_gm_help_node(TensorPtr ori_tensor)
{
    GraphPtr g = ori_tensor->node()->graph();
    std::string help_node_name = ori_tensor->node()->name() + "_help";
    std::string help_tensor_name = ori_tensor->name() + "_help";

    std::vector<int> help_tensor_shape = ori_tensor->shape();
    auto help_tensor_type = ori_tensor->type();
    TensorPtr help_tensor = std::make_shared<Tensor>(
        help_tensor_name,
        help_tensor_shape,
        help_tensor_type);
    help_tensor->shape() = ori_tensor->shape();

    NodePtr help_node = Node::create(help_node_name, OpType_Activation);
    help_node->add_input(ori_tensor);
    help_node->add_output(help_tensor);
    help_node->params()["method"] = "LEAKYRELU";
    help_node->params()["scale_type"] = Tensor::INT8;
    help_node->params()["scale_value"] = 1;
    help_node->params()["shift_type"] = Tensor::INT8;
    help_node->params()["shift_value"] = 0;
    help_node->params()["negative_slope_type"] = Tensor::INT8;
    help_node->params()["negative_slope_shift"] = 0;
    help_node->params()["negative_slope_value"] = 1;

    g->add_node(help_node);
    ADD_LOCATION_BY_NODE(ori_tensor->node(), help_node);

    return help_tensor;
}

/**
 * @brief swap_reshape_transpose
 * @param g
 * @return
 */
bool swap_reshape_transpose(GraphPtr g)
{
    GraphMatcherPattern pattern = {
        {{"0", OpType_Reshape}, {"1", OpType_Transpose}},
        {{"0", "1"}},
    };

    GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);
    bool is_changed = false;
    auto matched = matcher.match(g);
    for (auto &&nodes : matched)
    {
        NodePtr reshape = nodes[0];
        NodePtr transpose = nodes[1];

        std::vector<int> reshape_in_shape = reshape->inputs()[0]->shape();
        std::vector<int> reshape_out_shape = reshape->outputs()[0]->shape();
        std::vector<int> perm = transpose->params()["perm"];

        if (!creteria(reshape_in_shape, reshape_out_shape, perm))
        {
            continue;
        }
        is_changed = true;

        // don't change origin i/o
        const TensorPtr sg_in = reshape->inputs()[0];
        const TensorPtr sg_out = transpose->outputs()[0];

        NodePtr parent_node = sg_in->node();

        int a = reshape_out_shape[0];
        int b = reshape_out_shape[1];
        int c = reshape_out_shape[2];
        int d = reshape_out_shape[3];

        TensorPtr tensor_between_transpose_reshape = reshape->outputs()[0];
        // add a default layout here
        // for aviod transpose meet unknow DataLayout_Unkown
        // e.g. if transpose not support NCHWC16T8
        TensorShape new_shape({1, c * d, a, b}, sg_in->shape().layout());
        tensor_between_transpose_reshape->shape() = new_shape;

        transpose->params()["perm"] = {0, 3, 1, 2};

        // when reading from ddr
        TensorPtr help_tensor = sg_in;
        if (createria_read_from_ddr(parent_node))
        {
            // insert a read data layout here
            help_tensor = insert_dma_to_gm_help_node(sg_in);
        }

        // swap!
        transpose->clear_inputs();
        transpose->clear_outputs();
        reshape->clear_inputs();
        reshape->clear_outputs();
        transpose->add_input(help_tensor);
        transpose->add_output(tensor_between_transpose_reshape);
        reshape->add_input(tensor_between_transpose_reshape);
        reshape->add_output(sg_out);
    }

    return is_changed;
}

} // namespace aipubt
