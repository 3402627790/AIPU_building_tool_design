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
 * @Date: 2023-11-03 11:19:03
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2023-11-03 11:19:03
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    std::vector<TensorPtr> collapse_repeated(TensorPtr labels, TensorPtr seq_length)
    {
        if (labels->shape().dim() != 2)
        {
            ERROR() << "Labels tensor shape should be [batch, max value in seq_length].";
            return {};
        }
        if (seq_length->shape().dim() != 1)
        {
            ERROR() << "Seq_length tensor shape should be [batch].";
            return {};
        }
        auto g = labels->node()->graph();
        auto node = Node::create(g->get_valid_node_name("collapse_repeated"), OpType_CollapseRepeated);

        TensorShape collapsed_labels_shape(labels->shape());
        auto collapsed_labels = std::make_shared<Tensor>(g->get_valid_tensor_name("collapse_repeated/collapsed_labels"), collapsed_labels_shape);
        add_tensor_info(labels, collapsed_labels);

        TensorShape new_seq_length_shape(seq_length->shape());
        auto new_seq_length = std::make_shared<Tensor>(g->get_valid_tensor_name("collapse_repeated/new_seq_length"), new_seq_length_shape);
        add_tensor_info(seq_length, new_seq_length);

        node->add_input(labels);
        node->add_input(seq_length);
        node->add_output(collapsed_labels);
        node->add_output(new_seq_length);
        g->add_node(node);
        quantize(node);
        return {collapsed_labels, new_seq_length};
    }
}; //namespace ops
}; // namespace aipubt