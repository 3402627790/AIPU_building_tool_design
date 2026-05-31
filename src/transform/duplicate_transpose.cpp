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
 * @Author: Yuchou.Gan
 * @Date: 2020-02-25 17:58:05
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-05-12 12:40:44
 */
#include "builder/lowering/lowering.h"
#include "core/core.h"
#include <algorithm>
#include <builder/target.h>
#include <cmath>
#include <functional>
#include <sstream>
namespace aipubt
{
std::atomic<int> duplicate_transpose_counter{0};

static bool criteria(const NodePtr node)
{
    if (node->type() != OpType_Transpose)
    {
        return false;
    }
    if (node->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    if (node->children().size() < 2)
    {
        return false;
    }

    if (duplicate_transpose_counter != 0)
    {
        return false;
    }
    return true;
};
/*
          input
            |
        transpose
        /        \
       /          \
    ==========>
           input
           /   \
        trans  trans
         /       \
*/
bool duplicate_transpose(GraphPtr g)
{
    bool is_changed = false;
    auto m_nodes = g->nodes();
    for (auto &&n : m_nodes)
    {
        if (!criteria(n))
        {
            continue;
        }

        auto n_children = n->children();
        for (size_t i = 0; i < n_children.size(); i++)
        {
            NodePtr transpose = Node::create(g->get_valid_node_name(n->name()), OpType_Transpose);

            ADD_LOCATION_BY_NODE(transpose, n);
            g->add_node(transpose);
            for (auto &&param : n->params())
            {
                transpose->params()[param.first] = param.second;
            }
            transpose->add_input(n->inputs());

            // create subnode outputs
            TensorPtr transpose_out = std::make_shared<Tensor>(
                g->get_valid_tensor_name(transpose->name()),
                n->outputs()[0]->shape(),
                n->outputs()[0]->type());
            transpose_out->quantization() = n->outputs()[0]->quantization();
            transpose->add_output(transpose_out);
            n_children[i]->replace_input_by(n->outputs()[0], transpose_out);
        }
        g->remove_node(n);
        is_changed = true;
    }
    duplicate_transpose_counter = 1;
    return is_changed;
}
}; // namespace aipubt
