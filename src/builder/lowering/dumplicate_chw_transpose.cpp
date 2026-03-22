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
static void duplicate_chw_transpose(GraphPtr g, std::function<bool(NodePtr)> criterion)
{
    auto m_nodes = g->nodes();
    for (auto &&n : m_nodes)
    {
        if (!criterion(n))
        {
            continue;
        }

        auto n_children = n->children();
        for (size_t i = 0; i < n_children.size(); i++)
        {
            NodePtr transpose = Node::create(n->name() + "_trans" + std::to_string(i), OpType_Transpose);

            ADD_LOCATION_BY_NODE(transpose, n);
            g->add_node(transpose);
            for (auto &&param : n->params())
            {
                transpose->params()[param.first] = param.second;
            }
            transpose->add_input(n->inputs());

            // create subnode outputs
            TensorPtr transpose_out = std::make_shared<Tensor>(transpose->name(), n->outputs()[0]->shape(), n->outputs()[0]->type());
            transpose_out->quantization() = n->outputs()[0]->quantization();
            transpose->add_output(transpose_out);
            n_children[i]->remove_input(-1);
            n_children[i]->add_input(transpose_out);
        }
        g->remove_node(n);
    }
}
/*
    onnx/caffe is nchw, hardware support nchw conv from x1 when c=3
    the following subgraph cannot match fused_op "transpose_conv" like CB-3334
          input
            |
        transpose
        /        \
       /          \
    conv          conv
    ==========>
           input
           /   \
        trans  trans
         /       \
        conv     conv
    the new subgraph has only two fused_op "transpose_conv" and no transpose cost
*/
void duplicate_chw_transpose(GraphPtr g)
{
    auto hwa_criterion = [](NodePtr node) -> bool {
        auto target = aipubt::get_target();
        aipubt::ISAType arch = target->isainfo().type;
        // only X1 or from X2
        if ((arch < aipubt::ISA_ZYv5) && (target->name() != "X1_1204"))
        {
            return false;
        }

        if (node->type() != OpType_Transpose)
        {
            return false;
        }
        std::vector<int> perm = node->params()["perm"];
        // perm=[0,2,3,1]
        if ((perm.size() != 4) || (perm[0] != 0) || (perm[1] != 2) || (perm[2] != 3) || (perm[3] != 1))
        {
            return false;
        }

        if (node->children().size() < 2)
        {
            return false;
        }
        // all children
        for (auto &c : node->children())
        {
            if (c->type() != OpType_Convolution)
            {
                return false;
            }

            auto conv_input = c->inputs()[0];
            auto conv_params = c->params();
            int kernel_x = conv_params["kernel_x"];
            int kernel_y = conv_params["kernel_y"];
            int stride_x = conv_params["stride_x"];
            int stride_y = conv_params["stride_y"];
            int group = conv_params["group"];
            if ((group != 1) ||
                (kernel_x == 1 && kernel_y == 1 && stride_x == 2 && stride_y == 2) ||
                (conv_input->shape().c_total() > 4))
            {
                return false;
            }
            if ((((conv_input->shape().h() * conv_input->shape().w() * Tensor::dtype_size(conv_input->type())) & 0xffffff) >> 24) != 0)
            {
                // although mtp_iact_surface_stride has 29bit reg from X1, hardware said the internal bit is 24bit they used
                // max input 4096*2160*2=0x10e_0000 which is 25bit
                // see e-mail "X1 RGB CHW mode ACT size"
                return false;
            }
        }

        return true;
    };
    duplicate_chw_transpose(g, hwa_criterion);
}
}; // namespace aipubt
