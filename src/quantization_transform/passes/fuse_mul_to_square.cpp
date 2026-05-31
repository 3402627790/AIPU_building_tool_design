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
 * @Date: 2024-10-12 11:21:13
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2024-10-12 11:21:13
 */

#include "core/core.h"
#include "core/location.h"
#include "quantization_transform/passes/passes.h"

using namespace aipubt;

namespace qtlibn
{

static bool criteria(aipubt::NodePtr node)
{
    if (node->type() == aipubt::OpType_Mul ||
        (node->type() == aipubt::OpType_Eltwise &&
         node->params()["method"] == "MUL"))
    {
        bool ret = node->inputs()[0] == node->inputs()[1];
        ret = ret && (node->params().count("with_activation") == 0 ||
                      node->params()["with_activation"] == "NONE");
        return ret;
    }
    return false;
}

void fuse_mul_to_square(aipubt::GraphPtr g)
{
    for (auto n : g->nodes())
    {
        if (!criteria(n))
        {
            continue;
        }

        auto square_n = Node::create(g->get_valid_node_name(n->name() + "_square"), aipubt::OpType_Square);
        square_n->copy_attrs(n);
        ADD_LOCATION_BY_NODE(square_n, n);
        g->replace_node_by(n, square_n);
        square_n->remove_input(1);
    }
}

} // namespace qtlibn
