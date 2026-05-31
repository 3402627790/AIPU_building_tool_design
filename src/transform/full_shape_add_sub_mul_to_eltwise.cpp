//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2022 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Yuxian Chen
 * @Date: 2023-04-24 18:13:51
 * @Last Modified by: Yuxian Chen
 * @Last Modified time: 2023-04-24 18:13:51
 */

#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include "quantization_transform/utils/quantize_utils.h"

namespace aipubt
{

static inline bool criteria(NodePtr n, std::string &method)
{
    static std::map<OpType, std::string> tmap = {
        {OpType_Add, "ADD"},
        {OpType_Sub, "SUB"},
        {OpType_Mul, "MUL"},
    };
    if (tmap.count(n->type()))
    {
        auto inputs = n->inputs();
        if (inputs[0]->shape() == inputs[1]->shape())
        {
            method = tmap[n->type()];
            return true;
        }
    }
    return false;
}

bool full_shape_add_sub_mul_to_eltwise(GraphPtr g)
{
    bool is_changed = false;
    auto nodes = g->nodes();
    for (auto &&n : nodes)
    {
        std::string method = "NONE";
        if (!criteria(n, method))
        {
            continue;
        }

        auto eltwise = Node::create(n->name(), OpType_Eltwise);

        eltwise->constants() = std::move(n->constants());
        eltwise->params() = std::move(n->params());
        eltwise->params()["method"] = method;
        if (eltwise->params().count("with_activation") == 0)
        {
            eltwise->params()["with_activation"] = "NONE";
        }
        ADD_LOCATION_BY_NODE(eltwise, n);
        g->replace_node_by(n, eltwise);
        is_changed = true;
    }

    return is_changed;
}
}; // namespace aipubt