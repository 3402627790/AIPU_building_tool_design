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
 * @Author: Lang Hu
 * @Date: 2024-11-11 15:20:57
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-11-11 15:20:57
 */
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/core.h"
#include "core/config.h"
#include "core/ops/ops.h"

namespace aipubt
{
static inline bool criteria(NodePtr n0, NodePtr n1)
{
    if (n0->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    if (!n0->attrs().count(Constants::w_align))
    {
        return false;
    }
    std::vector<int> perm = n1->params()["perm"];
    if (perm != std::vector<int>({0, 1, 3, 2}))
    {
        return false;
    }
    return true;
}

void lift_transpose(GraphPtr g)
{
    static GraphMatcherPattern pattern = {
        {{"0", OpType_Pad}, {"1", OpType_Transpose}},
        {{"0", "1"}},
    };
    static GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::StrictMode);

    auto matched = matcher.match(g);
    while (matched.size())
    {
        bool is_lifted = false;
        for (auto &&m : matched)
        {
            if (!criteria(m[0], m[1]))
            {
                continue;
            }
            auto pad_in = m[0]->inputs()[0];
            m[1]->clear_inputs();
            m[1]->add_input(pad_in);
            std::vector<int> shape_list = pad_in->shape();
            std::swap(shape_list[2], shape_list[3]);
            auto trans_out = m[1]->outputs()[0];
            trans_out->shape() = TensorShape(shape_list, pad_in->shape().layout());
            m[1]->infer_shape(true);
            MODIFY_NODE(m[1]);

            auto consumers = trans_out->consumers();
            m[0]->add_input(trans_out, m[0]->remove_input(pad_in));
            int alignment = m[0]->get_attr<int>(Constants::w_align);
            shape_list[3] = (shape_list[3] + alignment - 1) / alignment * alignment;
            auto pad_out = m[0]->outputs()[0];
            pad_out->shape() = TensorShape(shape_list, trans_out->shape().layout());
            std::vector<std::vector<int>> pads(4, {0, 0});
            std::vector<std::vector<Expr>> ds_pads(4, {0, 0});
            pads[3][1] = (alignment - trans_out->shape()[3] % alignment) % alignment;
            ds_pads[3][1] = (alignment - trans_out->shape().symbol()[3] % alignment) % alignment;
            m[0]->params()["pads"] = pads;
            m[0]->params()["ds_pads"] = ds_pads;
            m[0]->erase_attr(Constants::w_align);
            m[0]->set_attr(Constants::c_align, alignment);
            m[0]->infer_shape(true);
            MODIFY_NODE(m[0]);

            for (auto c : consumers)
            {
                c->add_input(pad_out, c->remove_input(trans_out));
                c->infer_shape(true);
                MODIFY_NODE(c);
            }

            is_lifted = true;
        }
        if (!is_lifted)
        {
            break;
        }
        matched = matcher.match(g);
    }
}
} // namespace aipubt
