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
 * @Date: 2024-07-23 16:22:35
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-07-23 16:22:35
 */
#include "builder/lowering/lowering.h"
#include "core/core.h"

namespace aipubt
{

bool remove_layout_convertor(GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto &&n : nodes)
    {
        if (n->type() == OpType_LayoutConvertor)
        {
            n->set_attr("to", n->outputs()[0]->shape().layout());
            TensorShape inp_shape = n->inputs()[0]->shape();
            TensorShape outp_shape = n->outputs()[0]->shape();
            if (inp_shape == outp_shape)
            {
                if (n->attrs().count(Constants::use_out_tensor))
                {
                    for (auto c : n->children())
                    {
                        c->add_input(n->inputs()[0], c->remove_input(n->outputs()[0]));
                    }
                }
                else // use_in_tensor
                {
                    for (auto p : n->parents())
                    {
                        for (auto c : n->inputs()[0]->consumers())
                        {
                            if (c != n)
                            {
                                c->add_input(n->outputs()[0], c->remove_input(n->inputs()[0]));
                            }
                        }
                        p->add_output(n->outputs()[0], p->remove_output(n->inputs()[0]));
                    }
                }
                g->remove_node(n);
            }
        }
    }
    return true;
}

} // namespace aipubt
