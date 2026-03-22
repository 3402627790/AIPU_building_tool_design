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
 * @Author: Neo Wang
 * @Date: 2020-12-31 10:27:40
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-12-31 10:30:00
 */

#include "builder/lowering/lowering.h"
#include "builder/mem_remapping.h"
#include "core/core.h"
#include <algorithm>
namespace aipubt
{

bool remove_constant(GraphPtr g)
{
    auto m_nodes = g->nodes();
    bool ret = false;
    for (auto &&n : m_nodes)
    {
        if (n->type() == OpType_Constant && !n->outputs()[0]->is_io_tensor())
        {
            auto c = Node::create(n->name(), OpType_InternalConstant);
            c->constants() = n->constants();
            c->constants()["weights"]->set_attr(Constants::zerocopy_cst, true);
            c->params() = std::move(n->params());
            std::vector<LayoutList> valid_layouts = {
                {DataLayout_NCHWC32},
                {DataLayout_NCHWC16},
                {DataLayout_NHWC},
                {DataLayout_Flat},
            }; // support all
            c->set_attr(Constants::valid_input_layouts, valid_layouts);
            c->set_attr(Constants::valid_output_layouts, valid_layouts);
            g->replace_node_by(n, c);
            ret = true;
        }
    }
    return ret;
}
}; // namespace aipubt
