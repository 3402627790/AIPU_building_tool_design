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
 * @Author: Kyle Huang
 * @Date: 2023-9-15 10:27:40
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2023-9-15 10:27:40
 */
#include "core/core.h"
#include <algorithm>

namespace aipubt
{
bool set_layer_id(GraphPtr g)
{
    int real_layer_id = 0;
    for (auto node : g->nodes()) // set final layer id
    {
        auto sgn = SubGraphNode::from_node(node);
        if (sgn)
        {
            for (auto n : sgn->get_real_node())
            {
                n->set_attr("layer_id", real_layer_id);
                real_layer_id++;
            }
        }
        else
        {
            node->set_attr("layer_id", real_layer_id);
            real_layer_id++;
        }
    }
    return true;
}
} // namespace aipubt
