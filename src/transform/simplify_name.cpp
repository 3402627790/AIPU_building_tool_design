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
 * @Author: Tom Gao
 * @Date: 2023-02-16 14:49:22
 * @Last Modified by: Tom Gao
 * @Last Modified time: 2023-02-16 15:52:20
 */
#include "core/core.h"
#include "utils/utils.h"
#include <algorithm>
#include <set>

namespace aipubt
{

static void simplify_node_names(GraphPtr g)
{
    std::map<OpType, std::string> name_map = {
        {OpType_Input, "input"},
        {OpType_Convolution, "conv"},
        {OpType_DepthwiseConv, "dwconv"},
        {OpType_ConvTranspose, "deconv"},
        {OpType_FullyConnected, "fc"},
        {OpType_DepthToSpace, "d2s"},
    };
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        auto layer_id = n->get_attr<int32_t>("layer_id");
        auto type = n->type();
        auto type_str = optype_to_string(type);
        std::string new_name = name_map.count(type) ? name_map.at(type) : utils::tolower(type_str);
        new_name += +"_" + std::to_string(layer_id);
        n->set_name(new_name);
    }
}

static void simplify_tensor_names(GraphPtr g)
{
    auto nodes = g->nodes();
    auto input_tensors = g->get_input_tensors();
    auto output_tensors = g->get_output_tensors();
    for (auto n : nodes)
    {
        auto new_name = n->name() + "_out";
        if (n->outputs().size() == 1)
        {
            n->outputs()[0]->set_name(new_name);
        }
        else if (n->outputs().size() > 1)
        {
            for (size_t i = 0; i < n->outputs().size(); ++i)
            {
                n->outputs()[0]->set_name(new_name + "_" + std::to_string(i));
            }
        }
    }
}

bool simplify_name(GraphPtr g)
{
    INFO() << "try to simplify node/tensor names...";
    simplify_node_names(g);
    simplify_tensor_names(g);
    // g->serialize("graph_sim.def", "weight_sim.bin");
    return true;
}
}; // namespace aipubt
