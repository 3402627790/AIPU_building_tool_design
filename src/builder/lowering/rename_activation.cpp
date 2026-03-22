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
 * @Last Modified time: 2020-10-13 13:44:11
 */
#include "core/core.h"
#include "builder/lowering/lowering.h"
#include <algorithm>
#include <set>
namespace aipubt
{
const static std::set<std::string> support_method = {
    "TANH",
    "SIGMOID",
    "RELU"};

void rename_activation(GraphPtr g)
{
    auto criterion = [](NodePtr node) -> bool {
        if (node->inputs().empty())
        {
            return false;
        }
        bool res = OpType_Activation == node->type();
        return res;
    };

    auto m_nodes = g->nodes();
    for (auto &&n : m_nodes)
    {
        if (!criterion(n))
        {
            continue;
        }

        std::string method = n->params()["method"];
        std::transform(method.begin(), method.end(), method.begin(), ::toupper);

        if (support_method.find(method) == support_method.end())
        {
            ERROR() << "Unspported activation type "
                    << method << " at node " << n;
            exit(-1);
        }
    }
}
}; // namespace aipubt
