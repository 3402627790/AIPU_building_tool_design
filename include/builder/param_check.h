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
 * @Author: Samule Han
 * @Date: 2020-03-25 09:54:10
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-07-28 11:14:29
 */

#ifndef __PARAM_CHECK__
#define __PARAM_CHECK__

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <memory>
#define TPC_PARAM_CHECK(n)                                                                 \
    do                                                                                     \
    {                                                                                      \
        bool ret = true;                                                                   \
        ret = ret && aipubt::check_tpc_type(n);                                            \
        ret = ret && aipubt::check_tpc_value(n);                                           \
        if (!ret)                                                                          \
        {                                                                                  \
            ERROR() << optype_to_string(n->type()) << " ro generator param check failed."; \
        }                                                                                  \
    } while (0)

#endif //end of __PARAM_CHECK__

namespace aipubt
{
class Node;
using NodePtr = std::shared_ptr<Node>;
bool check_tpc_value(NodePtr  node);
bool check_tpc_type(NodePtr  node);

}; // namespace aipubt
