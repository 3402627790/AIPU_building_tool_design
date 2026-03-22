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
 * @Last Modified time: 2024-08-21 16:34:20
 */
#include "hash_constant.h"
#include "core/core.h"
#include "utils/utils.h"
#include <algorithm>
#include <set>
namespace aipubt
{
bool HashConstant::operator()(GraphPtr g) const
{
    for (auto n : g->nodes())
    {
        for (auto kv : n->constants())
        {
            // avoid nullptr
            if (!kv.second || !kv.second->data<void>())
            {
                continue;
            }
            auto hash = utils::sha256_hex(kv.second->data<void>(), kv.second->mem_size());
            kv.second->set_attr(Constants::hash, hash);
        }
    }
    return false;
}

}; // namespace aipubt
