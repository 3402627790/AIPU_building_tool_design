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
 * @Author: Yuchou Gan
 * @Date: 2024-09-5 10:41:00
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-12-09 11:41:12
 */

#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include "utils/utils.h"
#include <unordered_map>
#include <vector>

namespace aipubt
{
bool fuse_same_constant(GraphPtr g)
{
    bool is_changed = false;
    std::unordered_map<std::string, std::vector<NodePtr>> record;
    auto nodes = g->nodes();
    for (auto &n : nodes)
    {
        if (n->type() == OpType_Constant)
        {
            for (auto kv : n->constants())
            {
                // avoid nullptr
                if (!kv.second || !kv.second->data<void>())
                {
                    continue;
                }
                auto hash = utils::sha256_hex(kv.second->data<void>(), kv.second->mem_size());
                if (record.count(hash))
                {
                    record[hash].push_back(n);
                }
                else
                {
                    record[hash] = {n};
                }
            }
        }
    }

    for (auto &kv : record)
    {
        if (kv.second.size() > 1)
        {
            NodePtr const_node = kv.second[0];
            TensorPtr const_tensor = kv.second[0]->outputs()[0];
            if (const_tensor->shape().size() < 32)
            {
                continue;
            }
            for (uint32_t i = 1; i < kv.second.size(); i++)
            {
                TensorPtr origin_tensor = kv.second[i]->outputs()[0];
                auto consumers = origin_tensor->consumers();
                if (const_tensor->type() != origin_tensor->type())
                {
                    continue;
                }

                if (const_tensor->quantization() != origin_tensor->quantization())
                {
                    continue;
                }

                if (const_tensor->shape() != origin_tensor->shape())
                {
                    continue;
                }

                is_changed = true;
                for (auto consumer : consumers)
                {
                    consumer->replace_input_by(origin_tensor, const_tensor);
                }
                g->remove_node(kv.second[i], true);
                ADD_LOCATION_BY_NODE(const_node, kv.second[i]);
            }
        }
    }
    if (is_changed)
    {
        g->reset_layer_id();
    }
    return is_changed;
}

} // namespace aipubt