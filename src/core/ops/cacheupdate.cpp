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
 * @Date: 2024-12-04 10:37:09
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-12-04 10:37:09
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>

namespace aipubt
{
namespace ops
{
    std::vector<TensorPtr> cacheupdate(TensorPtr cache,
                                       TensorPtr position_id,
                                       TensorPtr value,
                                       int block_index,
                                       int axis)
    {
        auto g = cache->node()->graph();

        if (cache->shape().dim() != 3 && cache->shape().dim() != 4)
        {
            ERROR() << "unsupported cache dimension: " << cache->shape().dim();
            return std::vector<TensorPtr>();
        }
        if (value->shape().dim() != 4)
        {
            ERROR() << "unsupported value dimension: " << value->shape().dim();
            return std::vector<TensorPtr>();
        }
        if (position_id->shape().dim() != 2)
        {
            ERROR() << "unsupported position_id dimension: " << position_id->shape().dim();
            return std::vector<TensorPtr>();
        }
        if (position_id->type() != Tensor::INT32 &&
            position_id->type() != Tensor::UINT32)
        {
            ERROR() << "unsupported position_id data type: "
                    << Tensor::dtype_to_string(position_id->type());
            return std::vector<TensorPtr>();
        }
        auto block_num = cache->shape()[0];
        if (cache->shape().dim() == 3)
        {
            block_num = 1;
        }
        if (block_index < 0 || block_index >= block_num)
        {
            ERROR() << "unsupported block_index: " << block_index;
            return std::vector<TensorPtr>();
        }
        if (axis != 2 && axis != 3)
        {
            ERROR() << "unsupported axis: " << axis;
            return std::vector<TensorPtr>();
        }
        int max_len = cache->shape()[2];
        if (cache->shape().dim() == 3)
        {
            max_len = cache->shape()[1];
        }
        int align = 32 / Tensor::dtype_size(value->type());
        if (max_len % align != 0)
        {
            ERROR() << "max_len must be " << align << " aligned, got " << max_len;
            return std::vector<TensorPtr>();
        }
        std::vector<int> shape = value->shape();
        if (shape[axis] > max_len)
        {
            ERROR() << "unsupported value shape: " << value->shape().to_string();
            return std::vector<TensorPtr>();
        }

        shape[axis] = max_len;
        auto name = g->get_valid_tensor_name("cacheupdate");
        auto output0 = std::make_shared<Tensor>(name, TensorShape(shape, value->shape().layout()));
        auto output1 = std::make_shared<Tensor>(g->get_valid_tensor_name(name + "/dump"), cache->shape());
        add_tensor_info(value, output0);
        add_tensor_info(cache, output1);

        auto node = Node::create(g->get_valid_node_name("cacheupdate"), OpType_CacheUpdate);
        node->add_input(cache);
        node->add_input(position_id);
        node->add_input(value);
        node->add_output(output0);
        node->add_output(output1);
        node->params()["block_index"] = block_index;
        node->params()["axis"] = axis;
        g->add_node(node);
        quantize(node);

        return {output0, output1};
    }
} // namespace ops
} // namespace aipubt
