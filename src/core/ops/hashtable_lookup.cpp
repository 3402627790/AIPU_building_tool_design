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
 * @Author: ALex Zheng
 * @Date: 2021-07-14 10:10:54
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-05-12 20:25:48
 */

#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>

namespace aipubt
{
namespace ops
{
    std::vector<TensorPtr> hashtable_lookup(TensorPtr value,
                                            TensorPtr key,
                                            TensorPtr lookup)
    {
        GraphPtr g = value->node()->graph();
        if ((value == nullptr) || (key == nullptr) || lookup == nullptr)
        {
            ERROR() << "value or key or lookup is nullptr!";
            return {};
        }
        if (key->shape().dim() != 1)
        {
            ERROR() << "key shape dim must equal 1!";
            return {};
        }
        std::vector<int32_t> output_dims;
        output_dims.insert(output_dims.begin(), lookup->shape().begin(), lookup->shape().end());
        output_dims.insert(output_dims.end(), value->shape().begin() + 1, value->shape().end());
        TensorShape output_shape(output_dims);
        output_shape.set_layout(value->shape().layout());

        TensorShape hits_shape = lookup->shape();
        hits_shape.set_layout(value->shape().layout());

        TensorPtr output = std::make_shared<Tensor>(g->get_valid_tensor_name("hashtable_lookup_output"), output_shape);
        add_tensor_info(value, output);

        TensorPtr hits = std::make_shared<Tensor>(g->get_valid_tensor_name("hashtable_lookup_hits"), hits_shape);
        // float IR we set hits dtype as int32
        if (value->quantization().empty())
        {
            hits->set_type(Tensor::INT32);
        }
        else
        {
            hits->quantization() = Quantization(1.0, 0, 8, value->quantization().scheme());
            hits->set_type(Tensor::UINT8);
        }

        NodePtr node = Node::create(g->get_valid_node_name("hashtable_lookup"), OpType_HashLookupTable);

        node->params()["miss_default_data"] = 0.f;
        node->set_attr("pad_quantized", NodeParamValue(false));
        node->add_input(value);
        node->add_input(key);
        node->add_input(lookup);
        node->add_output(output);
        node->add_output(hits);

        g->add_node(node);
        quantize(node);
        return {output, hits};
    }
}; //namespace ops
}; // namespace aipubt