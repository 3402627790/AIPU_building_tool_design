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
 * @Author: Alex Zheng
 * @Date: 2023-10-24 10:55:59
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2023-10-24 10:55:59
 */

#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>

namespace aipubt
{
namespace ops
{
    TensorPtr embedding_lookup_sparse(TensorPtr params,
                                      TensorPtr sp_ids,
                                      TensorPtr sp_value,
                                      TensorPtr weight_value,
                                      const TensorShape dense_shape,
                                      const EmbeddingLookupSparseCombiner combiner,
                                      const float max_norm,
                                      const Quantization &quant)
    {
        if ((params == nullptr) || (sp_ids == nullptr) || (sp_value == nullptr) || (weight_value == nullptr))
        {
            ERROR() << "Any input tensor cannot be empty";
            return nullptr;
        }
        TensorShape ds_shape = dense_shape;
        if (ds_shape.dim() == 0)
        {
            if (sp_ids->shape().dim() != 1)
            {
                ERROR() << "The sp_ids shape is not right.";
                return nullptr;
            }
            ds_shape = TensorShape({sp_ids->shape()[0]});
        }

        if (sp_ids->shape().dim() != ds_shape.dim())
        {
            ERROR() << "The sp_ids shape is not right.";
            return nullptr;
        }
        if ((sp_value->shape().dim() != 1) || (weight_value->shape().dim() != 1))
        {
            ERROR() << "sp_ids,sp_value,weight_value only support 1-dimension.";
            return nullptr;
        }
        if ((sp_value->shape()[0] != sp_ids->shape()[0]) || (sp_value->shape()[0] != weight_value->shape()[0]))
        {
            ERROR() << "sp_value,weight_value must have the same batch.";
            return nullptr;
        }
        if ((sp_value->shape()[0] > ds_shape.size()) || sp_value->shape()[0] < ds_shape[0])
        {
            ERROR() << "sp_value lenth must in rank [" << ds_shape[0] << "," << ds_shape.size() << "].";
            return nullptr;
        }

        std::map<EmbeddingLookupSparseCombiner, std::string> combiner_str =
            {
                {EmbeddingLookupSparseCombiner::MEAN, "MEAN"},
                {EmbeddingLookupSparseCombiner::SUM, "SUM"},
                {EmbeddingLookupSparseCombiner::SQRTN, "SQRTN"},
            };
        auto g = params->node()->graph();
        TensorShape output_shape = params->shape();
        output_shape[0] = ds_shape[0];
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("embedding_lookup_sparse"), output_shape);
        add_tensor_info(params, output, quant);

        auto node = Node::create(g->get_valid_node_name("embedding_lookup_sparse"), OpType_EmbeddingLookupSparse);
        node->add_input(params);
        node->add_input(sp_ids);
        node->add_input(sp_value);
        node->add_input(weight_value);
        node->add_output(output);
        auto &node_params = node->params();
        node_params["combiner"] = combiner_str[combiner];
        if (max_norm == 0)
        {
            node_params["max_norm"] = NodeParamValue("None");
        }
        else
        {
            node_params["max_norm"] = NodeParamValue(std::to_string(max_norm));
        }

        g->add_node(node);
        quantize(node);
        return output;
    }
}; // namespace ops
}; // namespace aipubt