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
 * @Date: 2023-10-25 17:05:05
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2023-10-25 17:05:05
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>

namespace aipubt
{
namespace ops
{
    std::vector<TensorPtr> normalize_moments(TensorPtr mean_ss,
                                              TensorPtr variance_ss,
                                              TensorPtr shift,
                                              const int32_t counts,
                                              const std::vector<Quantization> &quants)
    {
        if (quants.size() != 2)
        {
            WARN() << "Recommend output quantization size == 2 when build the normalize_moments op in OpsAPI,"\
                    <<" but now is " << quants.size();
        }

        auto g = mean_ss->node()->graph();

        TensorPtr mean_output = std::make_shared<Tensor>(g->get_valid_tensor_name("normalize_moments/mean"), mean_ss->shape());
        if ((quants.size() >0) && !quants[0].empty())
        {
            add_tensor_info(mean_ss, mean_output, quants[0]);
        }
        else
        {
            add_tensor_info(mean_ss, mean_output);
        }

        TensorPtr variance_output = std::make_shared<Tensor>(g->get_valid_tensor_name("normalize_moments/variance"), variance_ss->shape());
        if ((quants.size() >1) && !quants[1].empty())
        {
            add_tensor_info(variance_ss, variance_output, quants[1]);
        }
        else
        {
            add_tensor_info(variance_ss, variance_output);
        }

        auto node = Node::create(g->get_valid_node_name("normalize_moments"), OpType_NormalizedMoments);
        node->add_input(mean_ss);
        node->add_input(variance_ss);
        node->add_input(shift);
        node->add_output(mean_output);
        node->add_output(variance_output);
        node->params()["counts"] = counts;
        g->add_node(node);
        quantize(node);

        return {mean_output, variance_output};
    }

}; //namespace ops
}; // namespace aipubt