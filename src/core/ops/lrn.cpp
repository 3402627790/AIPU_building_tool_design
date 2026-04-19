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

#include "utils/utils.h"
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>

namespace aipubt
{
namespace ops
{
    TensorPtr lrn(TensorPtr input,
                  int32_t size,
                  LrnMethod method,
                  float_t alpha,
                  float_t beta,
                  float_t bias,
                  const Quantization &quant)
    {

        std::map<LrnMethod, std::string> method_table =
        {
            {LrnMethod::AcrossChannels, "ACROSS_CHANNELS"},
            {LrnMethod::WithinChannel, "WITHIN_CHANNEL"},
        };
        auto g = input->node()->graph();
        auto node = Node::create(g->get_valid_node_name("lrn"), OpType_LRN);
        TensorShape output_shape = input->shape();
        TensorPtr output = std::make_shared<Tensor>(g->get_valid_tensor_name("lrn"), output_shape);
        add_tensor_info(input, output, quant);
        node->add_input(input);
        node->add_output(output);

        node->params()["size"] = size;
        node->params()["alpha"] = alpha;
        node->params()["beta"] = beta;
        node->params()["bias"] = bias;
        node->params()["method"] = enum_to_string(method_table, method);
        g->add_node(node);
        node->infer_shape(true);
        quantize(node);

        return output;
    }

}; //namespace ops
}; // namespace aipubt
