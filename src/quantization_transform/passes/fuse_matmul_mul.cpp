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
 * @Author: Samuel Han
 * @Date: 2025-01-07 10:56:28
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-01-07 16:53:37
 */

#include "core/core.h"
#include "core/location.h"
#include "quantization_transform/passes/passes.h"

using namespace aipubt;

namespace qtlibn
{

static bool criteria(std::vector<aipubt::NodePtr> nodes)
{
    auto node_num = nodes.size();
    aipubt::NodePtr elt_node = nodes[node_num - 1];

    if (elt_node->params().count("method") && elt_node->params()["method"] != "MUL")
    {
        return false;
    }

    if (elt_node->params().count("with_activation") && elt_node->params()["with_activation"] != "NONE")
    {
        return false;
    }

    aipubt::NodePtr elt_another_in_node = elt_node->inputs()[0]->node();
    if (elt_another_in_node->type() == aipubt::OpType_Constant)
    {
        elt_another_in_node = elt_node->inputs()[1]->node();
    }

    // check matmul or matmul child output consumer, if has multi-consumers, should not merge
    for (uint32_t i = 0; i < nodes.size() - 1; i++)
    {
        auto n = nodes[i];
        if (n->outputs()[0]->consumers().size() > 1)
        {
            return false;
        }
    }

    if (elt_node->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    return true;
}

void fuse_matmul_mul(aipubt::GraphPtr g)
{
    std::vector<aipubt::GraphMatcherPattern> patterns = {
        {{{"0", aipubt::OpType_MatMul}, {"1", aipubt::OpType_Reshape}, {"2", aipubt::OpType_Constant}, {"3", aipubt::OpType_Eltwise}},
         {{"0", "1"}, {"1", "3"}, {"2", "3"}}},
        {{{"0", aipubt::OpType_MatMul}, {"1", aipubt::OpType_Constant}, {"2", aipubt::OpType_Mul}},
         {{"0", "2"}, {"1", "2"}}},
    };

    for (auto &pattern : patterns)
    {
        aipubt::GraphMatcher matcher(pattern);
        matcher.set_mode(aipubt::GraphMatcher::MatchMode::StrictMode);
        auto matched = matcher.match(g);
        while (matched.size())
        {
            bool is_merged = false;
            for (auto &&m : matched)
            {
                if (!criteria(m))
                {
                    continue;
                }
                auto node_num = m.size();
                auto const_node = m[node_num - 2];

                aipubt::ComputeTensor w = const_node->constants()["weights"];
                auto unique_w = w.unique();
                auto uq_size = unique_w->shape().size();
                if (uq_size != 1)
                {
                    continue;
                }

                auto const_qinfo = const_node->constants()["weights"]->quantization();
                auto const_dtype = const_node->constants()["weights"]->type();
                float_t fused_multiplier = 1.0;
                if (const_node->attrs().count("fused_multiplier"))
                {
                    fused_multiplier = const_node->get_attr("fused_multiplier");
                    // INFO() << "fused_multiplier in attrs "<< fused_multiplier;
                }
                else if (aipubt::Tensor::is_float(const_dtype))
                {
                    fused_multiplier = float(unique_w[0]);
                }
                else if (!const_qinfo.empty())
                {
                    auto scale = const_qinfo.scales()[0];
                    auto zp = const_qinfo.offsets()[0];
                    // dequantize the qvalue to float value
                    fused_multiplier = float((unique_w.astype(aipubt::Tensor::INT32) - zp) * scale);
                }
                else
                {
                    WARN() << "unsupported condition, will not fuse matmul_mul";
                    continue;
                }

                auto elt_node = m[node_num - 1];
                auto matmul_node = m[0];
                matmul_node->params()["fused_multiplier"] = fused_multiplier;
                auto elt_qinfo = elt_node->outputs()[0]->quantization();

                auto elt_another_in_node = elt_node->inputs()[0]->node();
                if (elt_another_in_node->type() == aipubt::OpType_Constant)
                {
                    elt_another_in_node = elt_node->inputs()[1]->node();
                }

                // update matmul qinfo, and its matmul's child qinfo
                if (!elt_qinfo.empty())
                {
                    matmul_node->outputs()[0]->quantization() = elt_qinfo;
                    elt_another_in_node->outputs()[0]->quantization() = elt_qinfo;
                }

                // modify the graph
                const_node->clear_outputs();
                g->remove_node(const_node);

                auto new_out = elt_another_in_node->outputs()[0];
                auto elt_consumers = elt_node->outputs()[0]->consumers();
                for (auto consumer : elt_consumers)
                {
                    auto idx = consumer->remove_input(elt_node->outputs()[0]);
                    consumer->add_input(new_out, idx);
                }

                ADD_LOCATION_BY_NODE(matmul_node, const_node, elt_node);
                elt_node->clear_inputs();
                elt_node->clear_outputs();
                g->remove_node(elt_node);
                is_merged = true;
            }
            if (!is_merged)
            {
                break;
            }
            matched = matcher.match(g);
        }
    }
}

} // namespace qtlibn
