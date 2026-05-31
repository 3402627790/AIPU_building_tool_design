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
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2024-09-03 16:20:39
 */
#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/location.h"
#include "core/ops/ops.h"
#include "quantization_transform/passes/passes.h"

namespace qtlibn
{
// [CSM-64].onnx model has globalavgpool different scale/zp of input/output
// static bool criterion(aipubt::NodePtr n)
// {
//     if (n->type() == aipubt::OpType_Pooling && n->params()["method"] == "AVG")
//     {
//         auto pooling_support = [](aipubt::NodePtr n) {
//             if (n->inputs()[0]->quantization().offset() == n->outputs()[0]->quantization().offset() &&
//                 (n->inputs()[0]->quantization().scale() == n->outputs()[0]->quantization().scale() ||
//                 n->inputs()[0]->quantization().offset() == 0))
//             {
//                 return true;
//             }
//             return false;
//         };

//         if (is_global_avgpool(n) && !pooling_support(n))
//         {
//             return true;
//         }
//     }
//     return false;
// }

/**
 * @brief [CSM-64].onnx model
 * layerlib's pooling doesn't support input/output's scale/zp are different
 */
static inline void globalpool_to_reduce(aipubt::GraphPtr g, aipubt::NodePtr n, const std::string method)
{
    auto reduce_method = method == "AVG" ? aipubt::ops::ReduceMethod::Mean : aipubt::ops::ReduceMethod::Max;

    bool unquantifiable = n->inputs()[0]->quantization().unquantifiable();
    n->inputs()[0]->quantization().set_unquantifiable(true);
    auto output_t = aipubt::ops::reduce(n->inputs()[0], reduce_method, {1, 2}, true, n->outputs()[0]->quantization());
    auto reduce_n = output_t->node();
    reduce_n->inputs()[0]->quantization().set_unquantifiable(unquantifiable);
    reduce_n->copy_attrs(n);

    g->replace_node_by(n, reduce_n);
}

static inline bool is_global_pool(aipubt::NodePtr n)
{
    if (n->type() == aipubt::OpType_Pooling &&
        n->params()["kernel_y"] == n->inputs()[0]->shape().h() &&
        n->params()["kernel_x"] == n->inputs()[0]->shape().w() &&
        n->params()["pad_top"] == 0 &&
        n->params()["pad_bottom"] == 0 &&
        n->params()["pad_left"] == 0 &&
        n->params()["pad_right"] == 0)
    {
        return true;
    }
    return false;
};

static inline void split_pool_or_reduce(aipubt::GraphPtr g, aipubt::NodePtr n)
{
    // pool splits to pool(in_scale==out_scale and in_zp == out_zp) + learkyrelu
    // reduce_max/reduce_min splits to reduce_max/reduce_min(in_scale==out_scale and in_zp==out_zp) + leakrelu
    auto out = n->outputs()[0];
    auto out_quantization = out->quantization();

    // 1. get leakyrelu input tensor and update quantization to input quantization
    auto leaky_input = std::make_shared<aipubt::Tensor>(g->get_valid_tensor_name(n->outputs()[0]->name()), n->outputs()[0]->shape(), n->outputs()[0]->type());
    leaky_input->quantization() = n->inputs()[0]->quantization();

    n->remove_output(out);
    n->add_output(leaky_input);

    // 2. generate leakyrelu to compensate the zp
    leaky_input->quantization().set_unquantifiable(true);
    auto leakyrelu_out = aipubt::ops::leaky_relu(leaky_input, (float)1.0, out_quantization);
    auto leakyrelu_node = leakyrelu_out->node();
    leakyrelu_node->copy_attrs(n);
    leakyrelu_node->remove_output(leakyrelu_out);
    leakyrelu_node->add_output(out);

    g->add_node(leakyrelu_node);
}

void pool_and_reduce_preprocess(aipubt::GraphPtr g)
{
    auto m_nodes = g->nodes();
    for (auto n : m_nodes)
    {
        if (n->type() != aipubt::OpType_Pooling && n->type() != aipubt::OpType_Reduce)
        {
            continue;
        }
        if (n->type() == aipubt::OpType_Pooling)
        {
            if (n->params()["method"] != "MAX" && n->params()["method"] != "AVG")
            {
                continue;
            }
        }

        if (n->type() == aipubt::OpType_Reduce)
        {
            if (n->params()["method"] != "MAX" && n->params()["method"] != "MIN")
            {
                continue;
            }
        }

        auto method = n->params()["method"];
        auto in_scale = n->inputs()[0]->quantization().scale();
        auto in_zp = n->inputs()[0]->quantization().offset();
        auto out_scale = n->outputs()[0]->quantization().scale();
        auto out_zp = n->outputs()[0]->quantization().offset();
        if (in_scale == out_scale && in_zp == out_zp)
        {
            continue;
        }
        else if (in_scale == out_scale && in_zp != out_zp)
        {
            split_pool_or_reduce(g, n);
        }
        else if (in_scale != out_scale && in_zp == out_zp && in_zp == 0)
        {
            bool is_globalpool = is_global_pool(n);
            if (is_globalpool)
            {
                /* when is_globalpool == true, inScale!= outScale, inZp==outZp==0
                 * in pooling.cpp, pool.quantize can handle this condition, qtlib
                 * will calculate the scale_value/shift_value, which lib can directly
                 * use.
                 */
                // globalpool_to_reduce(g, n, method);
            }
            else
            {
                WARN() << "QTLib now does not support this node(=" << n->type() << ")(with method=" << method
                       << ") which in_scale(=" << in_scale << ") != out_scale(=" << out_scale
                       << ") and in_zp(=" << in_zp << ") == out_zp(=" << out_zp
                       << ") == 0, and will split this pool to pool + leakyrelu(which may cause accurary issue).";
                split_pool_or_reduce(g, n);
            }
        }
        else if (in_scale != out_scale && in_zp == out_zp && in_zp != 0)
        {
            bool is_globalpool = is_global_pool(n);
            if (is_globalpool)
            {
                globalpool_to_reduce(g, n, method);
            }
            else
            {
                WARN() << "QTLib now does not support this node(=" << n->type() << ")(with method=" << method
                       << ") which in_scale(=" << in_scale << ") != out_scale(=" << out_scale
                       << ") and in_zp(=" << in_zp << ") == out_zp(=" << out_zp
                       << ") != 0, and will split this pool to pool + leakyrelu(which may cause accurary issue).";
                split_pool_or_reduce(g, n);
            }
        }
        else // in_scale != out_scale and in_zp != out_zp
        {
            WARN() << "QTLib now does not support this node(=" << n->type() << ")(with method=" << method
                   << ") which in_scale(=" << in_scale << ") != out_scale(=" << out_scale
                   << ") and in_zp(=" << in_zp << ") != out_zp(=" << out_zp
                   << ") != 0, and will split this pool to pool + leakyrelu(which may cause accurary issue).";
            auto is_globalpool = is_global_pool(n);
            if (is_globalpool)
            {
                globalpool_to_reduce(g, n, method);
            }
            else
            {
                split_pool_or_reduce(g, n);
            }
        }
    }
}
}; // namespace qtlibn
