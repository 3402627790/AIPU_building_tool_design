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
 * @Date: 2022-12-03 15:45:06
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-12-14 18:01:38
 */

#include "core/core.h"
#include "core/ops/ops.h"
#include "quantization_transform/passes/passes.h"

namespace qtlibn
{

static std::vector<int> eliminate_shape(const aipubt::TensorShape &shape)
{
    std::vector<int> new_shape = shape;
    for (auto intr = shape.begin(); intr != shape.end(); ++intr)
    {
        if (*intr != 1)
        {
            new_shape.assign(intr, shape.end());
            break;
        }
    }
    return new_shape;
}

template <typename T>
static int64_t get_data(aipubt::TensorPtr t)
{
    auto data = t->data<T>();
    return static_cast<int64_t>(data[0]);
}

static std::map<aipubt::Tensor::Dtype, std::function<int64_t(aipubt::TensorPtr)>> _get_data_map = {
    {aipubt::Tensor::UINT8, get_data<uint8_t>},
    {aipubt::Tensor::INT8, get_data<int8_t>},
    {aipubt::Tensor::UINT16, get_data<uint16_t>},
    {aipubt::Tensor::INT16, get_data<int16_t>},
    {aipubt::Tensor::UINT32, get_data<uint32_t>},
    {aipubt::Tensor::INT32, get_data<int32_t>},
};

static aipubt::TensorPtr shrink_tensor(aipubt::TensorPtr orig_t)
{
    auto orig_shape = orig_t->shape();
    auto squzeezed_shape = eliminate_shape(orig_shape);
    auto t = orig_t;
    t->shape() = aipubt::TensorShape(squzeezed_shape);
    do
    {
        t = shrink_tensor_data(t);
    } while (t && t->shape().dim() > 0);
    return t;
}

static bool criteria(std::vector<aipubt::NodePtr> &nodes)
{
    auto pool_node = nodes[0];
    auto constant_node = nodes[1];
    auto eltwise_node = nodes[2];

    if (constant_node->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    std::string method = eltwise_node->params()["method"];
    std::string with_activation = eltwise_node->params()["with_activation"];
    if (with_activation != "NONE")
    {
        return false;
    }

    // now we only support method == add, and todo: support method == mul
    if (method != "ADD")
    {
        return false;
    }

    // when method==add, limit the constant has only one data, and data == 0, zp==0
    auto constant_tensor = constant_node->constants()["weights"];
    std::vector<int32_t> constant_zp = constant_tensor->quantization().offsets();
    if (std::all_of(constant_zp.begin(), constant_zp.end(), [](int32_t z) { return z == 0 ? false : true; }))
    {
        return false;
    }
    auto t = shrink_tensor(constant_tensor);
    if (!t || t->shape().dim() > 0)
    {
        return false;
    }
    else
    {
        auto val = _get_data_map[t->type()](t);
        if (val != 0)
        {
            return false;
        }
    }
    return true;
}

void fuse_constant_eltwise_to_leakyrelu(aipubt::GraphPtr g)
{
    // now defaultly disable this  pass
    return;
    std::vector<aipubt::GraphMatcherPattern> patterns = {
        {{{"0", aipubt::OpType_Pooling}, {"1", aipubt::OpType_Constant}, {"2", aipubt::OpType_Eltwise}}, {{"0", "2"}, {"1", "2"}}},
        {{{"1", aipubt::OpType_Pooling}, {"0", aipubt::OpType_Constant}, {"2", aipubt::OpType_Eltwise}}, {{"0", "2"}, {"1", "2"}}},
        // {{ {"0", aipubt::OpType_Constant}, {"1", aipubt::OpType_Eltwise}}, {{"1", "2"}}},
    };
    // bool is_changed = false;
    for (auto &pattern : patterns)
    {
        aipubt::GraphMatcher matcher(pattern);
        matcher.set_mode(aipubt::GraphMatcher::MatchMode::RelaxedMode);
        auto matched = matcher.match(g);
        while (matched.size())
        {
            bool is_merged = false;
            for (auto &&m : matched)
            {
                bool ret = criteria(m);
                if (!ret)
                {
                    continue;
                }
                auto eltwise_node = m[2];
                // eltwise_add_scales = [doscale, scale0, scale1]
                int32_t pool_scale_index_in_eltwise_scales = 1;
                auto pool_node = m[0];
                auto constant_node = m[1];
                if (eltwise_node->inputs()[0]->node()->type() == aipubt::OpType_Constant)
                {
                    pool_scale_index_in_eltwise_scales = 2;
                    constant_node = m[0];
                    pool_node = m[1];
                }
                auto scales = eltwise_node->params()["scale_value"];
                int32_t shift = eltwise_node->params()["shift_value"];
                int32_t pool_branch_scale = scales[pool_scale_index_in_eltwise_scales];
                int32_t eltwise_do_scale = scales[0];
                float total_scale = eltwise_do_scale * pool_branch_scale; // total_scale max bits 30
                while (total_scale > (1 << 15) - 1)
                {
                    total_scale /= 2;
                    shift--;
                }

                aipubt::NodePtr leakyrelu_node = aipubt::Node::create(m[2]->name(), aipubt::OpType_Activation);
                leakyrelu_node->params()["method"] = "LEAKYRELU";
                leakyrelu_node->params()["negative_slope_type"] = aipubt::Tensor::INT8;
                leakyrelu_node->params()["negative_slope_value"] = 1;
                leakyrelu_node->params()["negative_slope_shift"] = 0;
                leakyrelu_node->params()["scale_type"] = dtype_utils::range_to_dtype(0, total_scale, false);
                leakyrelu_node->params()["scale_value"] = static_cast<int32_t>(round(total_scale));
                leakyrelu_node->params()["shift_type"] = aipubt::Tensor::INT8;
                leakyrelu_node->params()["shift_value"] = shift;
                leakyrelu_node->add_input(pool_node->outputs()[0]);
                leakyrelu_node->add_output(eltwise_node->outputs()[0]);
                if (eltwise_node->attrs().count("layer_id"))
                {
                    leakyrelu_node->set_attr("layer_id", eltwise_node->get_attr("layer_id"));
                }

                constant_node->clear_outputs();
                eltwise_node->clear_outputs();
                eltwise_node->clear_inputs();

                g->remove_node(constant_node);
                g->remove_node(eltwise_node);
                g->add_node(leakyrelu_node);
                // g->serialize();
                // is_changed = true;
                is_merged = true;
            }
            if (!is_merged)
            {
                break;
            }
            matched = matcher.match(g);
        }
    }
    // return is_changed;
}

} // namespace qtlibn
