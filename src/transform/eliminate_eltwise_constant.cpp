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
 * @Author: Xiaolong Zhang
 * @Date: 2022-11-01 13:38:32
 * @Last Modified by: Xialong Zhang
 * @Last Modified time: 2022-07-05 10:40:45
 */

#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include "quantization_transform/utils/quantize_utils.h"

namespace aipubt
{

template <typename T>
int32_t get_data(TensorPtr t)
{
    int32_t val = t->data<T>()[0];
    return val;
}

static std::map<Tensor::Dtype, std::function<int32_t(TensorPtr t)>> get_data_table = {
    {Tensor::UINT8, get_data<uint8_t>},
    {Tensor::INT8, get_data<int8_t>},
    {Tensor::UINT16, get_data<uint16_t>},
    {Tensor::INT16, get_data<int16_t>},
    {Tensor::UINT32, get_data<uint32_t>},
    {Tensor::INT32, get_data<int32_t>},
};

static inline bool criteria(std::vector<NodePtr>& m)
{
    NodePtr cst  = m[0];
    NodePtr eltwise_op = m[2];

    if (cst->outputs()[0]->is_io_tensor() || cst->children().size() > 1)
    {
        return false;
    }

    if (m[1]->outputs()[0]->quantization().scale() != eltwise_op->outputs()[0]->quantization().scale() ||
        m[1]->outputs()[0]->quantization().offset() != eltwise_op->outputs()[0]->quantization().offset()) {
        return false;
    }

    auto op_type = eltwise_op->type();
    if (op_type == OpType_Add)
    {
        auto const_shape = cst->outputs()[0]->shape();
        if (!(const_shape == TensorShape({1})     ||
            const_shape == TensorShape({1,1})     ||
            const_shape == TensorShape({1,1,1})   ||
            const_shape == TensorShape({1,1,1,1})))
        {
            return false;
        }

        auto const_ptr = cst->constants()["weights"];
        auto const_dtype = const_ptr->type();
        if (!get_data_table.count(const_dtype))
        {
            return false;
        }

        int32_t const_value = get_data_table[const_dtype](const_ptr);
        if (const_value + cst->outputs()[0]->quantization().offset() != 0)
        {
            return false;
        }
    }
    else if (op_type == OpType_Mul)
    {
        return false;
    }

    return true;
}


bool eliminate_eltwise_constant(GraphPtr g)
{
    std::vector<GraphMatcherPattern> patterns = {
        {{{"0", OpType_Constant}, {"1", OpType_Reshape}, {"2", OpType_Add}},{{"0", "2"},{"1", "2"}}},
        {{{"0", OpType_Constant}, {"1", OpType_Round}, {"2", OpType_Add}},{{"0", "2"},{"1", "2"}}},
        {{{"0", OpType_Constant}, {"1", OpType_Reshape}, {"2", OpType_Add}, {"3", OpType_Cast}, {"4", OpType_Cast}},{{"1", "3"},{"0", "4"},{"3", "2"},{"4", "2"}}},
        // {{{"0", OpType_Constant}, {"1", OpType_Reshape}, {"2", OpType_Mul}},{{"0", "2"},{"1", "2"}}},
    };

    bool is_changed = false;
    for (auto &pattern : patterns)
    {
        GraphMatcher matcher(pattern);
        matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
        auto matched = matcher.match(g);
        for (auto &&m : matched)
        {
            if (!criteria(m))
            {
                continue;
            }

            if (m.size() == 3)
            {
                // m[0]: const
                // m[1]: reshape / round
                // m[2]: add
                auto m1_out_tensor   = m[1]->outputs()[0];
                auto add_out_tensor  = m[2]->outputs()[0];
                auto out_node = m[2]->children()[0];
                if (m1_out_tensor->shape() != add_out_tensor->shape())
                {
                    auto fused_out = ops::reshape(m1_out_tensor, add_out_tensor->shape());
                    auto new_reshape = fused_out->node();
                    out_node->replace_input_by(add_out_tensor, fused_out);
                    g->add_node(new_reshape);
                    g->remove_node(m[0]);
                    g->remove_node(m[2]);
                    ADD_LOCATION_BY_NODE(out_node, new_reshape, m[0], m[2]);
                }
                else
                {
                    out_node->replace_input_by(add_out_tensor, m1_out_tensor);
                    g->remove_node(m[0]);
                    g->remove_node(m[2]);
                    ADD_LOCATION_BY_NODE(out_node, m[0], m[2]);
                }

                is_changed = true;
            }
            else if (m.size() == 5)
            {
                // m[0]: const
                // m[1]: reshape
                // m[2]: add
                // m[3]: cast
                // m[4]: cast
                auto inp = m[1]->inputs()[0];
                auto out_node = m[2]->children()[0];
                auto add_out_tensor = m[2]->outputs()[0];
                auto dtype = add_out_tensor->type();

                auto outp = ops::cast(inp, dtype);
                auto cast_node = outp->node();
                g->add_node(cast_node);
                out_node->replace_input_by(add_out_tensor, inp);

                g->remove_node(m[0]);
                g->remove_node(m[1]);
                g->remove_node(m[2]);
                g->remove_node(m[3]);
                g->remove_node(m[4]);
                ADD_LOCATION_BY_NODE(cast_node, m[0], m[1], m[2], m[3], m[4]);

                is_changed = true;
            }
        }
    }

    return is_changed;
}
}; // namespace aipubt