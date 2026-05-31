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
 * @Author: Tom Gao
 * @Date: 2022-03-22 10:25:43
 * @Last Modified by: Aaron Gaoao
 * @Last Modified time: 2022-09-10 21:54:12
 */
#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include <algorithm>
#include <set>

namespace aipubt
{

static inline bool creteria(NodePtr n)
{
    for (size_t i = 0; i < n->inputs().size(); ++i)
    {
        if (n->inputs()[0]->type() != n->inputs()[i]->type())
        {
            return false;
        }
        if (n->inputs()[0]->quantization() != n->inputs()[i]->quantization())
        {
            return false;
        }
    }
    if (n->inputs()[0]->type() != n->outputs()[0]->type())
    {
        return false;
    }
    if (n->inputs()[0]->quantization() != n->outputs()[0]->quantization())
    {
        return false;
    }
    bool is_float = Tensor::is_float(n->outputs()[0]->type());

    auto parents = n->parents();
    std::vector<TensorPtr> new_w;
    for (size_t i = 0; i < parents.size(); ++i)
    {
        if (parents[i]->type() == OpType_Constant)
        {
            if (parents[i]->outputs().size() != 1)
            {
                return false;
            }
            if (parents[i]->children().size() != 1)
            {
                return false;
            }

            auto weights = parents[i]->constants()["weights"];
            auto w = weights;
            do
            {
                DEBUG() << w->shape();
                DEBUG() << w->shape().size();
                DEBUG() << w->shape().dim();
                w = shrink_tensor_data(w);
            } while (w && w->shape().dim() * w->shape().size() != 1 && w->shape().dim() != 0);
            if (w == nullptr)
            {
                return false;
            }
            if (w->shape().size() != 1)
            {
                return false;
            }
            new_w.push_back(w);
        }
    }
    if (new_w.size() < 1)
    {
        return false;
    }
    if (new_w.size() != parents.size() - 1)
    {
        return false;
    }
    for (size_t i = 0; i < new_w.size(); ++i)
    {
        for (size_t j = 0; j < new_w[i]->mem_size(); ++j)
        {
            if (*(new_w[0]->data<char>() + j) != *(new_w[i]->data<char>() + j))
            {
                return false;
            }
        }
    }
    if (!is_float)
    {
        int pad_val = (0 - n->outputs()[0]->quantization().offset());
        if ((new_w[0]->type() == Tensor::INT8 && (*new_w[0]->data<int8_t>() != pad_val)) ||
            (new_w[0]->type() == Tensor::UINT8 && (*new_w[0]->data<uint8_t>() != pad_val)) ||
            (new_w[0]->type() == Tensor::INT16 && (*new_w[0]->data<int16_t>() != pad_val)) ||
            (new_w[0]->type() == Tensor::UINT16 && (*new_w[0]->data<uint16_t>() != pad_val)))
        {
            return false;
        }
    }

    return true;
}

bool fuse_concat_constant_to_pad(GraphPtr g)
{
    GraphMatcherPattern pattern = {
        {{"0", OpType_Concat}},
        {}};
    GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
    auto matched = matcher.match(g);
    bool is_changed = false;
    while (matched.size())
    {
        bool is_merged = false;
        for (auto &&m : matched)
        {
            if (!creteria(m[0]))
            {
                continue;
            }

            size_t body_idx = 0;
            std::vector<NodePtr> constants;
            for (size_t i = 0; i < m[0]->inputs().size(); ++i)
            {
                if (m[0]->inputs()[i]->node()->type() != OpType_Constant)
                {
                    body_idx = i;
                }
                else
                {
                    constants.push_back(m[0]->inputs()[i]->node());
                }
            }

            int axis = m[0]->params()["axis"];
            std::vector<std::pair<uint32_t, uint32_t>> pads(m[0]->outputs()[0]->shape().dim(), {0, 0});
            for (size_t i = 0; i < m[0]->inputs().size(); ++i)
            {
                if (i == body_idx)
                {
                    continue;
                }
                else if (i < body_idx)
                {
                    pads[axis].first += m[0]->inputs()[i]->shape()[axis];
                }
                else
                {
                    pads[axis].second += m[0]->inputs()[i]->shape()[axis];
                }
            }
            std::vector<std::vector<tongue::Expr>> ds_pads = {};
            if (m[0]->is_dynamic())
            {
                ds_pads.assign(m[0]->outputs()[0]->shape().dim(), {0, 0});
                for (size_t i = 0; i < m[0]->inputs().size(); ++i)
                {
                    if (i == body_idx)
                    {
                        continue;
                    }
                    else if (i < body_idx)
                    {
                        ds_pads[axis][0] += m[0]->inputs()[i]->shape().symbol()[axis];
                    }
                    else
                    {
                        ds_pads[axis][1] += m[0]->inputs()[i]->shape().symbol()[axis];
                    }
                }
            }

            auto inp = m[0]->remove_input(body_idx);
            for (size_t i = 0; i < m[0]->inputs().size(); ++i)
            {
                if (i == body_idx)
                {
                    continue;
                }
                m[0]->remove_input(i);
            }
            auto out = m[0]->remove_output(0);

            auto weights = constants[0]->constants()["weights"];
            auto weights_b = ComputeTensor(weights);
            TensorPtr pad_out = nullptr;
            if (Tensor::is_float(weights->type()))
            {
                // ops::pad float will do quantization
                weights_b = weights_b.astype(Tensor::FP32);
                float pad_value = weights_b[0];
                pad_out = ops::pad(inp, pads, ops::PadMode::Constant, pad_value, ds_pads);
            }
            else
            {
                // ops::pad int will not quantization
                weights_b = weights_b.astype(Tensor::INT32);
                int pad_value = weights_b[0];
                pad_out = ops::pad(inp, pads, ops::PadMode::Constant, pad_value, ds_pads);
            }
            auto pad_node = pad_out->node();
            pad_node->remove_output(0);
            pad_node->add_output(out);
            ADD_LOCATION_BY_NODE(pad_node, m[0]);

            g->remove_node(m[0]);
            for (size_t i = 0; i < constants.size(); ++i)
            {
                g->remove_node(constants[i]);
            }
            is_changed = true;
            is_merged = true;
        }
        if (!is_merged)
        {
            break;
        }
        matched = matcher.match(g);
    }
    return is_changed;
}
}; // namespace aipubt
