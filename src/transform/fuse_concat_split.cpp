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
 * @Author: Samuel Han
 * @Date: 2022-04-21 13:50:59
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-04-24 15:36:12
 */

#include "builder/lowering/lowering.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include <functional>

namespace aipubt
{
using shape_type = std::vector<int32_t>;
static bool _matched(const shape_type &shape1,
                     const shape_type &shape2,
                     std::vector<shape_type> &matched_from_shapes,
                     std::vector<shape_type> &matched_to_shapes)
{
    auto _part_matched = [](int32_t from_v, const shape_type &to_shapes, int32_t &to_begin, int32_t &to_end) -> bool
    {
        int32_t cur_sum = 0;
        int32_t to_size = to_shapes.size();
        for (int32_t j = to_begin; j < to_size; j++)
        {
            cur_sum += to_shapes[j];
            if (from_v == cur_sum)
            {
                to_end = j + 1;
                break;
            }
            else if (from_v < cur_sum)
            {
                return false;
            }
        }
        return true;
    };

    int32_t s1_size = shape1.size();
    int32_t s1_pos = 0;
    int32_t s2_pos = 0;
    std::vector<int32_t> fs;
    std::vector<int32_t> ts;
    for (s1_pos = 0; s1_pos < s1_size;)
    {
        auto s1 = shape1[s1_pos];
        auto s2 = shape2[s2_pos];
        int32_t to_end = 0;
        if (s1 >= s2)
        {
            fs.push_back(s1_pos);
            auto ret = _part_matched(s1, shape2, s2_pos, to_end);
            if (!ret)
            {
                return false;
            }
            for (int32_t m = s2_pos; m < to_end; m++)
            {
                ts.push_back(m);
            }
            s1_pos += 1;
            s2_pos = to_end;
        }
        else if (s1 < s2)
        {
            ts.push_back(s2_pos);
            auto ret = _part_matched(s2, shape1, s1_pos, to_end);
            if (!ret)
            {
                return false;
            }
            for (int32_t m = s1_pos; m < to_end; m++)
            {
                fs.push_back(m);
            }
            s1_pos = to_end;
            s2_pos += 1;
        }
        matched_from_shapes.push_back(fs);
        matched_to_shapes.push_back(ts);
        fs.clear();
        ts.clear();
    }
    return true;
}

static bool _check(std::vector<int32_t> &shape1, std::vector<int32_t> &shape2)
{
    std::vector<shape_type> from_shapes;
    std::vector<shape_type> to_shapes;
    return _matched(shape1, shape2, from_shapes, to_shapes);
}

static bool criteria(NodePtr n1, NodePtr n2)
{
    if (n1->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    for (auto ot : n2->outputs())
    {
        if (ot->is_io_tensor())
        {
            return false;
        }
    }
    int32_t n1_axis = n1->params()["axis"];
    int32_t n2_axis = n2->params()["axis"];
    if (n1_axis != n2_axis)
    {
        return false;
    }
    std::vector<int32_t> n1_shape_in_axis;
    std::vector<int32_t> n2_shape_in_axis = n2->params()["splits"];
    auto n1_it_num = n1->inputs().size();
    for (uint32_t in = 0; in < n1_it_num; in++)
    {
        auto shape = n1->inputs()[in]->shape()[n1_axis];
        n1_shape_in_axis.push_back(shape);
    }
    return _check(n1_shape_in_axis, n2_shape_in_axis);
}

static bool criteria_same(NodePtr n1, NodePtr n2)
{
    if (n1->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    for (auto ot : n2->outputs())
    {
        if (ot->is_io_tensor())
        {
            return false;
        }
    }
    int32_t n1_axis = n1->params()["axis"];
    int32_t n2_axis = n2->params()["axis"];
    if (n1_axis != n2_axis)
    {
        return false;
    }
    std::vector<int32_t> n1_shape_in_axis;
    std::vector<int32_t> n2_shape_in_axis = n2->params()["splits"];
    auto n1_it_num = n1->inputs().size();
    for (uint32_t in = 0; in < n1_it_num; in++)
    {
        auto shape = n1->inputs()[in]->shape()[n1_axis];
        if (shape != n2_shape_in_axis[in])
        {
            return false;
        }
    }
    return true;
}

static bool lower_reshape_one_level(NodePtr node, GraphPtr g)
{
    auto split_node = node->children()[0];

    split_node->remove_input(0);
    split_node->add_input(node->inputs()[0]);

    std::vector<int32_t> splits;
    for (size_t i = 0; i < split_node->outputs().size(); ++i)
    {
        auto split_out = split_node->outputs()[i];
        auto new_shape = node->inputs()[0]->shape();
        new_shape[-1] = 1;
        new_shape[-1] = split_node->outputs()[i]->shape().size() / new_shape.size();
        TensorPtr reshape_in = std::make_shared<Tensor>(node->name() + std::to_string(i), new_shape, split_node->outputs()[i]->type());
        split_node->remove_output(i);
        split_node->add_output(reshape_in, i);
        split_node->params()["axis"] = new_shape.dim() - 1;
        splits.push_back(new_shape[-1]);

        auto reshape_out = ops::reshape(reshape_in, split_node->outputs()[i]->shape());
        auto reshape = reshape_out->node();

        reshape->remove_output(0);
        reshape->add_output(split_out);
        reshape->params()["shape"] = split_out->shape();
        ADD_LOCATION_BY_NODE(reshape, node);
        g->add_node(reshape);
    }
    split_node->params()["splits"] = splits;

    g->remove_node(node);

    return true;
}

static bool criterial_concat(NodePtr n)
{
    if ((n->type() != OpType_Concat && n->type() != OpType_InternalConcatenated) || n->children().size() == 0 || n->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    auto params = get_real_node(n)->params();
    if (params.count("scale_value") && params.count("shift_value"))
    {
        auto scale = params["scale_value"];
        std::vector<int> shift = params["shift_value"];
        for (uint32_t i = 0; i < scale.arr.size(); i++)
        {
            if (((1 << shift[i]) != (int)scale[i]) && (255 != (int)scale[i]))
            {
                return false;
            }
        }
    }

    return true;
}

static bool criterial_reshape_split(NodePtr n)
{
    if (n->type() != OpType_Reshape || n->children().size() == 0 || n->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    if (n->children().size() > 1)
    {
        return false;
    }

    auto child = n->children()[0];
    if (child->type() != OpType_Split)
    {
        return false;
    }

    auto output = n->outputs()[0];
    int axis = child->params()["axis"];
    if (axis != output->shape().dim() - 1) // only support split channel;
    {
        return false;
    }

    auto p = n->parents()[0];
    if (!criteria_same(p, child))
    {
        return false;
    }

    return true;
}

static bool criterial_split(NodePtr n)
{
    if (n->type() != OpType_Split && n->type() != OpType_InternalSpliter)
    {
        return false;
    }

    return true;
}

bool fuse_concat_splits(NodePtr n, GraphPtr g)
{
    bool is_changed = false;
    for (auto child : n->children())
    {
        if (!criterial_split(child))
        {
            continue;
        }
        if (!criteria_same(n, child))
        {
            continue;
        }
        is_changed = true;
        for (size_t i = 0; i < n->inputs().size(); ++i)
        {
            n->inputs()[i]->shape().reset_strides();
            auto consumers = child->outputs()[i]->consumers();
            for (auto c : consumers)
            {
                for (uint32_t j = 0; j < c->inputs().size(); j++)
                {
                    if (c->inputs()[j] == child->outputs()[i])
                    {
                        c->remove_input(j);
                        c->add_input(n->inputs()[i], j);
                    }
                }
            }
        }
        g->remove_node(child);
    }

    if (n->children().size() == 0)
    {
        g->remove_node(n);
    }
    return is_changed;
}

bool fuse_concat_reshape_splits(GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        if (!criterial_concat(n))
        {
            continue;
        }
        bool valid_fuse = true;
        for (auto child : n->children())
        {
            if (!criterial_reshape_split(child))
            {
                valid_fuse = false;
                break;
            }
        }
        if (valid_fuse)
        {
            for (auto child : n->children())
            {
                lower_reshape_one_level(child, g);
            }
            fuse_concat_splits(n, g);
        }
    }
    return true;
}

bool fuse_concat_splits(GraphPtr g)
{
    auto nodes = g->nodes();
    bool is_changed = false;
    for (auto n : nodes)
    {
        if (!criterial_concat(n))
        {
            continue;
        }

        fuse_concat_splits(n, g);

        is_changed = true;
    }
    return is_changed;
}

bool fuse_concat_split(GraphPtr g)
{
    std::vector<GraphMatcherPattern> patterns = {
        {{{"0", OpType_Concat}, {"1", OpType_Split}}, {{"0", "1"}}},
        {{{"0", OpType_InternalConcatenated}, {"1", OpType_InternalSpliter}}, {{"0", "1"}}},
    };

    bool is_changed = false;
    for (auto &pattern : patterns)
    {
        GraphMatcher matcher(pattern);
        matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
        auto matched = matcher.match(g);
        while (matched.size())
        {
            bool is_merged = false;
            for (auto &&m : matched)
            {
                if (!criteria(m[0], m[1]))
                {
                    continue;
                }
                int32_t axis = m[0]->params()["axis"];

                // temp criteria
                if (OpType_InternalConcatenated == m[0]->type())
                {
                    if (axis != 0) // merge split n only
                        continue;
                }
                // check axis of new pattern to be safer

                std::vector<int32_t> m0_shape_in_axis;
                std::vector<int32_t> m1_shape_in_axis = m[1]->params()["splits"];
                auto m0_inp_ts = m[0]->inputs();
                auto m1_out_ts = m[1]->outputs();
                auto m0_it_num = m0_inp_ts.size();
                // auto m1_ot_num = m1_out_ts.size();
                for (uint32_t in = 0; in < m0_it_num; in++)
                {
                    auto shape = m0_inp_ts[in]->shape()[axis];
                    m0_shape_in_axis.push_back(shape);
                }
                std::vector<shape_type> from_shapes;
                std::vector<shape_type> to_shapes;
                _matched(m0_shape_in_axis, m1_shape_in_axis, from_shapes, to_shapes);

                for (uint32_t k = 0; k < from_shapes.size(); k++)
                {
                    auto fs = from_shapes[k];
                    auto ts = to_shapes[k];
                    std::vector<TensorPtr> inps;
                    for (auto idx : fs)
                    {
                        inps.push_back(m0_inp_ts[idx]);
                    }
                    std::vector<TensorPtr> outs;
                    for (auto idx : ts)
                    {
                        outs.push_back(m1_out_ts[idx]);
                    }

                    if (fs.size() > 1)
                    {
                        auto c_out = ops::concat(inps, axis);
                        auto c_node = c_out->node();
                        c_node->clear_outputs();
                        c_node->add_output(outs);
                        g->add_node(c_node);
                        ADD_LOCATION_BY_NODE(c_node, m[0], m[1]);
                    }
                    else if (ts.size() > 1)
                    {
                        auto s_out = ops::split(inps[0], inps.size(), axis);
                        std::vector<int32_t> splits;
                        for (auto it : outs)
                        {
                            splits.push_back(it->shape()[axis]);
                        }
                        auto s_node = s_out[0]->node();
                        s_node->clear_outputs();
                        s_node->add_output(outs);
                        s_node->params()["splits"] = splits;
                        g->add_node(s_node);
                        ADD_LOCATION_BY_NODE(s_node, m[0], m[1]);
                    }
                    else
                    {
                        auto m1_children = m[1]->children();
                        for (uint32_t c = 0; c < m1_children.size(); c++)
                        {
                            auto m1_child = m1_children[c];
                            for (uint32_t ci = 0; ci < m1_child->inputs().size(); ci++)
                            {
                                auto c_i_t = m1_child->inputs()[ci];
                                if (contains<std::vector<TensorPtr>>(outs, c_i_t))
                                {
                                    m1_child->remove_input(c_i_t);
                                    m1_child->add_input(inps[0], ci);
                                }
                            }
                        }
                    }
                }
                m[0]->clear_inputs();
                g->remove_node(m[0]);
                g->remove_node(m[1]);
                is_changed = true;
                is_merged = true;
            }
            if (!is_merged)
            {
                break;
            }
            matched = matcher.match(g);
        }
    }
    return is_changed;
}

}; // namespace aipubt
