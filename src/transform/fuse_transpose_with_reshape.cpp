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
 * @Author: Neo Wang
 * @Date: 2020-05-26 17:25:56
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-03-24 17:01:54
 */
#include "builder/lowering/lowering.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include <algorithm>
#include <set>
#include <vector>

namespace aipubt
{

static std::vector<int> integer_factorize(int n)
{
    std::vector<int> ret;
    int f = 2;
    while (f * f <= n)
    {
        while (n % f == 0)
        {
            ret.push_back(f);
            n = n / f;
        }
        f += 1;
    }
    if (n > 1)
    {
        ret.push_back(n);
    }
    return ret;
}

static int prod(TensorShape &s)
{
    int p = 1;
    for (auto i : s)
    {
        p *= i;
    }
    return p;
}

bool is_strides_changed(TensorShape &shape0, TensorShape &shape1, std::vector<int> &split)
{
    std::vector<std::vector<int>> a, b;
    for (auto i : shape0)
    {
        a.push_back(integer_factorize(i));
    }
    for (auto i : shape1)
    {
        b.push_back(integer_factorize(i));
    }
    auto p = prod(shape0);
    int s = 1;
    while (a.size() > 0 && b.size() > 0 && s != p)
    {
        int v = 1;
        auto a0 = a[0];
        for (auto i : a0)
        {
            if (contains(b[0], i))
            {
                v *= i;
                s *= i;
                b[0].erase(std::find(b[0].begin(), b[0].end(), i));
                a[0].erase(std::find(a[0].begin(), a[0].end(), i));
            }
        }
        if (v != 1)
        {
            split.push_back(v);
        }
        bool res = false;
        if (a[0].size() == 0)
        {
            a.erase(a.begin());
            res = true;
        }
        if (b[0].size() == 0)
        {
            b.erase(b.begin());
            res = true;
        }
        if (!res)
        {
            return false;
        }
    }
    return true;
}

// split a shape into vector of vector
std::vector<std::vector<int>> split_shape(const std::vector<int> &split, TensorShape &shape)
{
    uint32_t idx = 0;
    std::vector<std::vector<int>> ret;
    for (auto i : shape)
    {
        if (i == 1)
        {
            ret.push_back({1});
            continue;
        }
        int p = 1;
        std::vector<int> spl;
        for (; idx < split.size();)
        {
            p *= split[idx];
            spl.push_back(split[idx]);
            ++idx;
            if (p == i)
            {
                ret.push_back(spl);
                break;
            }
        }
    }
    return ret;
}

// split a shape into vector of vector
std::vector<std::vector<tongue::Expr>> split_ds_shape(const std::vector<int> &split,
                                                      TensorShape &shape)
{
    uint32_t idx = 0;
    std::vector<std::vector<tongue::Expr>> exprs;
    auto symbol = shape.symbol();
    for (int i = 0; i < shape.dim(); ++i)
    {
        int d = shape[i];
        if (d == 1)
        {
            exprs.push_back({symbol[i]});
            continue;
        }
        int p = 1;
        std::vector<int> spl;
        std::vector<tongue::Expr> expr;
        for (; idx < split.size();)
        {
            p *= split[idx];
            spl.push_back(split[idx]);
            ++idx;
            if (p == d)
            {
                for (auto sp : spl)
                {
                    int divisor = p / sp;
                    expr.push_back(symbol[i].floordiv(divisor));
                }
                exprs.push_back(expr);
                break;
            }
        }
    }
    return exprs;
}

bool fuse_transpose_with_reshape(GraphPtr g)
{
    GraphMatcherPattern pattern = {
        {{"0", OpType_Transpose}, {"1", OpType_Transpose}, {"2", OpType_Reshape}},
        {{"0", "2"}, {"2", "1"}}};
    GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
    auto matched = matcher.match(g);
    bool is_changed = false;
    std::set<NodePtr> removed;
    while (matched.size())
    {
        bool is_fused = false;
        for (auto &&m : matched)
        {
            if (removed.count(m[0]) || removed.count(m[1]) || removed.count(m[2]))
            {
                continue;
            }
            std::vector<int> split;
            if (m[0]->outputs()[0]->is_io_tensor() || m[2]->outputs()[0]->is_io_tensor())
            {
                continue;
            }
            if (!is_strides_changed(m[2]->inputs()[0]->shape(), m[2]->outputs()[0]->shape(), split))
            {
                continue;
            }
            // trans+reshape+trans =>  reshape+trans+reshape
            auto trans0 = m[0];
            auto trans1 = m[1];
            auto reshape = m[2];

            auto tr_shape = trans0->outputs()[0]->shape();
            auto tr_shape_split = split_shape(split, tr_shape);
            std::vector<int> perm0 = trans0->params()["perm"];
            std::vector<int> perm1 = trans1->params()["perm"];
            std::vector<int> new_perm0;
            std::vector<std::vector<int>> before_trans, after_reshape;
            before_trans.resize(perm0.size());
            for (uint32_t i = 0; i < perm0.size(); ++i)
            {
                before_trans[perm0[i]] = tr_shape_split[i];
            }

            std::vector<tongue::Expr> new_reshape0_exprs;
            if (trans0->is_dynamic())
            {
                std::vector<std::vector<tongue::Expr>> exprs;
                exprs = split_ds_shape(split, tr_shape);
                std::vector<std::vector<tongue::Expr>> before_trans_exprs;
                before_trans_exprs.resize(perm0.size());
                for (uint32_t i = 0; i < perm0.size(); ++i)
                {
                    before_trans[perm0[i]] = tr_shape_split[i];
                    if (exprs.size() > i)
                        before_trans_exprs[perm0[i]] = exprs[i];
                }
                for (auto &&i : before_trans_exprs)
                {
                    for (auto j : i)
                    {
                        new_reshape0_exprs.push_back(j);
                    }
                }
            }
            int acc = 0;
            std::vector<int> accumulate_size, after_trans_split;
            for (auto i : before_trans)
            {
                acc = i.size() + acc - 1;
                accumulate_size.push_back(acc);
            }
            for (uint32_t i = 0; i < perm0.size(); ++i)
            {
                for (uint32_t j = 0; j < before_trans[perm0[i]].size(); ++j)
                {
                    new_perm0.push_back(perm0[i] + accumulate_size[perm0[i]] + j - before_trans[perm0[i]].size() + 1);
                }
                for (auto j : before_trans[perm0[i]])
                {
                    after_trans_split.push_back(j);
                }
            }
            uint32_t idx = 0;
            for (auto s : reshape->outputs()[0]->shape())
            {
                int prod = 1;
                std::vector<int> temp;
                if (s < after_trans_split[idx])
                {
                    after_reshape.push_back(temp);
                    continue;
                }
                for (; idx < after_trans_split.size(); ++idx)
                {
                    if (prod == s)
                    {
                        if (temp.size() == 0)
                        {
                            temp.push_back(new_perm0[idx]);
                            idx++;
                        }
                        break;
                    }
                    prod *= after_trans_split[idx];
                    temp.push_back(new_perm0[idx]);
                }
                after_reshape.push_back(temp);
            }
            std::vector<int> new_reshape0;
            std::vector<uint32_t> new_perm;
            for (auto &&i : before_trans)
            {
                for (auto j : i)
                {
                    new_reshape0.push_back(j);
                }
            }
            for (uint32_t i = 0; i < perm1.size(); ++i)
            {
                for (auto j : after_reshape[perm1[i]])
                {
                    new_perm.push_back(j);
                }
            }
            // for some dim==1, then perm for this dim is not important
            //  we move it to the back
            if (new_perm.size() != new_reshape0.size())
            {
                for (uint32_t i = 0; i < new_reshape0.size(); ++i)
                {
                    if (!contains(new_perm, i))
                    {
                        new_perm.push_back(i);
                    }
                }
            }
            auto inp = m[0]->remove_input(0);
            auto out = m[1]->remove_output(0);
            TensorShape new_tensorshape0(new_reshape0, inp->shape().layout());
            auto reshape0 = ops::reshape(inp, new_tensorshape0, new_reshape0_exprs);
            auto trans_out = ops::transpose(reshape0, new_perm);
            auto reshape1 = ops::reshape(trans_out, out->shape());
            auto reshape_node = reshape1->node();
            reshape_node->remove_output(0);
            reshape_node->add_output(out);
            ADD_LOCATION_BY_TENSOR(reshape0, m[0], m[1], m[2]);
            ADD_LOCATION_BY_TENSOR(trans_out, m[0], m[1], m[2]);
            ADD_LOCATION_BY_NODE(reshape_node, m[0], m[1], m[2]);

            for (auto node : m)
            {
                g->remove_node(node);
                removed.insert(node);
            }
            is_changed = true;
            is_fused = true;
        }
        if (!is_fused)
        {
            break;
        }
        matched = matcher.match(g);
    }
    return is_changed;
}
}; // namespace aipubt
