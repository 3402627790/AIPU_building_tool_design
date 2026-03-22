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
 * @Author: Lang Hu
 * @Date: 2023-04-24 17:56:07
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2023-04-26 16:19:28
 */
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include "utils/utils.h"

namespace aipubt
{
static std::vector<uint32_t> get_broadcast_shape(std::vector<uint32_t> &s0, std::vector<uint32_t> &s1)
{
    std::vector<uint32_t> ret;
    std::vector<uint32_t> rs0(s0.rbegin(), s0.rend());
    std::vector<uint32_t> rs1(s1.rbegin(), s1.rend());
    uint32_t dim = std::min(rs0.size(), rs1.size());
    for (uint32_t i = 0; i < dim; i++)
    {
        ret.push_back(std::max(rs0[i], rs1[i]));
    }
    if (rs0.size() != rs1.size())
    {
        std::vector<uint32_t> &l = rs0.size() > rs1.size() ? rs0 : rs1;
        for (uint32_t i = dim; i < l.size(); i++)
        {
            ret.push_back(l[i]);
        }
    }
    std::reverse(ret.begin(), ret.end());
    return ret;
}
static GraphPtr get_root_graph(NodePtr n)
{
    auto root_g = n->graph();
    auto sg = std::dynamic_pointer_cast<SubGraph>(root_g);
    if (sg)
    {
        auto sgn = sg->subgraphnode();
        if (sgn)
        {
            root_g = get_root_graph(sgn);
        }
    }
    return root_g;
}
static bool has_data_aware_symbol(const std::vector<Expr> &exprs,
                                  const std::map<std::string, SymbolWrap> &symbols)
{
    for (auto &&expr : exprs)
    {
        for (auto &&var : expr->variables())
        {
            if (symbols.count(var.name()))
            {
                auto &sym = symbols.at(var.name());
                if (sym.type == 1)
                {
                    return true;
                }
            }
        }
    }
    return false;
}
// some dynamic nodes cannot convert to tile+eltwise
// for instance: a mul node with two inputs: [d0,r1] and [d0,1]
// normally, we will tile input1 from [d0,1] to [d0,r1] to match shape
// but for branch input1, the data-aware symbol "r1" may be unknown
static bool check_data_aware_dependency(NodePtr n)
{
    if (!n->is_dynamic())
    {
        return true;
    }

    auto in0_shape = n->inputs()[0]->shape();
    auto in1_shape = n->inputs()[1]->shape();
    if (in0_shape.size() == 1 || in1_shape.size() == 1)
    {
        return true;
    }

    std::vector<uint32_t> s0 = in0_shape;
    std::vector<Expr> ss0 = in0_shape.symbol();
    std::vector<uint32_t> s1 = in1_shape;
    std::vector<Expr> ss1 = in1_shape.symbol();
    auto bs = get_broadcast_shape(s0, s1);

    auto max_size = std::max(s0.size(), s1.size());
    while (s0.size() < max_size)
    {
        s0.insert(s0.begin(), 1);
        ss0.insert(ss0.begin(), 1);
    }
    while (s1.size() < max_size)
    {
        s1.insert(s1.begin(), 1);
        ss1.insert(ss1.begin(), 1);
    }
    auto &symbols = get_root_graph(n)->symbols();
    if ((s0 == bs && s1 != bs) || (s1 == bs && s0 != bs))
    {
        if (s1 == bs)
        {
            std::swap(s0, s1);
            std::swap(ss0, ss1);
        }
        for (size_t i = 0; i < max_size; ++i)
        {
            if (s1[i] == 1 && has_data_aware_symbol({ss0[i]}, symbols))
            {
                return false;
            }
        }
    }

    return true;
}
static int32_t get_broadcast_axis(std::vector<uint32_t> &shape0, std::vector<uint32_t> &shape1)
{
    uint32_t dim0 = shape0.size();
    uint32_t dim1 = shape1.size();
    uint32_t max_dim = std::max(dim0, dim1);
    uint32_t broadcast_axis = -1;
    if (max_dim > dim0)
    {
        shape0.insert(shape0.begin(), max_dim - dim0, 1);
    }
    if (max_dim > dim1)
    {
        shape1.insert(shape1.begin(), max_dim - dim1, 1);
    }
    for (uint32_t i = 0; i < max_dim; i++)
    {
        if (shape0[i] == shape1[i] && shape0[i] != 1)
        {
            broadcast_axis = i;
            break;
        }
    }
    return broadcast_axis;
}
static bool criteria(const Target *target,
                     NodePtr n,
                     std::string &method,
                     std::vector<uint32_t> &bs,
                     bool &just_reshape,
                     std::vector<std::vector<uint32_t>> &rs_params,
                     std::vector<std::vector<Expr>> &rs_ds_params,
                     std::vector<std::vector<uint32_t>> &t_params)
{
    static std::map<OpType, std::string> method_map = {
        {OpType_Add, "ADD"},
        {OpType_Sub, "SUB"},
        {OpType_Mul, "MUL"},
        {OpType_Div, "DIV"},
        {OpType_DivMod, "DIVMOD"},
    };

    if (!method_map.count(n->type()))
    {
        return false;
    }

    if (!check_data_aware_dependency(n))
    {
        return false;
    }

    bool ret = false;
    method = method_map[n->type()];
    auto in0 = n->inputs()[0];
    auto in1 = n->inputs()[1];
    std::vector<uint32_t> s0 = in0->shape();
    std::vector<Expr> ss0 = in0->shape().symbol();
    std::vector<uint32_t> s1 = in1->shape();
    std::vector<Expr> ss1 = in1->shape().symbol();

    // lib support specified DivMod broadcast forms
    // s0: [n,h,w,c] s1: [1,..1,dim[broadcast_axis],1..1]
    // broadcast_axis = -1 means in1 is a scalar or cannot be broadcasted
    if (n->type() == OpType_DivMod)
    {
        auto broadcast_axis = get_broadcast_axis(s0, s1);
        uint32_t res = 1;
        for (int32_t i = 0; i < broadcast_axis; i++)
        {
            res *= s1[i];
        }
        for (uint32_t i = broadcast_axis + 1; i < s1.size(); i++)
        {
            res *= s1[i];
        }
        if (res == 1)
        {
            return false;
        }
    }

    bs = get_broadcast_shape(s0, s1);
    bool multidirect = s0 != bs && s1 != bs;
    std::vector<uint32_t> no_tile(bs.size(), 1);
    if (multidirect)
    {
        auto create_params = [&](const std::vector<uint32_t> &s, const std::vector<Expr> &ss) {
            std::vector<uint32_t> rs_p = s;
            std::vector<Expr> rs_ds_p = ss;
            while (bs.size() > rs_p.size())
            {
                rs_p.insert(rs_p.begin(), 1);
                rs_ds_p.insert(rs_ds_p.begin(), 1);
            }
            std::vector<uint32_t> t_p = bs;
            for (uint32_t i = 0; i < t_p.size(); i++)
            {
                if (t_p[i] == rs_p[i])
                {
                    t_p[i] = 1;
                }
            }
            rs_params.push_back(rs_p);
            rs_ds_params.push_back(rs_ds_p);
            t_params.push_back(t_p);
        };
        create_params(s0, ss0);
        create_params(s1, ss1);
        just_reshape = false;
        ret = true;
    }
    else
    {
        bool eq = s0 == bs;
        std::vector<uint32_t> &l = eq ? s1 : s0;
        std::vector<Expr> &ll = eq ? ss1 : ss0;
        uint32_t max_d = 0, prod = 1;
        for (auto &&d : l)
        {
            max_d = std::max(max_d, d);
            prod *= d;
        }
        bool force_to_convert = false;
        if (check_plugin_option("FORCE_ADD_SUB_MUL_TO_ELTWISE") && n->is_dynamic())
        {
            force_to_convert = true;
        }
        if (target->isainfo().type == ISA_ZYv3 && in0->type() == Tensor::INT32 && in1->type() == Tensor::INT32)
        {
            force_to_convert = true;
        }
        if (max_d == prod &&
            target->isainfo().type != ISA_ZY &&
            !force_to_convert &&
            n->type() != OpType_Div && n->type() != OpType_DivMod)
        {
            // int axis = bs.size() - 1;
            // if (max_d != 1)
            // {
            //     for (uint32_t i = 0; i < bs.size(); i++)
            //     {
            //         if (bs[i] == max_d)
            //         {
            //             axis = i;
            //         }
            //     }
            // }
            // std::vector<uint32_t> tpl(bs.size() - axis, 1);
            // tpl[0] = max_d;
            // if (l != tpl)
            // {
            //     if (l == s0)
            //     {
            //         rs_params = {tpl, s1};
            //     }
            //     else
            //     {
            //         rs_params = {s0, tpl};
            //     }
            //     t_params = {no_tile, no_tile};
            //     just_reshape = true;
            //     ret = true;
            // }
        }
        else
        {
            std::vector<uint32_t> rs_p = l;
            std::vector<Expr> rs_ds_p = ll;
            while (bs.size() > rs_p.size())
            {
                rs_p.insert(rs_p.begin(), 1);
                rs_ds_p.insert(rs_ds_p.begin(), 1);
            }
            std::vector<uint32_t> t_p = bs;
            for (uint32_t i = 0; i < t_p.size(); i++)
            {
                if (t_p[i] == rs_p[i])
                {
                    t_p[i] = 1;
                }
            }
            if (l == s0)
            {
                rs_params = {rs_p, s1};
                rs_ds_params = {rs_ds_p, ss1};
                t_params = {t_p, no_tile};
            }
            else
            {
                rs_params = {s0, rs_p};
                rs_ds_params = {ss0, rs_ds_p};
                t_params = {no_tile, t_p};
            }
            just_reshape = false;
            ret = true;
        }
    }
    return ret;
}
void add_sub_mul_to_eltwise(GraphPtr g)
{
    auto nodes = g->nodes();
    auto target = get_target();
    auto enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);
    for (auto &&n : nodes)
    {
        std::string method;
        std::vector<uint32_t> bs;
        bool just_reshape;
        std::vector<std::vector<uint32_t>> rs_params;
        std::vector<std::vector<Expr>> rs_ds_params;
        std::vector<std::vector<uint32_t>> t_params;
        if (!criteria(target, n, method, bs, just_reshape, rs_params, rs_ds_params, t_params))
        {
            continue;
        }
        if (target->isainfo().type == ISA_ZY)
        {
            INFO() << "Convert all ADD/SUB/MUL to TILE+ELTWISE for Z1 target";
        }
        if (target->isainfo().type == ISA_ZYv3 &&
            n->inputs()[0]->type() == Tensor::INT32 &&
            n->inputs()[1]->type() == Tensor::INT32)
        {
            INFO() << "Convert all INT32 ADD/SUB/MUL to TILE+ELTWISE for X1 target";
        }

        std::vector<TensorPtr> inputs = n->inputs();
        DataLayout in_layout = inputs[0]->shape().layout();
        if (inputs[0]->shape().dim() < inputs[1]->shape().dim())
        {
            in_layout = inputs[1]->shape().layout();
        }
        std::vector<uint32_t> no_tile(bs.size(), 1);
        for (uint32_t i = 0; i < inputs.size(); i++)
        {
            TensorPtr inp = inputs[i];
            std::vector<uint32_t> si = inp->shape();
            if (rs_params[i] != si)
            {
                std::vector<int> shape(rs_params[i].size());
                for (uint32_t j = 0; j < shape.size(); j++)
                {
                    shape[j] = rs_params[i][j];
                }
                auto new_shape = TensorShape(shape, in_layout);
                inp = ops::reshape(inp, new_shape);
                auto reshape_node = inp->node();
                if (enable_ds)
                {
                    reshape_node->params()["ds_shape"] = rs_ds_params[i];
                }
                reshape_node->infer_shape(true);
                g->add_node(reshape_node);
                ADD_LOCATION_BY_NODE(reshape_node, n);
            }
            if (t_params[i] != no_tile)
            {
                inp = ops::tile(inp, t_params[i]);
                auto tile_node = inp->node();
                tile_node->infer_shape(true);
                g->add_node(tile_node);
                ADD_LOCATION_BY_NODE(tile_node, n);
            }
            inputs[i] = inp;
        }

        if (just_reshape)
        {
            // if (inputs[0] != n->inputs()[0])
            // {
            //     n->replace_input_by(n->inputs()[0], inputs[0]);
            // }
            // else
            // {
            //     n->replace_input_by(n->inputs()[1], inputs[1]);
            // }
        }
        else
        {
            if (enable_ds)
            {
                // support constant [n0,h0,w0,c0] -> tile -> dynamic [n,h,w,c]
                bool is_in0_constant = inputs[0]->shape().symbol().is_constant();
                bool is_in1_constant = inputs[1]->shape().symbol().is_constant();
                auto in0_node = inputs[0]->node();
                auto in1_node = inputs[1]->node();
                if ((is_in0_constant && !is_in1_constant && in0_node->type() == OpType_Tile) ||
                    (!is_in0_constant && is_in1_constant && in1_node->type() == OpType_Tile))
                {
                    NodePtr tile_node;
                    std::vector<Expr> ds_repeats;
                    std::vector<Expr> tile_in_shape, tile_out_shape;
                    if (is_in0_constant)
                    {
                        tile_node = in0_node;
                        tile_in_shape = in0_node->inputs()[0]->shape().symbol();
                        tile_out_shape = inputs[1]->shape().symbol();
                    }
                    else
                    {
                        tile_node = in1_node;
                        tile_in_shape = in1_node->inputs()[0]->shape().symbol();
                        tile_out_shape = inputs[0]->shape().symbol();
                    }
                    bool use_ds_tile = true;
                    ds_repeats.resize(tile_in_shape.size());
                    for (size_t i = 0; i < tile_in_shape.size(); ++i)
                    {
                        int in_dim = tile_in_shape[i]->value();
                        if (tile_out_shape[i]->is_constant())
                        {
                            int out_dim = tile_out_shape[i]->value();
                            if (out_dim >= in_dim && out_dim % in_dim == 0)
                            {
                                ds_repeats[i] = out_dim / in_dim;
                            }
                            else
                            {
                                use_ds_tile = false;
                                break;
                            }
                        }
                        else if (in_dim == 1)
                        {
                            ds_repeats[i] = tile_out_shape[i];
                        }
                        else
                        {
                            use_ds_tile = false;
                            break;
                        }
                    }
                    if (use_ds_tile)
                    {
                        auto &symbols = get_root_graph(n)->symbols();
                        if (!has_data_aware_symbol(ds_repeats, symbols))
                        {
                            tile_node->params()["ds_repeats"] = ds_repeats;
                        }
                        tile_node->infer_shape(true);
                    }
                }
            }

            auto elt_optype = (n->type() == OpType_Div || n->type() == OpType_DivMod) ? n->type() : OpType_Eltwise;
            auto elt_node = Node::create(n->name(), elt_optype);
            elt_node->constants() = std::move(n->constants());
            elt_node->params() = std::move(n->params());
            if (elt_optype == OpType_Eltwise)
            {
                elt_node->params()["method"] = method;
                if (!elt_node->params().count("with_activation"))
                {
                    elt_node->params()["with_activation"] = "NONE";
                }
            }

            ADD_LOCATION_BY_NODE(elt_node, n);
            g->replace_node_by(n, elt_node);
            for (uint32_t i = 0; i < inputs.size(); i++)
            {
                if (inputs[i] != elt_node->inputs()[i])
                {
                    elt_node->replace_input_by(elt_node->inputs()[i], inputs[i]);
                }
            }
            elt_node->infer_shape(true);
        }
    }
}
} // namespace aipubt
