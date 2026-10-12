//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020-2026 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Lang Hu
 * @Date: 2023-10-18 17:19:07
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2023-10-18 17:19:07
 */
#include "builder/lowering/lowering.h"
#include "core/config.h"
#include "core/core.h"
#include "tongue/symbolic.h"
#include "utils/utils.h"

using namespace tongue;

namespace aipubt
{
static void reset_dynamic_status(GraphPtr g)
{
    bool enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);
    if (g->attrs().count(Constants::is_dynamic))
    {
        g->attrs().erase(Constants::is_dynamic);
    }
    g->walk_on_node([enable_ds](NodePtr n) {
        if (!enable_ds)
        {
            if (n->params().count("ds_data_aware_symbol"))
            {
                n->params().erase("ds_data_aware_symbol");
            }
        }
        if (n->attrs().count(Constants::is_dynamic))
        {
            n->erase_attr(Constants::is_dynamic);
        }
        for (auto out : n->outputs())
        {
            std::vector<Expr> s;
            for (int d = 0; d < out->shape().dim(); ++d)
            {
                s.push_back(out->shape()[d]);
            }
            out->shape().symbol() = SymbolShape(s, out->shape().layout(), out->shape().indexing());
        }
        if (n->attrs().count(Constants::output_symbol_shape))
        {
            n->erase_attr(Constants::output_symbol_shape);
        }
        if (n->attrs().count(Constants::output_shape_base))
        {
            n->erase_attr(Constants::output_shape_base);
        }
        if (n->attrs().count(Constants::output_shape_offset))
        {
            n->erase_attr(Constants::output_shape_offset);
        }
        if (n->attrs().count(Constants::output_shape_size))
        {
            n->erase_attr(Constants::output_shape_size);
        }
    });
}
// allocate output shape memory for data-aware nodes
static void alloc_output_shape(GraphPtr g)
{
    uint32_t shape_offset = 0;
    for (auto out : g->get_output_tensors())
    {
        shape_offset += 4 * out->shape().dim();
    }
    g->walk_on_node([&shape_offset](NodePtr n) {
        uint32_t shape_size = 0;
        if (n->type() == OpType_NonZero || n->type() == OpType_CacheUpdate)
        {
            shape_size = 4;
        }
        else if (n->type() == OpType_ConstantOfShape)
        {
            shape_size = 4 * n->inputs()[0]->shape().size();
        }
        else if (n->type() == OpType_Slice && n->inputs().size() == 5)
        {
            shape_size = 4 * n->inputs()[0]->shape().dim();
        }
        else if (n->type() == OpType_ForRangeEnder)
        {
            shape_size = 4 * n->inputs()[2]->shape().size();
        }
        else if (n->type() == OpType_IfEnder)
        {
            for (auto out : n->outputs())
            {
                shape_size += 4 * out->shape().dim();
            }
        }
        else if (n->type() == OpType_TopK && n->inputs().size() == 2)
        {
            shape_size = 4 * n->inputs()[1]->shape().size();
        }
        if (shape_size > 0)
        {
            auto name = n->name() + "/shape_base";
            auto shape = TensorShape({(int)shape_size / 4}, DataLayout_Flat);
            auto shape_base = std::make_shared<Tensor>(name, shape, Tensor::UINT32);
            n->set_attr(Constants::output_shape_base, shape_base);
            n->set_attr(Constants::output_shape_offset, shape_offset);
            n->set_attr(Constants::output_shape_size, shape_size);
            shape_offset += shape_size;
        }
    });
}
void init_symbol_shape(GraphPtr g)
{
    reset_dynamic_status(g);
    g->refresh_symbols();
    std::map<std::string, ExprPtr> symbols;
    std::set<std::string> dims;
    auto enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);
    if (enable_ds)
    {
        dims = BuilderConfig::get<std::set<std::string>>(Constants::dynamic_input_dim);
    }
    uint32_t index = 0;
    for (auto inp : g->get_input_tensors())
    {
        auto node = inp->node();
        auto shape = get_base_shape(inp->shape());
        std::vector<Expr> s;
        if (node->params().count("ds_output_shape"))
        {
            std::vector<std::vector<Expr>> ds_output_shape = node->params()["ds_output_shape"];
            for (int i = 0; i < shape.dim(); ++i)
            {
                auto expr = ds_output_shape[0][i];
                if (expr->op() == ExprImpl::Op::Constant)
                {
                    auto name = expr->symbol().name();
                    if (dims.count(name))
                    {
                        if (!g->symbols().count(name))
                        {
                            g->set_symbol(SymbolWrap(expr, shape[i]));
                        }
                        auto sym = g->symbols().at(name);
                        sym.type = 0; // global
                        sym.dims.push_back(index + i);
                        g->set_symbol(sym);
                        expr->set_value(shape[i]);
                        symbols[name] = expr;
                    }
                    else
                    {
                        g->erase_symbol(name);
                        symbols[name] = Expr(shape[i]);
                    }
                    s.push_back(symbols.at(name));
                }
                else
                {
                    s.push_back(shape[i]);
                }
            }
        }
        else
        {
            for (int i = 0; i < shape.dim(); ++i)
            {
                auto name = "d" + std::to_string(index + i);
                if (dims.count(name))
                {
                    if (!g->symbols().count(name))
                    {
                        g->set_symbol(SymbolWrap(Symbol(name), shape[i]));
                    }
                    auto sym = g->symbols().at(name);
                    sym.type = 0; // global
                    sym.dims.push_back(index + i);
                    g->set_symbol(sym);
                    symbols[name] = sym.expr;
                }
                else
                {
                    g->erase_symbol(name);
                    symbols[name] = Expr(shape[i]);
                }
                s.push_back(symbols.at(name));
            }
        }
        index += shape.dim();
        if (inp->shape().dim() == 0)
        {
            s.clear();
        }
        inp->shape().symbol() = SymbolShape(s, inp->shape().layout(), inp->shape().indexing());
        if (enable_ds)
        {
            node->params()["ds_output_shape"] = std::vector<std::vector<Expr>>{s};
        }
    }
    alloc_output_shape(g);
    if (enable_ds)
    {
        g->walk_on_node([&symbols, g](NodePtr n) {
            if (n->attrs().count(Constants::output_shape_offset) &&
                n->attrs().count(Constants::output_shape_size))
            {
                auto shape_offset = n->get_attr<uint32_t>(Constants::output_shape_offset);
                auto shape_size = n->get_attr<uint32_t>(Constants::output_shape_size);

                std::vector<Expr> da_symbols;
                if (n->params().count("ds_data_aware_symbol"))
                {
                    da_symbols = n->params()["ds_data_aware_symbol"];
                }
                if (da_symbols.empty())
                {
                    for (size_t i = 0; i < shape_size / 4; ++i)
                    {
                        da_symbols.push_back(Symbol("r" + std::to_string(shape_offset / 4 + i)));
                    }
                }
                for (size_t i = 0; i < da_symbols.size(); ++i)
                {
                    auto expr = da_symbols[i];
                    auto name = expr->symbol().name();
                    if (!g->symbols().count(name))
                    {
                        g->set_symbol(SymbolWrap(expr));
                    }
                    auto sym = g->symbols().at(name);
                    sym.type = 1; // data aware
                    sym.dims.push_back(shape_offset / 4 + i);
                    g->set_symbol(sym);
                    symbols[name] = expr;
                }
                n->params()["ds_data_aware_symbol"] = da_symbols;
            }
        });
    }
    g->walk_on_node([&symbols](NodePtr n) { n->replace_symbols(symbols); });
}
void infer_symbol_shape(GraphPtr g)
{
    auto enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);
    g->topological_sort();
    g->walk_on_node([enable_ds](NodePtr n) {
        if (n->type() == OpType_Input ||
            n->type() == OpType_SubGraphInput ||
            SubGraphNode::is_subgraph_node(n))
        {
            return;
        }
        if (enable_ds && (n->is_dynamic() || n->type() == OpType_Empty))
        {
            n->infer_shape(true);
        }
        else
        {
            for (auto out : n->outputs())
            {
                std::vector<Expr> s;
                for (int d = 0; d < out->shape().dim(); ++d)
                {
                    s.push_back(out->shape()[d]);
                }
                out->shape().symbol() = SymbolShape(s, out->shape().layout(), out->shape().indexing());
            }
        }
    });
}
} // namespace aipubt
