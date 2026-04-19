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
 * @Date: 2021-04-14 11:04:33
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-08-15 15:51:31
 */
#include "quantization_transform/utils/data_type.h"
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>

namespace aipubt
{
namespace ops
{
    static TensorPtr _input(TensorPtr t,
                            const TensorShape &shape,
                            Tensor::Dtype dtype,
                            const Quantization &quant,
                            GraphPtr g,
                            bool enable_ds,
                            std::vector<Expr> exprs)
    {
        if (!g)
        {
            g = get_default_graph();
        }

        auto _get_start_idx = [](aipubt::GraphPtr g) -> int32_t {
            int32_t start_idx = 0;
            for (auto &n : g->nodes())
            {
                if (n->type() == OpType_Input)
                {
                    start_idx += n->outputs()[0]->shape().dim();
                }
            }
            return start_idx;
        };

        if (!t)
        {
            t = std::make_shared<Tensor>(g->get_valid_tensor_name("input"), shape);
            t->quantization() = quant;
            t->set_type(dtype);
        }

        if (t->shape().layout() == DataLayout_Flat)
        {
            set_default_layout(t->shape());
        }
        auto node = Node::create(g->get_valid_node_name("input"), OpType_Input);
        node->add_output(t);

        // create default symbols if enable_ds and empty exprs
        if (enable_ds && exprs.empty())
        {
            auto start_idx = _get_start_idx(g);
            for (int i = 0; i < shape.dim(); ++i)
            {
                exprs.push_back(Symbol("d" + std::to_string(start_idx + i)));
            }
        }

        if (exprs.size())
        {
            if ((int)exprs.size() != shape.dim())
            {
                ERROR() << "expect " << shape.dim() << " exprs, but got " << exprs.size();
                exprs.clear();
            }

            for (auto expr : exprs)
            {
                if (expr->op() != ExprImpl::Op::Val && expr->op() != ExprImpl::Op::Constant)
                {
                    ERROR() << "only exprs of const and symbol type are supported";
                    exprs.clear();
                    break;
                }
            }
        }

        if (exprs.size())
        {
            t->shape().symbol() = SymbolShape(exprs, t->shape().layout(), t->shape().indexing());
            node->params()["ds_output_shape"] = std::vector<std::vector<Expr>>{exprs};
            for (int i = 0; i < shape.dim(); ++i)
            {
                auto expr = exprs[i];
                if (expr->op() == ExprImpl::Op::Constant)
                {
                    expr->set_value(shape[i]);
                    g->set_symbol(SymbolWrap(expr, shape[i]));
                }
            }
        }

        g->add_node(node);
        quantize(node);
        return t;
    }

    TensorPtr input(const TensorShape &shape,
                    Tensor::Dtype dtype,
                    const Quantization &quant,
                    GraphPtr g,
                    bool enable_ds,
                    std::vector<tongue::Expr> exprs)
    {
        return _input(nullptr, shape, dtype, quant, g, enable_ds, exprs);
    }

    TensorPtr input(TensorPtr t,
                    GraphPtr g,
                    bool enable_ds,
                    std::vector<tongue::Expr> exprs)
    {
        auto q = t->quantization().empty() ? Quantization() : t->quantization();
        return _input(t, t->shape(), t->type(), q, g, enable_ds, exprs);
    }

} // namespace ops
} // namespace aipubt
