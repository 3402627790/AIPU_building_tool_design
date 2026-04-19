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
 * @Date: 2026-01-06 16:39:04
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2026-01-06 16:39:04
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>

namespace aipubt
{
namespace ops
{
    TensorPtr empty(const TensorShape &shape,
                    Tensor::Dtype dtype,
                    const std::vector<Expr> &exprs,
                    GraphPtr g)
    {
        if (exprs.size() && shape.dim() != (int)exprs.size())
        {
            ERROR() << "dims of shape and exprs are not equal";
            return nullptr;
        }

        if (!g)
        {
            g = get_default_graph();
        }

        TensorShape out_shape(shape);
        if (out_shape.layout() == DataLayout_Flat)
        {
            set_default_layout(out_shape);
        }

        auto out_name = g->get_valid_tensor_name("empty/output");
        TensorPtr out = std::make_shared<Tensor>(out_name, out_shape, dtype);

        auto node = Node::create(g->get_valid_node_name("empty"), OpType_Empty);
        node->add_output(out);
        if (exprs.size())
        {
            node->params()["ds_output_shape"] = std::vector<std::vector<Expr>>{exprs};
        }
        node->infer_shape(true);

        g->add_node(node);
        quantize(node);
        return out;
    }
} // namespace ops
} // namespace aipubt
