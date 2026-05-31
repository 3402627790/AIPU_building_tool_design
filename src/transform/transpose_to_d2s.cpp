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
 * @Author: Barney Zhao
 * @Date: 2022-11-03 10:24:00
 * @Last Modified by: Barney Zhao
 * @Last Modified time: 2022-11-03 11:30:00
 */
#include "builder/lowering/lowering.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include <algorithm>
#include <builder/target.h>

namespace aipubt
{
bool transpose_to_d2s(GraphPtr g)
{
    auto is_dcr = [](std::vector<int> perm, std::vector<int> in_shape) -> bool
    {
        // d2s dcr
        //   perm=[0,2,1,3,4]
        //   perm=[0,1,3,2,4,5]
        if (perm.size() == 5)
        {
            std::for_each(perm.begin(), perm.end(), [](int &i)
                          { i++; });
            perm.insert(perm.begin(), 0);
            in_shape.insert(in_shape.begin(), 1);
        }
        if (perm.size() == 6)
        {
            int block_size_y = in_shape[3];
            int block_size_x = in_shape[4];
            std::vector<int> gold = {0, 1, 3, 2, 4, 5};
            return (std::equal(perm.begin(), perm.end(), gold.begin()) && block_size_y == block_size_x);
        }
        return false;
    };

    auto is_crd = [](std::vector<int> perm, std::vector<int> in_shape) -> bool
    {
        // d2s crd
        //   perm=[0,3,1,4,2]
        //   perm=[0,1,4,2,5,3]
        if (perm.size() == 5)
        {
            std::for_each(perm.begin(), perm.end(), [](int &i)
                          { i++; });
            perm.insert(perm.begin(), 0);
            in_shape.insert(in_shape.begin(), 1);
        }
        if (perm.size() == 6)
        {
            int block_size_y = in_shape[4];
            int block_size_x = in_shape[5];
            std::vector<int> gold = {0, 1, 4, 2, 5, 3};
            return (std::equal(perm.begin(), perm.end(), gold.begin()) && block_size_y == block_size_x);
        }
        return false;
    };

    auto criterion = [is_dcr, is_crd](NodePtr node) -> bool
    {
        if (node->inputs().empty())
        {
            return false;
        }
        if (OpType_Transpose != node->type())
        {
            return false;
        }

        std::vector<int> input_shape = node->inputs()[0]->shape();
        std::vector<int> perm = node->params()["perm"];
        bool be_dcr = is_dcr(perm, input_shape);
        bool be_crd = is_crd(perm, input_shape);
        bool res = be_dcr || be_crd;

        if (node->is_dynamic())
        {
            auto symbol_i = node->inputs()[0]->shape().symbol();
            auto dim = symbol_i.dim();
            if (be_dcr && (!symbol_i[dim - 2]->is_constant() || !symbol_i[dim - 3]->is_constant()))
            {
                res = false;
            }
            if (be_crd && (!symbol_i[dim - 1]->is_constant() || !symbol_i[dim - 2]->is_constant()))
            {
                res = false;
            }
        }
        return res;
    };

    bool is_changed = false;
    auto m_nodes = g->nodes();
    for (auto &&n : m_nodes)
    {
        if (!criterion(n))
        {
            continue;
        }

        std::vector<int> perm = n->params()["perm"];
        std::vector<int> inp_shape = n->inputs()[0]->shape();
        std::vector<int> out_shape = n->outputs()[0]->shape();
        auto ori_out_shape = n->outputs()[0]->shape();
        int block_size_y = 0;
        int block_size_x = 0;
        int batch = 0;
        int ih = 0;
        int iw = 0;
        int ic = 0;

        ops::DepthToSpaceMode mode = ops::DepthToSpaceMode::DCR;
        auto trans_in_tensor = n->inputs()[0];
        auto trans_out_tensor = n->outputs()[0];
        if (is_crd(perm, inp_shape))
        {
            // x = inp.view(N, H, W, new_channel, block_size_y, block_size_x)
            // x = x.permute(0, 1, 4, 2, 5, 3).contiguous()  # (N, H, bs, W, bs, C//bs^2)
            mode = ops::DepthToSpaceMode::CRD;
            block_size_y = inp_shape[3];
            block_size_x = inp_shape[4];
            batch = 1;
            ih = inp_shape[0];
            iw = inp_shape[1];
            ic = inp_shape[2] * block_size_y * block_size_x;
            if (inp_shape.size() == 6)
            {
                block_size_y = inp_shape[4];
                block_size_x = inp_shape[5];
                batch = inp_shape[0];
                ih = inp_shape[1];
                iw = inp_shape[2];
                ic = inp_shape[3] * block_size_y * block_size_x;
            }
        }
        else
        {
            aipu_assert(is_dcr(perm, inp_shape));
            // x = inp.view(N, H, W, block_size_y, block_size_x, new_channel)  # (N, H, W, bs, bs, C//bs^2)
            // x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (N, H, bs, W, bs, C//bs^2)
            block_size_y = inp_shape[2];
            block_size_x = inp_shape[3];
            batch = 1;
            ih = inp_shape[0];
            iw = inp_shape[1];
            ic = inp_shape[4] * block_size_y * block_size_x;
            if (inp_shape.size() == 6)
            {
                block_size_y = inp_shape[3];
                block_size_x = inp_shape[4];
                batch = inp_shape[0];
                ih = inp_shape[1];
                iw = inp_shape[2];
                ic = inp_shape[5] * block_size_y * block_size_x;
            }
        }
        TensorShape d2s_in_shape(batch, ih, iw, ic);
        std::vector<tongue::Expr> reshape_exprs;
        if (n->is_dynamic())
        {
            auto symbol_i = n->inputs()[0]->shape().symbol();
            reshape_exprs.resize(4);
            if (is_crd(perm, inp_shape))
            {
                reshape_exprs[0] = 1;
                reshape_exprs[1] = symbol_i[0];
                reshape_exprs[2] = symbol_i[1];
                reshape_exprs[3] = symbol_i[2] * block_size_y * block_size_x;
                if (symbol_i.size() == 6)
                {
                    reshape_exprs[0] = symbol_i[0];
                    reshape_exprs[1] = symbol_i[1];
                    reshape_exprs[2] = symbol_i[2];
                    reshape_exprs[3] = symbol_i[3] * block_size_y * block_size_x;
                }
            }
            else
            {
                aipu_assert(is_dcr(perm, inp_shape));
                reshape_exprs[0] = 1;
                reshape_exprs[1] = symbol_i[0];
                reshape_exprs[2] = symbol_i[1];
                reshape_exprs[3] = symbol_i[4] * block_size_y * block_size_x;
                if (symbol_i.size() == 6)
                {
                    reshape_exprs[0] = symbol_i[0];
                    reshape_exprs[1] = symbol_i[1];
                    reshape_exprs[2] = symbol_i[2];
                    reshape_exprs[3] = symbol_i[5] * block_size_y * block_size_x;
                }
            }
        }

        auto input_reshape = ops::reshape(trans_in_tensor, d2s_in_shape, reshape_exprs);
        auto d2s = ops::depth_to_space(input_reshape, block_size_x, mode);
        auto output_reshape_node = ops::reshape(d2s, ori_out_shape)->node();
        output_reshape_node->remove_output(0);
        output_reshape_node->add_output(trans_out_tensor);
        ADD_LOCATION_BY_NODE(input_reshape->node(), n);
        ADD_LOCATION_BY_NODE(d2s->node(), n);
        ADD_LOCATION_BY_NODE(output_reshape_node, n);

        g->remove_node(n);
        is_changed = true;
    }
    return is_changed;
}
}; // namespace aipubt
