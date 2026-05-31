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
 * @Author: Yuxian Chen
 * @Date: 2024-02-28 15:41:00
 * @Last Modified by: Yuxian Chen
 * @Last Modified time: 2024-02-28 15:41:00
 */

/*
    This pass is used to convert Reshape+Transpose+Reshape+Conv to Conv.
    The pattern can be expressed by:
        input(shape: [n, h, w, c]) -> Reshape(shape: [n*h*w, c0, c1, c2], c0*c1*c2 is equal to c) \
        -> Transpose(switch pos among c0, c1 and c2, output shape is t0, perm is p0) \
        -> Reshape(shape: [n, h, w, c]) -> Conv(Cin=c, weight, bias)
    It will be converted to y = Conv(input, new_weight, bias), in which new_weight can be gotten by
    doing inverse operations of input to the original weight:
        new_weight = Reshape(Transpose(Reshape(weight, t0'), p0'), weight.shape),
        t0' is the shape of c gotten by t0, p0' is the invert perm of part of p0
 */
#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"

namespace aipubt
{
static inline bool creteria(const std::vector<NodePtr> &nodes, std::vector<int> &weight_shape, std::vector<int> &weight_perm)
{
    // 1) Check the number of children and io tensor for Reshape and Transpose nodes
    for (int i = 0; i < 2; i++)
    {
        if (nodes[i]->children().size() != 1 || nodes[i]->outputs()[0]->is_io_tensor())
        {
            return false;
        }
    }
    auto trans = nodes[0];
    auto post_reshape = nodes[1];
    auto conv = nodes[2];
    std::vector<int> conv_in_shape = post_reshape->outputs()[0]->shape();
    int channels = conv_in_shape[3];
    // 2) Check whether trans+reshape is only operating on channels:
    //    perm is continuous from 0 to the axis where transpose starts(start_tran_axis);
    //    the product of shapes from start_tran_axis to the end is smaller than channels.
    std::vector<int> trans_out_shape = trans->outputs()[0]->shape();
    std::vector<int> perm = trans->params()["perm"];
    std::vector<int> input_perm;
    int shape_prod = 0;
    int start_tran_axis = -1;
    for (int i = 0; i < (int)perm.size(); i++)
    {
        if (shape_prod == 0 && perm[i] != i)
        {
            shape_prod = 1;
            start_tran_axis = i;
            weight_shape.push_back(-1);
            input_perm.push_back(0);
        }
        if (shape_prod >= 1)
        {
            if (trans->outputs()[0]->shape().symbol().dim() > i &&
                !trans->outputs()[0]->shape().symbol()[i]->is_constant())
            {
                return false;
            }
            shape_prod *= trans_out_shape[i];
            weight_shape.push_back(trans_out_shape[i]);
            input_perm.push_back(perm[i] - start_tran_axis + 1);
        }
    }
    if (shape_prod > channels)
    {
        return false;
    }
    for (int i = 0; i < (int)input_perm.size(); i++) // get inverse perm of input_perm
    {
        int index = std::find(input_perm.begin(), input_perm.end(), i) - input_perm.begin();
        weight_perm.push_back(index);
    }
    return true;
}

bool fuse_reshape_transpose_conv(GraphPtr g)
{
    std::vector<GraphMatcherPattern> patterns = {
        {{{"0", OpType_Transpose}, {"1", OpType_Reshape}, {"2", OpType_Convolution}},
         {{"0", "1"}, {"1", "2"}}},
    };
    bool is_changed = false;
    for (auto &pattern : patterns)
    {
        GraphMatcher matcher(pattern);
        matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);
        auto matched = matcher.match(g);
        for (auto &&m : matched)
        {
            std::vector<int> weight_shape;
            std::vector<int> weight_perm;
            if (!creteria(m, weight_shape, weight_perm))
            {
                continue;
            }
            auto trans = m[0];
            auto post_reshape = m[1];
            auto conv = m[2];
            auto inp = trans->inputs()[0];

            // change the input of reshape node before conv to the input of transpose
            post_reshape->remove_input(0);
            post_reshape->add_input(inp, 0);
            MODIFY_NODE(post_reshape);

            // re-arrange conv weights
            ComputeTensor conv_weight = conv->constants()["weights"];
            auto conv_weight_shape = (*conv_weight).shape();
            conv_weight = conv_weight.reshape(weight_shape).permute(weight_perm);
            conv_weight = conv_weight.reshape(conv_weight_shape);
            conv_weight->monopolize_data();
            conv->constants()["weights"] = conv_weight;
            MODIFY_NODE(conv);

            // clean transpose
            g->remove_node(trans, true);
            is_changed = true;
        }
    }
    return is_changed;
}

} // namespace aipubt