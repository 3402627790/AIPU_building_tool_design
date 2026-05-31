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
 * @Author: Chao Liu
 * @Date: 2022-05-26 15:25:43
 * @Last Modified by: Chao Liu
 * @Last Modified time: 2022-05-26 15:25:43
 */
#include "core/core.h"
#include "core/ops/ops.h"
#include "core/compute_tensor.h"
#include "passes.h"
#include <algorithm>
#include <set>
namespace aipubt
{
    static void rearrange_w_cin_dim(ComputeTensor& conv2d_weight,std::vector<int> & reshape_shape)
    {
        /* weight shape[co, kx, ky, ci]
        reshape_shape=[H,W,c1,c2]   ci=c1*c2
        [co,kx,ky,c1*c2]---rehsape--->[co,kx,ky,c1,c2]
        [co,kx,ky,c1,c2]---transpose--->[co,kx,ky,c2,c1]
        [co,kx,ky,c2,c1]---rehsape--->[co,kx,ky,c2*c1]
        */
        auto co = (*conv2d_weight).shape()[0];
        auto kx = (*conv2d_weight).shape()[1];
        auto ky = (*conv2d_weight).shape()[2];

        conv2d_weight = conv2d_weight.reshape(co, kx, ky, reshape_shape[2],reshape_shape[3]);
        conv2d_weight = conv2d_weight.permute({0,1,2,4,3});
        conv2d_weight = conv2d_weight.reshape(co, kx, ky,reshape_shape[3]*reshape_shape[2]);
    }

    static inline bool creteria(const std::vector<NodePtr> &nodes)
    {
        auto trans = nodes[0];//transpose
        auto reshape = nodes[1];//reshape
        auto conv= (nodes.size()==4)? nodes[3]: nodes[2];//convolution
        std::vector<int> in_shape = trans->inputs()[0]->shape();//[256.256,2,32]
        if(in_shape.size() != 4)
        {
            return false;
        }
        std::vector<int> perm0 = trans->params()["perm"];
        std::vector<int> const_perm={0,1,3,2};
        if (perm0 != const_perm)
        {
            return false;
        }
        // if(reshape->outputs()[0]->shape().c()!= (in_shape[2] *in_shape[3]))
        // {
        //     return false;
        // }
        ComputeTensor conv2d_weight = conv->constants()["weights"];
        auto ci = (*conv2d_weight).shape()[3];
        if(ci!=(in_shape[2] *in_shape[3]))
        {
            return false;
        }
        return true;
    }

    bool fuse_transpose_conv(GraphPtr g)
    {
        std::vector<GraphMatcherPattern> patterns = {
            {{{"0", OpType_Transpose}, {"1", OpType_Reshape}, {"2", OpType_LayoutConvertor}, {"3",OpType_Convolution}},
            {{"0", "1"}, {"1", "2"}, {"2", "3"}}},
            {{{"0", OpType_Transpose}, {"1", OpType_Reshape}, {"2",OpType_Convolution}},
            {{"0", "1"}, {"1", "2"}}}
            };
        bool is_changed = false;
        for(auto &pattern : patterns)
        {
            GraphMatcher matcher(pattern);
            matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);
            auto matched = matcher.match(g);
            while (matched.size())
            {
                bool is_fused = false;
                for (auto &&m : matched)
                {
                    if (!creteria(m))
                    {
                        continue;
                    }
                    auto trans = m[0];
                    auto reshape = m[1];
                    auto conv = (m.size()==4)? m[3]: m[2];//convolution
                    std::vector<int> trans_in_shape= trans->inputs()[0]->shape();//[256,256,2,32]
                    std::vector<int> reshape_out_shape =reshape->outputs()[0]->shape();//[1,256,256,64]
                    auto inp = trans->remove_input(0);
                    auto out = reshape->remove_output(0);

                    TensorShape new_reshape_shape(reshape_out_shape[0],reshape_out_shape[1],reshape_out_shape[2],reshape_out_shape[3]);
                    new_reshape_shape.set_layout(DataLayout_NHWC);
                    auto new_reshape = ops::reshape(inp, new_reshape_shape)->node();
                    g->add_node(new_reshape);
                    new_reshape->remove_output(0);
                    new_reshape->add_output(out);
                    ADD_LOCATION_BY_NODE(new_reshape, m[0]);

                    //re-arrange the conv weight
                    ComputeTensor conv_weight = conv->constants()["weights"];
                    rearrange_w_cin_dim(conv_weight,trans_in_shape);
                    conv->constants()["weights"] = conv_weight;
                    g->remove_node(reshape);
                    g->remove_node(trans);
                    is_changed = true;
                    is_fused = true;
                }
                if (!is_fused)
                {
                    break;
                }
                matched = matcher.match(g);
            }
        }
        return is_changed;
    }

}