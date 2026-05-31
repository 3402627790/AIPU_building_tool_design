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
 * @Date: 2022-05-04 16:25:43
 * @Last Modified by: Chao Liu
 * @Last Modified time: 2022-05-04 16:25:43
 */
#include "core/core.h"
#include "core/ops/ops.h"
#include "core/compute_tensor.h"
#include "passes.h"
#include <algorithm>
#include <set>


namespace aipubt
{

static void rearrange_w_b(ComputeTensor& ln_weight, ComputeTensor& ln_bias)
{
    /*layernorm weight/bias shape [c0, c1, c2]
    [c0, c1, c2]--->reshape--->[c0, c1*c2]
    [c0, c1*c2]--->transpose--->[c1*c2, c0]
    [c0, c1*c2]--->reshape--->[1, 1, c0*c1*c2]
    */
    auto c0 = (*ln_weight).shape()[0];
    auto c1 = (*ln_weight).shape()[1];
    auto c2 = (*ln_weight).shape()[2];

    ln_weight = ln_weight.reshape(c0, c1*c2);
    ln_weight = ln_weight.permute({1,0});
    ln_weight = ln_weight.reshape(1, 1, c0*c1*c2);

    ln_bias = ln_bias.reshape(c0, c1*c2);
    ln_bias = ln_bias.permute({1,0});
    ln_bias = ln_bias.reshape(1, 1, c0*c1*c2);
}


static inline bool creteria(NodePtr n1, NodePtr n2)
{
    std::vector<int> perm0 = n1->params()["perm"];
    std::vector<int> const_perm={2,3,0,1};
    std::vector<int> axis0 = n2->params()["axis"];
    std::vector<int> const_axis={1,2,3};

    if (perm0 != const_perm)
    {
        return false;
    }
    if (axis0 != const_axis)
    {
        return false;
    }
    if(n2->outputs()[0]->is_io_tensor())
        return false;
    std::vector<int> inp_shape= n1->inputs()[0]->shape();
    if(inp_shape[3] == 1)
        return false;

    return true;
}

bool fuse_transpose_layernorm(GraphPtr g)
{
    GraphMatcherPattern pattern = {
        {{"0", OpType_Transpose}, {"1", OpType_LayerNorm}},
        {{"0", "1"}}};
    GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
    auto matched = matcher.match(g);
    bool is_changed = false;
    while (matched.size())
    {
        bool is_fused = false;
        for (auto &&m : matched)
        {
            if (!creteria(m[0], m[1]))
            {
                continue;
            }
            std::vector<int> inp_shape= m[0]->inputs()[0]->shape();//[256,256,32,2]
            auto input = m[0]->remove_input(0);

            //add transpose node
            auto transpose_node0 = ops::transpose(input, {0, 1, 3, 2})->node();//[256.256,2,32]
            g->add_node(transpose_node0);
            ADD_LOCATION_BY_NODE(transpose_node0, m[0]);

            //add reshape node0
            TensorShape reshape0_shape(1,1,inp_shape[0]*inp_shape[1]*inp_shape[3],inp_shape[2]);
            reshape0_shape.set_layout(DataLayout_NHWC);

            auto reshape_node0 = ops::reshape(transpose_node0->outputs()[0], reshape0_shape)->node();
            g->add_node(reshape_node0);
            ADD_LOCATION_BY_NODE(reshape_node0, m[0]);

            auto tmp = m[1]->children()[0];
            //modify the layer-normalize_node(axis), re-arrange weight & bias
            m[1]->params()["axis"] = std::vector<int>{1,2};
            m[1]->remove_input(0);
            m[1]->add_input(reshape_node0->outputs()[0]);
            auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("layer_norm"), m[1]->inputs()[0]->shape(),m[1]->outputs()[0]->type());
            m[1]->remove_output(0);
            m[1]->add_output(output);

            ComputeTensor ln_weight = m[1]->constants()["weights"];
            ComputeTensor ln_bias = m[1]->constants()["biases"];
            rearrange_w_b(ln_weight,ln_bias);
            m[1]->constants()["weights"] = ln_weight;
            m[1]->constants()["biases"]= ln_bias;

            //add reshape node1
            TensorShape reshape1_shape(inp_shape[0], inp_shape[1], inp_shape[3], inp_shape[2]);
            reshape1_shape.set_layout(DataLayout_NHWC);
            auto reshape_node1 = ops::reshape(m[1]->outputs()[0], reshape1_shape)->node();//{256,256,2,32}
            ADD_LOCATION_BY_NODE(reshape_node1, m[0]);
            g->add_node(reshape_node1);

            //add transpose node1
            auto transpose_node1 = ops::transpose(reshape_node1->outputs()[0],{0,1,3,2})->node();//{256,256,32,2}
            ADD_LOCATION_BY_NODE(transpose_node1, m[0]);
            g->add_node(transpose_node1);

            //add reshape node2
            TensorShape reshape2_shape(1,inp_shape[0],inp_shape[1], inp_shape[2] * inp_shape[3]);
            reshape2_shape.set_layout(DataLayout_NHWC);
            auto reshape_node2 = ops::reshape(transpose_node1->outputs()[0],reshape2_shape)->node();//{1,256,256,64}
            ADD_LOCATION_BY_NODE(reshape_node2, m[0]);
            g->add_node(reshape_node2);

            //add transpose node2(it will be eliminated with the next transpose from orgin graph)
            auto transpose_node2 = ops::transpose(reshape_node2->outputs()[0],{0,3,1,2})->node();//{1,64,256,256}
            g->add_node(transpose_node2);
            ADD_LOCATION_BY_NODE(transpose_node2, m[0]);

            //add reshape node3(it will be eliminated with the next reshape from orgin graph)
            reshape2_shape.set_layout(DataLayout_NHWC);
            TensorShape reshape3_shape(inp_shape[2],inp_shape[3],inp_shape[0],inp_shape[1]);
            auto reshape_node3 = ops::reshape(transpose_node2->outputs()[0], reshape3_shape)->node();//{32,2,256,256}
            ADD_LOCATION_BY_NODE(reshape_node3, m[0]);
            g->add_node(reshape_node3);

            //link reshape_node3 to orgin layer-normalize node's child node
            tmp->remove_input(0);
            tmp->add_input(reshape_node3->outputs()[0]);

            ADD_LOCATION_BY_NODE(transpose_node2, m[0]);
            g->remove_node(m[0]);

            is_changed = true;
            is_fused = true;
        }
        if (!is_fused)
        {
            break;
        }
        matched = matcher.match(g);
        eliminate_reshape_reshape(g);
        eliminate_transpose_transpose(g);
        move_activation(g);
        fuse_conv_transpose(g);
        fuse_transpose_conv(g);
        //split_transpose(g);
        //split_layernorm_channel(g);
    }
    return is_changed;
}

}