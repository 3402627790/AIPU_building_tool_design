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
 * @Date: 2022-05-11 10:25:43
 * @Last Modified by: Chao Liu
 * @Last Modified time: 2022-05-11 10:25:43
 */
#include "core/core.h"
#include "core/ops/ops.h"
#include "core/compute_tensor.h"
#include "passes.h"
#include <algorithm>
#include <set>
namespace aipubt
{

    static void rearrange_wb_cout_dim(ComputeTensor& conv2d_weight, ComputeTensor& conv2d_bias, std::vector<int> & reshape_shape)
    {
        /* weight shape[co, kx, ky, ci]
        [reshape_shape[2]*reshape_shape[3],kx,ky,ci]---rehsape--->[reshape_shape[2],reshape_shape[3],kx,ky,ci]
        [reshape_shape[2],reshape_shape[3],kx,ky,ci]---transpose--->[reshape_shape[3],reshape_shape[2],kx,ky,ci]
        [reshape_shape[3],reshape_shape[2],kx,ky,ci]---rehsape--->[reshape_shape[2]*reshape_shape[3],kx,ky,ci]
        */
        auto kx = (*conv2d_weight).shape()[1];
        auto ky = (*conv2d_weight).shape()[2];
        auto ci = (*conv2d_weight).shape()[3];
        conv2d_weight = conv2d_weight.reshape(reshape_shape[2],reshape_shape[3], kx, ky,ci);
        conv2d_weight = conv2d_weight.permute({1,0,2,3,4});
        conv2d_weight = conv2d_weight.reshape(reshape_shape[3]*reshape_shape[2], kx, ky,ci);
        /* bias shape [co] */
        conv2d_bias = conv2d_bias.reshape(reshape_shape[2],reshape_shape[3]);
        conv2d_bias = conv2d_bias.permute({1,0}).reshape(reshape_shape[2]*reshape_shape[3]);
    }

    static void rearrange_w_cin_dim(ComputeTensor& conv2d_weight)
    {
        /* weight shape[co, kx, ky, ci]
        [co,kx,ky,ci]---rehsape--->[co,kx,ky,ci/32,32]
        [co,kx,ky,c1,c2]---transpose--->[co,kx,ky,32,ci/32]
        [co,kx,ky,c2,c1]---rehsape--->[co,kx,ky,ci]
        */
        auto co = (*conv2d_weight).shape()[0];
        auto kx = (*conv2d_weight).shape()[1];
        auto ky = (*conv2d_weight).shape()[2];
        auto ci = (*conv2d_weight).shape()[3];
        auto t2 = 32;
        auto t1 = ci/t2;

        conv2d_weight = conv2d_weight.reshape(co, kx, ky, t1, t2);
        conv2d_weight = conv2d_weight.permute({0,1,2,4,3});
        conv2d_weight = conv2d_weight.reshape(co, kx, ky,ci);
    }

    static void rearrange_scale_shift(ComputeTensor& conv2d_scale, ComputeTensor& conv2d_shift, std::vector<int> & reshape_shape)
    {
        conv2d_scale = conv2d_scale.reshape(reshape_shape[2],reshape_shape[3]);
        conv2d_scale = conv2d_scale.permute({1,0});
        conv2d_scale = conv2d_scale.reshape(reshape_shape[3]*reshape_shape[2]);

        conv2d_shift = conv2d_shift.reshape(reshape_shape[2],reshape_shape[3]);
        conv2d_shift = conv2d_shift.permute({1,0});
        conv2d_shift = conv2d_shift.reshape(reshape_shape[3]*reshape_shape[2]);
    }

    template<typename T>
    void _arrange(std::vector<T> & nums, unsigned int n)
    {
        std::vector<T>tmp;
        auto m = nums.size()/n;
        for (size_t i = 0; i < m; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                tmp.push_back(nums[j*m+i]);
            }
        }
        std::copy(tmp.begin(), tmp.end(),nums.begin());
    }

    template<typename T>
    void rearrange_per_chn_quant(std::vector<std::vector<T>> & in, std::vector<int> & reshape_shape)
    {
        _arrange(in[0],reshape_shape[2]);
        _arrange(in[1],reshape_shape[2]);
    }

    static inline bool creteria(const std::vector<NodePtr> &nodes)
    {
        /* nodes.size()==5
            conv1
                 eltwise + reshape + transpose
            conv2

            node.size()==4
            conv
                 eltwise + reshape + transpose
            -----

            node.size()==3
            conv + reshape + transpose
        */
        auto reshape = (nodes.size()==5)? (nodes[3]):((nodes.size()==4)? nodes[2]:nodes[1]);
        auto trans = (nodes.size()==5)? (nodes[4]):((nodes.size()==4)? nodes[3]:nodes[2]);
        std::vector<int> n_shape= reshape->outputs()[0]->shape();//[256.256,32,2]
        if(n_shape.size() != 4)
        {
            return false;
        }
        if(trans->outputs()[0]->is_io_tensor())
            return false;
        std::vector<int> perm0 = trans->params()["perm"];
        std::vector<int> const_perm={0,1,3,2};

        if (perm0 != const_perm)
        {
            return false;
        }
        if(nodes.size()==5)
        {
            auto conv1 = nodes[0];
            auto conv2 = nodes[1];
            ComputeTensor conv2d1_weight = conv1->constants()["weights"];
            ComputeTensor conv2d2_weight = conv2->constants()["weights"];
            auto co1 = (*conv2d1_weight).shape()[0];
            auto co2 = (*conv2d2_weight).shape()[0];
            if(conv1->outputs()[0]->is_io_tensor() || conv2->outputs()[0]->is_io_tensor())
                return false;
            if(co1!=(n_shape[2] *n_shape[3]) || co2!=(n_shape[2] *n_shape[3]))
            {
                return false;
            }
            if((conv1->children().size()>1) || (conv2->children().size()>1))
            {
                return false;
            }
        }
        else
        {
            auto conv = nodes[0];//conv
            ComputeTensor conv2d_weight = conv->constants()["weights"];
            auto co = (*conv2d_weight).shape()[0];
            if(co!=(n_shape[2] *n_shape[3]))
            {
                return false;
            }
            if(conv->outputs()[0]->is_io_tensor())
                return false;
            if((conv->children().size()>1) && (nodes.size()==4))
            {
                return false;
            }
        }
        return true;
    }

    bool fuse_conv_transpose(GraphPtr g)
    {
         /* nodes.size()==5
            conv1
                 eltwise + reshape + transpose
            conv2

            node.size()==4
            conv
                 eltwise + reshape + transpose
            -----

            node.size()==3
            conv + reshape + transpose
        */
        std::vector<GraphMatcherPattern> patterns = {
            {{{"0", OpType_Convolution},{"1", OpType_Convolution}, {"2",OpType_Eltwise}, {"3", OpType_Reshape}, {"4",OpType_Transpose}},
            {{"0", "2"}, {"1", "2"}, {"2", "3"},{"3", "4"}}},
            {{{"0", OpType_Convolution}, {"1",OpType_Eltwise}, {"2", OpType_Reshape}, {"3",OpType_Transpose}},
            {{"0", "1"}, {"1", "2"}, {"2", "3"}}},
            {{{"0", OpType_Convolution}, {"1", OpType_Reshape}, {"2",OpType_Transpose}},
            {{"0", "1"}, {"1", "2"}}},
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
                    //auto conv = m[0];
                    auto reshape = (m.size()==5)? m[3]:((m.size()==4)? m[2]:m[1]);
                    auto trans = (m.size()==5)? m[4]:((m.size()==4)? m[3]:m[2]);
                    std::vector<int> reshape_shape= reshape->outputs()[0]->shape();//[256,256,32,2]
                    //re-arrange the conv weight/bias, quant value
                    if(m.size()==5)
                    {
                        auto conv1 = m[0];
                        auto conv2 = m[1];
                        ComputeTensor conv1_weight = conv1->constants()["weights"];
                        ComputeTensor conv1_bias = conv1->constants()["biases"];
                        rearrange_wb_cout_dim(conv1_weight,conv1_bias,reshape_shape);
                        conv1->constants()["weights"] = conv1_weight;
                        conv1->constants()["biases"] = conv1_bias;

                        if(conv1->params().count("weights_scale") &&
                            conv1->params().count("weights_zp") &&
                            conv1->params().count("biases_scale") &&
                            conv1->params().count("biases_zp"))
                        {
                            std::vector<std::vector<int>> zp;
                            std::vector<int>weights_zp = conv1->params()["weights_zp"];
                            std::vector<int>biases_zp = conv1->params()["biases_zp"];
                            zp.push_back(weights_zp);
                            zp.push_back(biases_zp);
                            rearrange_per_chn_quant(zp,reshape_shape);
                            conv1->params()["weights_zp"] = weights_zp;
                            conv1->params()["biases_zp"] = biases_zp;

                            std::vector<std::vector<float>> scale;
                            std::vector<float>weights_scale = conv1->params()["weights_scale"];
                            std::vector<float>biases_scale = conv1->params()["biases_scale"];
                            scale.push_back(weights_scale);
                            scale.push_back(biases_scale);
                            rearrange_per_chn_quant(scale,reshape_shape);
                            conv1->params()["weights_scale"] = weights_scale;
                            conv1->params()["biases_scale"] = biases_scale;
                        }
                        if((conv1->params().count("scale_size")) && (conv1->params().count("shift_size")))
                        {
                            ComputeTensor conv_scale = conv1->constants()["scale"];
                            ComputeTensor conv_shift = conv1->constants()["shift"];
                            rearrange_scale_shift(conv_scale,conv_shift,reshape_shape);
                        }
                        ComputeTensor conv2_weight = conv2->constants()["weights"];
                        ComputeTensor conv2_bias = conv2->constants()["biases"];
                        rearrange_wb_cout_dim(conv2_weight,conv2_bias,reshape_shape);
                        conv2->constants()["weights"] = conv2_weight;
                        conv2->constants()["biases"] = conv2_bias;

                        if(conv2->params().count("weights_scale") &&
                            conv2->params().count("weights_zp") &&
                            conv2->params().count("biases_scale") &&
                            conv2->params().count("biases_zp"))
                        {
                            std::vector<std::vector<int>> zp;
                            std::vector<int>weights_zp = conv2->params()["weights_zp"];
                            std::vector<int>biases_zp = conv2->params()["biases_zp"];
                            zp.push_back(weights_zp);
                            zp.push_back(biases_zp);
                            rearrange_per_chn_quant(zp,reshape_shape);
                            conv2->params()["weights_zp"] = weights_zp;
                            conv2->params()["biases_zp"] = biases_zp;

                            std::vector<std::vector<float>> scale;
                            std::vector<float>weights_scale = conv2->params()["weights_scale"];
                            std::vector<float>biases_scale = conv2->params()["biases_scale"];
                            scale.push_back(weights_scale);
                            scale.push_back(biases_scale);
                            rearrange_per_chn_quant(scale,reshape_shape);
                            conv2->params()["weights_scale"] = weights_scale;
                            conv2->params()["biases_scale"] = biases_scale;
                        }
                        if((conv2->params().count("scale_size")) && (conv2->params().count("shift_size")))
                        {
                            ComputeTensor conv_scale = conv2->constants()["scale"];
                            ComputeTensor conv_shift = conv2->constants()["shift"];
                            rearrange_scale_shift(conv_scale,conv_shift,reshape_shape);
                        }
                        if(!conv1->outputs()[0]->is_io_tensor() && !conv2->outputs()[0]->is_io_tensor())
                        {
                            if((conv1->parents()[0]->type()!= OpType_Convolution) &&
                                (conv2->parents()[0]->type()== OpType_Convolution) &&
                                (conv2->parents()[0]->children().size()>1))
                            {
                                ComputeTensor conv2_weight = conv2->constants()["weights"];
                                rearrange_w_cin_dim(conv2_weight);
                                conv2->constants()["weights"] = conv2_weight;
                            }

                            if((conv1->parents()[0]->type()== OpType_Convolution) &&
                                (conv2->parents()[0]->type()!= OpType_Convolution) &&
                                (conv1->parents()[0]->children().size()>1))
                            {
                                ComputeTensor conv1_weight = conv1->constants()["weights"];
                                rearrange_w_cin_dim(conv1_weight);
                                conv1->constants()["weights"] = conv1_weight;
                            }
                        }
                    }
                    else
                    {
                        auto conv =m[0];
                        ComputeTensor conv_weight = conv->constants()["weights"];
                        ComputeTensor conv_bias = conv->constants()["biases"];
                        rearrange_wb_cout_dim(conv_weight,conv_bias,reshape_shape);
                        conv->constants()["weights"] = conv_weight;
                        conv->constants()["biases"] = conv_bias;

                        if(conv->params().count("weights_scale") &&
                            conv->params().count("weights_zp") &&
                            conv->params().count("biases_scale") &&
                            conv->params().count("biases_zp"))
                        {
                            std::vector<std::vector<int>> zp;
                            std::vector<int>weights_zp = conv->params()["weights_zp"];
                            std::vector<int>biases_zp = conv->params()["biases_zp"];
                            zp.push_back(weights_zp);
                            zp.push_back(biases_zp);
                            rearrange_per_chn_quant(zp,reshape_shape);
                            conv->params()["weights_zp"] = weights_zp;
                            conv->params()["biases_zp"] = biases_zp;

                            std::vector<std::vector<float>> scale;
                            std::vector<float>weights_scale = conv->params()["weights_scale"];
                            std::vector<float>biases_scale = conv->params()["biases_scale"];
                            scale.push_back(weights_scale);
                            scale.push_back(biases_scale);
                            rearrange_per_chn_quant(scale,reshape_shape);
                            conv->params()["weights_scale"] = weights_scale;
                            conv->params()["biases_scale"] = biases_scale;
                        }
                        if((conv->params().count("scale_size")) && (conv->params().count("shift_size")))
                        {
                            ComputeTensor conv_scale = conv->constants()["scale"];
                            ComputeTensor conv_shift = conv->constants()["shift"];
                            rearrange_scale_shift(conv_scale,conv_shift,reshape_shape);
                        }
                    }
                    auto inp = reshape->remove_input(0);
                    auto out = trans->remove_output(0);
                    TensorShape reshape0_shape(reshape_shape[0],reshape_shape[1],reshape_shape[3],reshape_shape[2]);
                    reshape0_shape.set_layout(DataLayout_NHWC);
                    auto reshape_node0 = ops::reshape(inp, reshape0_shape)->node();
                    g->add_node(reshape_node0);
                    reshape_node0->remove_output(0);
                    reshape_node0->add_output(out);
                    ADD_LOCATION_BY_NODE(reshape_node0, m[0],m[1]);
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