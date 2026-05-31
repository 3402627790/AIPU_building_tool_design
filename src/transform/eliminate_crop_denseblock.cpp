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
#include "passes.h"
#include <algorithm>
#include <set>
namespace aipubt
{
    static inline bool creteria(const std::vector<NodePtr> &nodes)
    {
        if(nodes[4]->outputs()[0]->is_io_tensor())
            return false;
        if(nodes[4]->parents().size()!=5)
            return false;
        return true;
    }

    static void modify_conv(NodePtr & conv_node,GraphPtr g)
    {
        std::vector<int> conv_shape_in = conv_node->inputs()[0]->shape();
        std::vector<int> conv_shape_out = conv_node->outputs()[0]->shape();

        auto h = (conv_node->inputs()[0]->shape().layout()==DataLayout_NHWC)? conv_shape_in[1] : conv_shape_in[2];
        auto w = (conv_node->inputs()[0]->shape().layout()==DataLayout_NHWC)? conv_shape_in[2] : conv_shape_in[3];

        TensorShape shape4(conv_shape_out[0],h,w,conv_shape_out[3]); //out->NHWC when output NCHWC32,it wrong, but useless
        shape4.set_layout(conv_node->outputs()[0]->shape().layout());

        TensorShape shape5(conv_shape_out[0], conv_shape_out[1],h,w,conv_shape_out[4]); //out->NCHWC32
        shape5.set_layout(conv_node->outputs()[0]->shape().layout());

        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("conv2d"),
                (conv_node->outputs()[0]->shape().layout()==DataLayout_NHWC)? shape4: shape5, conv_node->outputs()[0]->type());
        output->quantization() = conv_node->outputs()[0]->quantization();
        conv_node->remove_output(0);
        conv_node->add_output(output);
        conv_node->params()["pad_top"] = conv_node->params()["pad_left"];
        conv_node->params()["pad_bottom"] = conv_node->params()["pad_right"];
    }

    bool eliminate_crop_denseblock(GraphPtr g)
    {
        /*
        static GraphMatcherPattern pattern = {
            {{"0", OpType_NoOp},{"1", OpType_Crop},{"2", OpType_Crop},{"3", OpType_Crop},{"4", OpType_Crop},
             {"5", OpType_Convolution},{"6", OpType_Crop},{"7", OpType_Crop},{"8", OpType_Concat},{"9", OpType_Convolution},
             {"10", OpType_Crop},{"11", OpType_Crop},{"12", OpType_Crop},{"13", OpType_Concat},{"14", OpType_Convolution},
             {"15", OpType_Crop},{"16", OpType_Crop},{"17", OpType_Concat},{"18", OpType_Convolution},{"19", OpType_Concat}},

            {{"0", "1"},{"0", "2"},{"0", "3"},{"0", "4"},{"0", "7"},
             {"4", "5"},
             {"5", "6"},{"5", "8"},{"5", "12"},{"5", "16"},
             {"8", "9"},
             {"9", "10"},{"9", "13"},{"9", "11"},
             {"13", "14"},
             {"14", "15"},{"14", "17"},
             {"17", "18"},
             {"7", "8"},
             {"12", "13"},{"2", "13"},
             {"11", "17"},{"16", "17"},{"3", "17"},
             {"1", "19"},{"6", "19"},{"10", "19"},{"15", "19"},{"18", "19"}}
        };
        */

        static GraphMatcherPattern pattern = {
            {{"0", OpType_Crop},{"1", OpType_Crop},{"2", OpType_Crop},{"3", OpType_Crop},{"4", OpType_Concat}},
            {{"0", "4"},{"1", "4"},{"2", "4"},{"3", "4"}}
        };

        GraphMatcher matcher(pattern);
        matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);
        auto matched = matcher.match(g);
        bool is_changed = false;
        while (matched.size())
        {
            bool is_fused = false;
            for (auto &&m : matched)
            {
                if (!creteria(m))
                {
                    continue;
                }
                if(m[4]->parents().size()!=5)
                    continue;
                auto _19_concat = m[4];

                uint32_t _18_idx = 65536;
                for (uint32_t i = 0; i < _19_concat->parents().size(); ++i)
                {
                    if(_19_concat->parents()[i]->type() == OpType_Convolution)
                        _18_idx = i ;
                }
                if(_18_idx >= _19_concat->parents().size())
                    continue;
                auto _18_conv = _19_concat->parents()[_18_idx];

                if((_18_conv->parents().size()>1) ||(_18_conv->parents()[0]->type()!= OpType_Concat))
                    continue;
                auto _17_concat = _18_conv->parents()[0];

                if(_17_concat->parents().size()!=4)
                    continue;
                uint32_t _17_crop_node_cnt = 0;
                uint32_t _17_conv_idx = 65536;
                for (uint32_t i = 0; i < _17_concat->parents().size(); ++i)
                {
                    if(_17_concat->parents()[i]->type() == OpType_Convolution)
                        _17_conv_idx = i ;
                    if(_17_concat->parents()[i]->type() == OpType_Crop)
                        _17_crop_node_cnt++;
                }
                if((_17_crop_node_cnt!=3) || (_17_conv_idx>=_17_concat->parents().size()))
                    continue;
                auto _14_conv = _17_concat->parents()[_17_conv_idx];

                if(_14_conv->children().size()!=2)
                    continue;
                uint32_t _15_idx = 65536;
                for (size_t i = 0; i < _14_conv->children().size(); ++i)
                {
                    if(_14_conv->children()[i]->type() == OpType_Crop)
                        _15_idx = i ;
                }
                if(_15_idx>=_14_conv->children().size())
                    continue;
                auto _15_crop = _14_conv->children()[_15_idx];

                if((_14_conv->parents().size()>1) || (_14_conv->parents()[0]->type()!=OpType_Concat))
                    continue;
                auto _13_concat = _14_conv->parents()[0];

                if(_13_concat->parents().size()!=3)
                    continue;
                uint32_t _13_crop_node_cnt = 0;
                uint32_t _13_conv_idx = 65536;
                for (uint32_t i = 0; i < _13_concat->parents().size(); ++i)
                {
                    if(_13_concat->parents()[i]->type()==OpType_Convolution)
                        _13_conv_idx = i ;
                    if(_13_concat->parents()[i]->type()==OpType_Crop)
                        _13_crop_node_cnt++;
                }
                if((_13_crop_node_cnt!=2) || (_13_conv_idx>=_13_concat->parents().size()))
                    continue;
                auto _9_conv = _13_concat->parents()[_13_conv_idx];

                if(_9_conv->children().size()!=3)
                    continue;
                uint32_t _10_idx = 65536;
                uint32_t _11_idx = 65536;
                for (uint32_t i = 0; i < _9_conv->children().size(); ++i)
                {
                    if((_9_conv->children()[i]->type() == OpType_Crop) && (_9_conv->children()[i]->children()[0] == _19_concat))
                        _10_idx = i ;
                    if((_9_conv->children()[i]->type() == OpType_Crop) && (_9_conv->children()[i]->children()[0] == _17_concat))
                        _11_idx = i ;
                }
                if(_10_idx>=_9_conv->children().size())
                    continue;
                auto _10_crop = _9_conv->children()[_10_idx];
                if(_11_idx>=_9_conv->children().size())
                    continue;
                auto _11_crop = _9_conv->children()[_11_idx];

                if((_9_conv->parents().size()>1) || (_9_conv->parents()[0]->type()!=OpType_Concat))
                    continue;
                auto _8_concat = _9_conv->parents()[0];

                if(_8_concat->parents().size()!=2)
                    continue;
                uint32_t _8_crop_node_cnt = 0;
                uint32_t _8_conv_idx = 65536;
                for (uint32_t i = 0; i < _8_concat->parents().size(); ++i)
                {
                    if(_8_concat->parents()[i]->type() == OpType_Convolution)
                        _8_conv_idx = i ;
                    if(_8_concat->parents()[i]->type() == OpType_Crop)
                        _8_crop_node_cnt++;
                }
                if((_8_crop_node_cnt!=1) || (_8_conv_idx>=_8_concat->parents().size()))
                    continue;
                auto _5_conv = _8_concat->parents()[_8_conv_idx];

                if(_5_conv->children().size()!= 4)
                    continue;
                uint32_t _6_idx = 65536;
                uint32_t _12_idx = 65536;
                uint32_t _16_idx = 65536;
                for (uint32_t i = 0; i < _5_conv->children().size(); ++i)
                {
                    if((_5_conv->children()[i]->type() == OpType_Crop) && (_5_conv->children()[i]->children()[0] == _19_concat))
                        _6_idx = i ;
                    if((_5_conv->children()[i]->type() == OpType_Crop) && (_5_conv->children()[i]->children()[0] == _17_concat))
                        _16_idx = i ;
                    if((_5_conv->children()[i]->type() == OpType_Crop) && (_5_conv->children()[i]->children()[0] == _13_concat))
                        _12_idx = i ;
                }
                if(_6_idx>=_5_conv->children().size())
                    continue;
                auto _6_crop = _5_conv->children()[_6_idx];
                if(_16_idx>=_5_conv->children().size())
                    continue;
                auto _16_crop = _5_conv->children()[_16_idx];
                if(_12_idx>=_5_conv->children().size())
                    continue;
                auto _12_crop = _5_conv->children()[_12_idx];

                if((_5_conv->parents().size()>1) ||(_5_conv->parents()[0]->type()!=OpType_Crop))
                    continue;
                auto _4_crop = _5_conv->parents()[0];

                if(_4_crop->parents().size()>1)
                    continue;
                auto _root = _4_crop->parents()[0];

                uint32_t _1_idx = 65536;
                uint32_t _7_idx = 65536;
                uint32_t _2_idx = 65536;
                uint32_t _3_idx = 65536;
                for (uint32_t i = 0; i < _root->children().size(); ++i)
                {
                    if((_root->children()[i]->type() == OpType_Crop) && (_root->children()[i]->children()[0] == _19_concat))
                        _1_idx = i ;
                    if((_root->children()[i]->type() == OpType_Crop) && (_root->children()[i]->children()[0] == _17_concat))
                        _3_idx = i ;
                    if((_root->children()[i]->type() == OpType_Crop) && (_root->children()[i]->children()[0] == _13_concat))
                        _2_idx = i ;
                    if((_root->children()[i]->type() == OpType_Crop) && (_root->children()[i]->children()[0] == _8_concat))
                        _7_idx = i ;
                }
                if(_1_idx>=_root->children().size())
                    continue;
                auto _1_crop = _root->children()[_1_idx];
                if(_3_idx>=_root->children().size())
                    continue;
                auto _3_crop = _root->children()[_3_idx];
                if(_2_idx>=_root->children().size())
                    continue;
                auto _2_crop = _root->children()[_2_idx];
                if(_7_idx>=_root->children().size())
                    continue;
                auto _7_crop = _root->children()[_7_idx];

                /*1.modify convolution, 2.delete useless crop 3.refine the concat*/
                //modify _5_conv
                modify_conv(_5_conv,g);

                //delete _7_crop
                g->remove_node(_7_crop);

                //delete _8_concat,add new concat_8
                auto axis_8 = _8_concat->params()["axis"];
                std::vector<TensorPtr>inps_8;
                inps_8.push_back(_5_conv->outputs()[0]);
                inps_8.push_back(_4_crop->outputs()[0]);
                auto concat_8 = ops::concat(inps_8, axis_8)->node();
                g->add_node(concat_8);
                _9_conv->remove_input(0);
                _9_conv->add_input(concat_8->outputs()[0]);
                g->remove_node(_8_concat);

                //modify _9_conv
                modify_conv(_9_conv,g);

                //delete _2_crop
                g->remove_node(_2_crop);

                //delete _12_crop
                g->remove_node(_12_crop);

                //delete _13_concat,add new concat_13
                auto axis_13 = _13_concat->params()["axis"];
                std::vector<TensorPtr>inps_13;
                inps_13.push_back(_9_conv->outputs()[0]);
                inps_13.push_back(_5_conv->outputs()[0]);
                inps_13.push_back(_4_crop->outputs()[0]);
                auto concat_13 = ops::concat(inps_13, axis_13)->node();
                g->add_node(concat_13);
                _14_conv->remove_input(0);
                _14_conv->add_input(concat_13->outputs()[0]);
                g->remove_node(_13_concat);

                //modify _14_comv
                modify_conv(_14_conv,g);

                //delete _3_crop
                g->remove_node(_3_crop);

                //delete _16_crop
                g->remove_node(_16_crop);

                //delete _11_crop
                g->remove_node(_11_crop);

                //delete _17_concat,add new concat_17
                auto axis_17 = _17_concat->params()["axis"];
                std::vector<TensorPtr>inps_17;
                inps_17.push_back(_14_conv->outputs()[0]);
                inps_17.push_back(_9_conv->outputs()[0]);
                inps_17.push_back(_5_conv->outputs()[0]);
                inps_17.push_back(_4_crop->outputs()[0]);
                auto concat_17 = ops::concat(inps_17, axis_17)->node();
                g->add_node(concat_17);
                _18_conv->remove_input(0);
                _18_conv->add_input(concat_17->outputs()[0]);
                g->remove_node(_17_concat);

                //modify _18_comv
                modify_conv(_18_conv,g);

                 //delete _1_crop
                g->remove_node(_1_crop);

                //delete _6_crop
                g->remove_node(_6_crop);

                //delete _10_crop
                g->remove_node(_10_crop);

                //delete _15_crop
                g->remove_node(_15_crop);

                //delete _19_concat,add new concat_19
                auto axis_19 = _19_concat->params()["axis"];
                std::vector<TensorPtr>inps_19;
                inps_19.push_back(_18_conv->outputs()[0]);
                inps_19.push_back(_14_conv->outputs()[0]);
                inps_19.push_back(_9_conv->outputs()[0]);
                inps_19.push_back(_5_conv->outputs()[0]);
                inps_19.push_back(_4_crop->outputs()[0]);
                auto concat_19 = ops::concat(inps_19, axis_19)->node();
                g->add_node(concat_19);

                //add crop_out node
                std::vector<int>crop_in_shape = concat_19->outputs()[0]->shape();
                std::vector<int>crop_out_shape = _19_concat->outputs()[0]->shape();
                std::vector<std::pair<int32_t, int32_t>> crops;
                auto N = crop_out_shape[0];
                auto H_o = (_19_concat->outputs()[0]->shape().layout()==DataLayout_NHWC)? crop_out_shape[1]:crop_out_shape[2];
                auto H_i = (concat_19->outputs()[0]->shape().layout()==DataLayout_NHWC)? crop_in_shape[1]:crop_in_shape[2];
                auto W = (_19_concat->outputs()[0]->shape().layout()==DataLayout_NHWC)? crop_out_shape[2]:crop_out_shape[3];
                auto C = (_19_concat->outputs()[0]->shape().layout()==DataLayout_NHWC)? crop_out_shape[3]:crop_out_shape[1]*crop_out_shape[4];
                auto h_start = (H_i - H_o)/2;
                auto h_end = h_start + H_o;
                crops.push_back({0, N});
                crops.push_back({h_start, h_end});
                crops.push_back({0, W});
                crops.push_back({0,C});
                auto crop_out = ops::crop(concat_19->outputs()[0], crops)->node();
                g->add_node(crop_out);
                for (auto child : _19_concat->children())
                {
                    child->replace_input_by(_19_concat->outputs()[0], crop_out->outputs()[0]);
                }
                g->remove_node(_19_concat);

                is_changed = true;
                is_fused = true;
            }
            if (!is_fused)
            {
                break;
            }
            matched = matcher.match(g);
            eliminate_concat_denseblock(g);
        }
        return is_changed;
    }
}