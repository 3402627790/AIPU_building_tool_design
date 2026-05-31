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
 * @Date: 2023-08-16 14:15:24
 * @Last Modified by: Chao Liu
 * @Last Modified time: 2023-08-16 14:15:24
 */
#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include <algorithm>
#include <set>
#include <string>
namespace aipubt
{
    enum CONV_H_TILING_TYPE
    {
        CONV_H_TILING_BOTTOM_SLICE = 0,
        CONV_H_TILING_MIDDLE_SLICE = 1,
        CONV_H_TILING_TOP_SLICE = 2,
    };

    static inline bool creteria(const std::vector<NodePtr> &nodes)
    {
        if(nodes[nodes.size()-1]->outputs()[0]->is_io_tensor())
            return false;
        return true;
    }

    static void modify_conv(NodePtr & conv_node, GraphPtr g)
    {
        std::vector<int> conv_shape_in = conv_node->inputs()[0]->shape();
        std::vector<int> conv_shape_out = conv_node->outputs()[0]->shape();

        auto h = (conv_node->inputs()[0]->shape().layout()==DataLayout_NHWC)? conv_shape_in[1] : conv_shape_in[2];
        auto w = (conv_node->inputs()[0]->shape().layout()==DataLayout_NHWC)? conv_shape_in[2] : conv_shape_in[3];

        TensorShape shape4d(conv_shape_out[0],h,w,conv_shape_out[3]); //out->NHWC when output NCHWC32,it wrong, but useless
        shape4d.set_layout(conv_node->outputs()[0]->shape().layout());

        TensorShape shape5d(conv_shape_out[0], conv_shape_out[1],h,w,conv_shape_out[4]); //out->NCHWC32
        shape5d.set_layout(conv_node->outputs()[0]->shape().layout());

        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("conv2d"),
                (conv_node->outputs()[0]->shape().layout()==DataLayout_NHWC)? shape4d: shape5d, conv_node->outputs()[0]->type());
        output->quantization() = conv_node->outputs()[0]->quantization();
        conv_node->remove_output(0);
        conv_node->add_output(output);
        conv_node->params()["pad_top"] = conv_node->params()["pad_left"];
        conv_node->params()["pad_bottom"] = conv_node->params()["pad_right"];
    }

    bool eliminate_dense_crop(GraphPtr g)
    {
        bool is_changed = false;
        uint32_t max_depth_limit = 16;
        const uint32_t min_depth_limit = 3;
        const uint32_t max_node_cnt = 65536;
        bool depth_empty = false;
        if(max_depth_limit <= min_depth_limit)
            return true;
        while(!depth_empty)
        {
            //1.Generate Pattern Set
            std::vector<GraphMatcherPattern> pattern_set;
            for(size_t cur = min_depth_limit; cur < max_depth_limit; cur++)
            {
                std::vector<std::pair<std::string, int>> nodes;
                for(size_t j = 0; j < cur-1; j++)
                {
                    nodes.push_back(std::pair<std::string, int>(std::to_string(j),OpType_Crop));
                }
                nodes.push_back(std::pair<std::string, int>(std::to_string(cur-1),OpType_Concat));
                std::vector<std::pair<std::string, std::string>> edges;
                for(size_t j = 0; j < cur-1; j++)
                {
                    edges.push_back(std::pair<std::string, std::string>(std::to_string(j),std::to_string(cur-1)));
                }
                GraphMatcherPattern _pattern(nodes,edges);
                pattern_set.push_back(_pattern);
            }

            //2.Traverse the Pattern Set
            std::vector<uint32_t> depth_set;
            for(std::vector<GraphMatcherPattern>::const_iterator iter = pattern_set.cbegin(); iter!= pattern_set.cend(); iter++)
            {
                uint32_t depth = 0;
                GraphMatcher matcher(*iter);
                matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);
                auto matched = matcher.match(g);
                for(auto &&m : matched)
                {
                    if(m[m.size()-1]->outputs()[0]->is_io_tensor())
                        continue;
                    depth = m.size();
                }
                if(depth)
                    depth_set.push_back(depth);
            }
            //Nothing Matched, Exit Loop!
            if(depth_set.empty())
            {
                depth_empty = true;
                break;
            }

            //3.Find Max Depth
            std::vector<uint32_t>::iterator biggest = std::max_element(std::begin(depth_set),std::end(depth_set));
            uint32_t max_depth = *biggest;
            uint32_t max_depth_index = std::distance(std::begin(depth_set),biggest);

            //4.Eliminate the Max Depth Crop and Fuse Graph
            GraphMatcherPattern pattern = pattern_set[max_depth_index];
            GraphMatcher matcher(pattern);
            matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);
            auto matched = matcher.match(g);

            while (matched.size())
            {
                bool is_fused = false;
                uint32_t depth = 0;
                for (auto &&m : matched)
                {
                    uint32_t conv_tiling_type = 0;
                    if (!creteria(m))
                    {
                        continue;
                    }
                    depth = m.size();
                    if(depth!= max_depth)
                        continue;
                    if(m[depth-1]->type()!= OpType_Concat)
                        continue;

                    if(m[depth-1]->parents().size()!=depth)
                        continue;
                    auto root_concat = m[depth-1];

                    auto cur_concat = root_concat;
                    std::vector<NodePtr>conv_nodes;
                    std::vector<NodePtr>crop_nodes;
                    std::vector<NodePtr>concat_nodes;
                    bool stop_search = false;

                    for(size_t cur = depth; cur > 1; cur--)
                    {
                        if(false == stop_search)
                        {
                            uint32_t conv_idx = max_node_cnt;
                            std::vector<uint32_t> crop_node_collect;
                            if(cur_concat->parents().size()!= cur)
                                continue;

                            concat_nodes.push_back(cur_concat);
                            //4.1 Find Conv and Crop Node
                            for(size_t i = 0; i < cur_concat->parents().size(); ++i)
                            {
                                if(cur_concat->parents()[i]->type() == OpType_Convolution)
                                    conv_idx = i ;
                                if(cur_concat->parents()[i]->type() == OpType_Crop)
                                    crop_node_collect.push_back(i);
                            }
                            if((crop_node_collect.size()!=(cur-1)) || (conv_idx >= cur_concat->parents().size()))
                                break;

                            //4.2 Collect Conv and Crop Node
                            auto cur_conv_node = cur_concat->parents()[conv_idx];
                            if((cur_conv_node->parents().size()>1)
                                /*|| (cur_conv_node->parents()[0]->type()!= OpType_Concat)*/)
                                break;

                            if((cur_conv_node->params()["pad_bottom"] == 0)&&(cur_conv_node->params()["pad_top"] != 0))
                                conv_tiling_type = CONV_H_TILING_TYPE::CONV_H_TILING_BOTTOM_SLICE;
                            if((cur_conv_node->params()["pad_bottom"] == 0)&&(cur_conv_node->params()["pad_top"] == 0))
                                conv_tiling_type = CONV_H_TILING_TYPE::CONV_H_TILING_MIDDLE_SLICE;
                            if((cur_conv_node->params()["pad_bottom"] != 0)&&(cur_conv_node->params()["pad_top"] == 0))
                                conv_tiling_type = CONV_H_TILING_TYPE::CONV_H_TILING_TOP_SLICE;


                            conv_nodes.push_back(cur_conv_node);

                            for(size_t i = 0; i < crop_node_collect.size(); i++)
                            {
                                crop_nodes.push_back(cur_concat->parents()[crop_node_collect[i]]);
                            }
                            if(cur_concat->parents().size()<=2)
                                stop_search = true;
                            else
                                cur_concat = cur_conv_node->parents()[0];
                        }
                    }

                    if(conv_nodes.empty() || crop_nodes.empty() || concat_nodes.empty())
                        break;
                    auto conv_cnt = conv_nodes.size();
                    auto crop_cnt = (((conv_cnt+1)*conv_cnt))>>1;
                    if((crop_nodes.size()!= crop_cnt) || (concat_nodes.size()!=conv_cnt))
                        break;

                    //4.3 Reverse Nodes(FILO)
                    std::reverse(conv_nodes.begin(),conv_nodes.end());
                    std::reverse(crop_nodes.begin(),crop_nodes.end());
                    std::reverse(concat_nodes.begin(),concat_nodes.end());

                    auto _input = conv_nodes[0]->parents()[0];

                    if(_input->parents().size()>1)
                        continue;

                    uint32_t idx_offset = 0;
                    std::vector<NodePtr> tmp_concat_set;
                    for(size_t cur = 0; cur < std::min(depth-1,(uint32_t)conv_nodes.size()); cur++)
                    {
                        //4.4 Modify Conv Node
                        auto conv_node = conv_nodes[cur];
                        modify_conv(conv_node,g);

                        //4.5 Delete Crop Node
                        for(size_t i = 0; i < (cur+1); i++)
                        {
                            auto crop_node = crop_nodes[idx_offset + i];
                            g->remove_node(crop_node);
                        }
                        idx_offset += (cur+1);

                        //4.6 Delete Old Concat,Add New Concat
                        auto old_concat = concat_nodes[cur];
                        auto axis = old_concat->params()["axis"];
                        std::vector<TensorPtr>inps;
                        inps.push_back(_input->outputs()[0]);
                        for(size_t i = 0; i < (cur+1); i++)
                        {
                            inps.push_back(conv_nodes[i]->outputs()[0]);
                        }
                        auto concat_new = ops::concat(inps, axis)->node();
                        g->add_node(concat_new);
                        if(cur< (std::min(depth-1,(uint32_t)conv_nodes.size())-1))
                        {
                            conv_nodes[cur+1]->remove_input(0);
                            conv_nodes[cur+1]->add_input(concat_new->outputs()[0]);
                        }
                        if(cur!=(depth-2))
                            g->remove_node(old_concat);
                        else
                            tmp_concat_set.push_back(concat_new);

                    }
                    auto last_old_concat = concat_nodes[depth-2];
                    auto last_new_concat = tmp_concat_set[0];

                    //4.7 Add Crop out Node
                    std::vector<int>crop_in_shape = last_new_concat->outputs()[0]->shape();
                    std::vector<int>crop_out_shape = concat_nodes[depth-2]->outputs()[0]->shape();
                    std::vector<std::pair<int32_t, int32_t>> crops;
                    auto N = crop_out_shape[0];
                    auto H_o = (last_old_concat->outputs()[0]->shape().layout()==DataLayout_NHWC)? crop_out_shape[1]:crop_out_shape[2];
                    auto H_i = (last_new_concat->outputs()[0]->shape().layout()==DataLayout_NHWC)? crop_in_shape[1]:crop_in_shape[2];
                    auto W = (last_old_concat->outputs()[0]->shape().layout()==DataLayout_NHWC)? crop_out_shape[2]:crop_out_shape[3];
                    auto C = (last_old_concat->outputs()[0]->shape().layout()==DataLayout_NHWC)? crop_out_shape[3]:crop_out_shape[1]*crop_out_shape[4];
                    if(conv_tiling_type == CONV_H_TILING_TYPE::CONV_H_TILING_MIDDLE_SLICE)
                    {
                        auto H_start = (H_i - H_o)/2;
                        auto H_end = H_start + H_o;
                        crops.push_back({0, N});
                        crops.push_back({H_start, H_end});
                        crops.push_back({0, W});
                        crops.push_back({0,C});
                    }
                    if(conv_tiling_type == CONV_H_TILING_TYPE::CONV_H_TILING_BOTTOM_SLICE)
                    {
                        auto H_start = 0;
                        auto H_end = H_o;
                        crops.push_back({0, N});
                        crops.push_back({H_start, H_end});
                        crops.push_back({0, W});
                        crops.push_back({0,C});
                    }
                     if(conv_tiling_type == CONV_H_TILING_TYPE::CONV_H_TILING_TOP_SLICE)
                    {
                        auto H_start = H_i - H_o;
                        auto H_end = H_i;
                        crops.push_back({0, N});
                        crops.push_back({H_start, H_end});
                        crops.push_back({0, W});
                        crops.push_back({0,C});
                    }
                    auto crop_out = ops::crop(last_new_concat->outputs()[0], crops)->node();
                    g->add_node(crop_out);
                    for(auto child : last_old_concat->children())
                    {
                        child->replace_input_by(last_old_concat->outputs()[0], crop_out->outputs()[0]);
                    }
                    g->remove_node(last_old_concat);

                    is_changed = true;
                    is_fused = true;
                }
                if(!is_fused)
                {
                    break;
                }
                matched = matcher.match(g);
                eliminate_dense_concat(g);
            }
            //Update the max_depth_limit
            max_depth_limit = max_depth;
        }//Traverse the Fused New Graph
        return is_changed;
    }
}