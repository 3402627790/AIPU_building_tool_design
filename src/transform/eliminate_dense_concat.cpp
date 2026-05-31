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
 * @Date: 2023-08-24 10:20:32
 * @Last Modified by: Chao Liu
 * @Last Modified time: 2023-08-24 10:20:32
 */

#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include <algorithm>
#include <set>

namespace aipubt
{
    static inline bool isContinue(std::vector<uint32_t> & idxs, bool up_order)
    {
        if(up_order == true)
        {
            for(size_t i = 0; i < idxs.size()-1; i++)
                if(idxs[i+1]!=(idxs[i]+1))
                    return false;
        }else
        {
            for(size_t i = 0; i < idxs.size()-1; i++)
                if(idxs[i+1]!=(idxs[i]-1))
                    return false;
        }
        return true;
    }


    static inline bool isPartContinue(std::vector<uint32_t>& idxs, bool up_order, uint32_t & out)
    {
        uint32_t _cnt = 0;
        std::vector<uint32_t>break_set;
        if(up_order == true)
        {
            for(size_t i = 0; i < idxs.size()-1; i++)
                if(idxs[i+1]!=(idxs[i]+1))
                {
                    break_set.push_back(i+1);
                }else
                {
                    _cnt++;
                }
        }else
        {
            for(size_t i = 0; i < idxs.size()-1; i++)
                if(idxs[i+1]!=(idxs[i]-1))
                {
                    break_set.push_back(i+1);
                }else
                {
                    _cnt++;
                }
        }
        if(break_set.size()>1 || (_cnt<2))
            return false;
        else
            out = break_set[0]-1;
        return true;
    }

    static inline bool isContained(NodePtr &large, NodePtr &small, uint32_t &_breakp)
    {
        uint32_t matched_cnt = 0;
        if(large->parents().size()< small->parents().size())
            return false;
        std::vector<uint32_t>idxs;

        /* PartContinue
        large->parents().size()=5 , small->parents().size()=4
        larger::=conv_3/tile/1
        larger::=conv_5/tile/1
        larger::=conv_7/tile/1
        larger::=conv_9/tile/1
        larger::=conv_3/tile/in_crop_pre/0_1
        small::=conv_3/tile/1
        small::=conv_5/tile/1
        small::=conv_7/tile/1
        small::=conv_3/tile/in_crop_pre/0_1
        */
        for(size_t j = 0; j < small->parents().size(); j++)
        {
            for(size_t i = 0; i < large->parents().size(); i++)
            {
                if(small->parents()[j] == large->parents()[i])
                {
                    matched_cnt++;
                    idxs.push_back(i);
                }
            }
        }
        if(matched_cnt != small->parents().size())
            return false;
        if(!isContinue(idxs,true))
        {
            bool t = isPartContinue(idxs, true, _breakp);
            if(!t)
                return false;
        }
        return true;
    }

    static inline bool creteria(const std::vector<NodePtr> &nodes)
    {
        if(nodes.empty())
            return false;
        if(nodes[nodes.size()-1]->outputs()[0]->is_io_tensor())
            return false;
        if(nodes[nodes.size()-1]->parents().size()!= nodes.size())//there is a match bug
            return false;
        if(nodes[nodes.size()-1]->type()!= OpType_Concat)
            return false;
        auto parents_nodes = nodes[nodes.size()-1]->parents();
        for(size_t i = 0; i < parents_nodes.size(); i++)
        {
            auto input_node = parents_nodes[i];
            if(input_node->type() == OpType_Concat)//it will externed later, it may support cascade concat
                return false;
            if(input_node->parents().size()>1)
                return false;
        }
        return true;
    }

    bool eliminate_dense_concat(GraphPtr g)
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
            for (size_t cur = min_depth_limit; cur < max_depth_limit; cur++)
            {
                std::vector<std::pair<std::string, int>> nodes;
                for (size_t j = 0; j < cur-1; j++)
                {
                    nodes.push_back(std::pair<std::string, int>(std::to_string(j),OpType_Convolution));
                }
                nodes.push_back(std::pair<std::string, int>(std::to_string(cur-1),OpType_Concat));
                std::vector<std::pair<std::string, std::string>> edges;
                for (size_t j = 0; j < cur-1; j++)
                {
                    edges.push_back(std::pair<std::string, std::string>(std::to_string(j),std::to_string(cur-1)));
                }
                GraphMatcherPattern _pattern(nodes,edges);
                pattern_set.push_back(_pattern);
            }

            //2.Traverse the Pattern set
            std::vector<uint32_t> depth_set;
            for(std::vector<GraphMatcherPattern>::const_iterator iter = pattern_set.cbegin(); iter!= pattern_set.cend(); iter++)
            {
                uint32_t depth = 0;
                GraphMatcher matcher(*iter);
                matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);
                auto matched = matcher.match(g);
                for (auto &&m : matched)
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

            //3.Find Max Depth Concat
            std::vector<uint32_t>::iterator biggest = std::max_element(std::begin(depth_set),std::end(depth_set));
            uint32_t max_depth = *biggest;
            uint32_t max_depth_index = std::distance(std::begin(depth_set),biggest);

            //4.Eliminate Max Depth Concat
            GraphMatcherPattern pattern = pattern_set[max_depth_index];
            GraphMatcher matcher(pattern);
            matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);
            auto matched = matcher.match(g);

            while(matched.size())
            {
                bool is_fused = false;
                for(auto &&m : matched)
                {
                    if (!creteria(m))
                    {
                        continue;
                    }
                    auto cur_concat = m[m.size()-1];
                    uint32_t last_concat_idx = max_node_cnt;
                    for(uint32_t i = 0; i < cur_concat->parents().size(); ++i)
                    {
                        if((cur_concat->parents()[i]->parents()[0]->type()== OpType_Concat) &&
                            (cur_concat->parents()[i]->parents()[0]->parents().size()== (m.size()-1)))
                        {
                            last_concat_idx = i;
                        }
                    }//fiter that(conv1, conv2, conv3,+ concat,that from tiling)
                    if(last_concat_idx >= cur_concat->parents().size())
                        continue;
                    auto inner_any_op = cur_concat->parents()[last_concat_idx];

                    if((inner_any_op->parents().size()>1) || (inner_any_op->parents()[0]->type()!= OpType_Concat))
                        continue;
                    auto last_concat = inner_any_op->parents()[0];

                    //5.1 Check dense connect
                    uint32_t ref_break_point = 0 ;
                    if(!isContained(cur_concat,last_concat,ref_break_point))
                        continue;
                    if(!ref_break_point)
                    {
                        //5.2 Remove cur multi-concat, Add new conncat with 2 nodes.
                        std::vector<TensorPtr>inps;
                        inps.push_back(last_concat->outputs()[0]);
                        inps.push_back(inner_any_op->outputs()[0]);
                        auto axis = cur_concat->params()["axis"];
                        auto new_conncat = ops::concat(inps, axis)->node();
                        g->add_node(new_conncat);
                        for(auto child : cur_concat->children())
                        {
                            child->replace_input_by(cur_concat->outputs()[0], new_conncat->outputs()[0]);
                        }
                        g->remove_node(cur_concat);
                        is_changed = true;
                        is_fused = true;
                    }else
                    {
                        //need to support part-contained
                        /* PartContinue
                            large->parents().size()=5 , small->parents().size()=4
                            larger[conv_3/tile/1, conv_5/tile/1, conv_7/tile/1, conv_9/tile/1, conv_3/tile/in_crop_pre/0_1]
                            small [conv_3/tile/1, conv_5/tile/1, conv_7/tile/1, conv_3/tile/in_crop_pre/0_1]
                        */
                        //5.2 Remove cur multi-concat, add new concat with 3 nodes, PartConcat1, PartConcat2, inner_any_op
                        #ifdef ENABLE_PART_CONTAINED_OPT
                            auto axis = last_concat->params()["axis"];
                            std::vector<TensorPtr>part_inps;
                            for (size_t i = 0; i < std::min(ref_break_point+1, (uint32_t)last_concat->parents().size()); i++)
                            {
                                part_inps.push_back(last_concat->parents()[i]->outputs()[0]);
                            }
                            auto new_part_conncat = ops::concat(part_inps, axis)->node();
                            g->add_node(new_part_conncat);
                            std::vector<TensorPtr>inps;
                            inps.push_back(new_part_conncat->outputs()[0]);
                            inps.push_back(inner_any_op->outputs()[0]);
                            inps.push_back(last_concat->parents()[last_concat->parents().size()-1]->outputs()[0]);
                            auto new_conncat = ops::concat(inps, axis)->node();
                            g->add_node(new_conncat);
                            for(auto child : cur_concat->children())
                            {
                                child->replace_input_by(cur_concat->outputs()[0], new_conncat->outputs()[0]);
                            }
                            g->remove_node(cur_concat);
                        #endif
                    }
                }
                if (!is_fused)
                {
                    break;
                }
                matched = matcher.match(g);
            }
            //Update the Max Depth Limit
            max_depth_limit = max_depth;
        }//Traverse the Fused New Graph
        return is_changed;
    }
}
