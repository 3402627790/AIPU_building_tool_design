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
 * @Date: 2023-06-26 14:20:32
 * @Last Modified by: Chao Liu
 * @Last Modified time: 2023-08-18 11:20:32
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

    static inline bool isContinue(std::vector<uint32_t> & data, bool order)
    {
        if(order == true)
        {
            for(size_t i = 0; i < data.size()-1; i++)
                if(data[i+1]!=(data[i]+1))
                    return false;
        }else
        {
            for(size_t i = 0; i < data.size()-1; i++)
                if(data[i+1]!=(data[i]-1))
                    return false;
        }
        return true;
    }

    static inline bool checkContine(NodePtr &large, NodePtr &small)
    {
        uint32_t cnt = 0;
        std::vector<uint32_t>idxs;
        for (size_t i = 0; i < large->parents().size(); i++)
        {
            for (size_t j = 0; j < small->parents().size(); j++)
            {
                if(small->parents()[j] == large->parents()[i])
                {
                    cnt++;
                    idxs.push_back(i);
                }
            }
        }
        if(cnt != small->parents().size())
            return false;
        if(!isContinue(idxs,true))
            return false;
        return true;
    }

    bool eliminate_concat_denseblock(GraphPtr g)
    {
        std::vector<GraphMatcherPattern> patterns = {
            // {{{"0", OpType_Convolution},{"1", OpType_Convolution},{"2", OpType_Convolution},
            //  {"3", OpType_Convolution},{"4", OpType_Convolution},{"5", OpType_Concat}},
            // {{"0", "5"},{"1", "5"},{"2", "5"},{"3", "5"} ,{"4", "5"}}},

            // {{{"0", OpType_Eltwise},{"1", OpType_Convolution},{"2", OpType_Convolution},
            //   {"3", OpType_Convolution},{"4", OpType_Convolution},{"5", OpType_Concat}},
            // {{"0", "5"},{"1", "5"},{"2", "5"},{"3", "5"} ,{"4", "5"}}},

            {{{"0", OpType_Convolution},{"1", OpType_Convolution},
             {"2", OpType_Convolution},{"3", OpType_Convolution},{"4", OpType_Concat}},
            {{"0", "4"},{"1", "4"},{"2", "4"},{"3", "4"}}},
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
                    if(m[4]->parents().size()!=5)
                        continue;
                    auto _10_concat = m[4];

                    if(_10_concat->children().size()>1)
                        continue;
                    auto _11_conv = _10_concat->children()[0];
                    uint32_t _9_idx = 65536;
                    for (uint32_t i = 0; i < _10_concat->parents().size(); ++i)
                    {
                        if((_10_concat->parents()[i]->type() == OpType_Convolution) &&
                            (_10_concat->parents()[i]->parents()[0]->type()== OpType_Concat) &&
                            (_10_concat->parents()[i]->parents()[0]->parents().size()==4))
                        {
                            _9_idx = i;
                        }
                    }
                    if(_9_idx >= _10_concat->parents().size())
                        continue;
                    auto _9_conv = _10_concat->parents()[_9_idx];

                    if((_9_conv->parents().size()>1) ||(_9_conv->parents()[0]->type()!= OpType_Concat))
                        continue;
                    auto _8_concat = _9_conv->parents()[0];
                    if(_8_concat->parents().size()!=4)
                        continue;

                    uint32_t _7_idx = 65536;
                    for (uint32_t i = 0; i < _8_concat->parents().size(); ++i)
                    {
                        if((_8_concat->parents()[i]->type() == OpType_Convolution) &&
                            ((_8_concat->parents()[i]->parents()[0]->type()== OpType_Concat) ||
                            (_8_concat->parents()[i]->parents()[0]->type()== OpType_InternalConcatenated)) &&
                            (_8_concat->parents()[i]->parents()[0]->parents().size()==3))
                        {
                            _7_idx = i;
                        }
                    }
                    if(_7_idx >= _8_concat->parents().size())
                        continue;
                    auto _7_conv = _8_concat->parents()[_7_idx];

                    if(_7_conv->parents().size()>1)
                        continue;
                    auto _6_concat = _7_conv->parents()[0];
                    if(_6_concat->parents().size()!=3)
                        continue;

                    //check dense connect
                    if(!checkContine(_8_concat,_6_concat))
                        continue;
                    if(!checkContine(_10_concat,_8_concat))
                        continue;

                    //remove conncat8, add new conncat with 2 nodes.
                    std::vector<TensorPtr>inps_8;
                    inps_8.push_back(_7_conv->outputs()[0]);
                    inps_8.push_back(_6_concat->outputs()[0]);
                    auto axis_8 = _8_concat->params()["axis"];
                    auto concat_8 = ops::concat(inps_8, axis_8)->node();
                    g->add_node(concat_8);
                    _9_conv->remove_input(0);
                    _9_conv->add_input(concat_8->outputs()[0]);
                    g->remove_node(_8_concat);

                    //remove conncat10, add new conncat with 2 nodes.
                    std::vector<TensorPtr>inps_10;
                    inps_10.push_back(_9_conv->outputs()[0]);
                    inps_10.push_back(concat_8->outputs()[0]);
                    auto axis_10 = _10_concat->params()["axis"];
                    auto concat_10 = ops::concat(inps_10, axis_10)->node();
                    g->add_node(concat_10);
                    _11_conv->remove_input(0);
                    _11_conv->add_input(concat_10->outputs()[0]);
                    g->remove_node(_10_concat);
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