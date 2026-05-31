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
 * @Author: Acheng Pan
 * @Date: 2022-10-19 11:25:43
 * @Last Modified by: Acheng Pan
 * @Last Modified time: 2022-10-19 13:37:25
 */
#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include <algorithm>
#include <set>

namespace aipubt
{
static inline bool check_curr_param(std::map<std::string, NodeParamValue> &curr_params)
{
    bool ret = true;

    ret = ret && curr_params.count("method");
    ret = ret && curr_params.count("pad_left");
    ret = ret && curr_params.count("pad_right");
    ret = ret && curr_params.count("pad_top");
    ret = ret && curr_params.count("pad_bottom");
    ret = ret && curr_params.count("dilation_x");
    ret = ret && curr_params.count("dilation_y");
    if (ret)
    {
        std::string method = curr_params["method"];
        transform(method.begin(), method.end(), method.begin(), ::toupper);

        int pad_left =  curr_params["pad_left"];
        int pad_right =  curr_params["pad_right"] ;
        int pad_top =  curr_params["pad_top"] ;
        int pad_bottom =  curr_params["pad_bottom"] ;

        int dilation_x =  curr_params["dilation_x"] ;
        int dilation_y =  curr_params["dilation_y"] ;

        // avg
        ret = ret && !method.compare("AVG");
        // no pad
        ret = ret && pad_left == 0 && pad_right == 0 && pad_top == 0 && pad_bottom == 0;
        // no dilation
        ret = ret && dilation_x == 1 && dilation_y == 1;
    }

    return ret;
}

static inline bool check_fuse_condition(std::vector<NodePtr> &match_list)
{
    auto match_list_size = match_list.size();
    bool ret = true;
    ret = ret && match_list_size > 1;
    if (ret)
    {
        auto last_match_node = match_list[match_list_size - 1];
        ret = ret && (last_match_node->outputs()[0]->shape().h() == 1);
        ret = ret && (last_match_node->outputs()[0]->shape().w() == 1);
        for (auto &m: match_list)
        {
            int kernel_x = m->params()["kernel_x"];
            int stride_x = m->params()["stride_x"];
            if (last_match_node->name() == m->name() && stride_x == 1)
            {
                ret = ret && kernel_x == last_match_node->inputs()[0]->shape().w();
            }
            else
            {
                ret = ret && kernel_x == stride_x;
            }

            int kernel_y = m->params()["kernel_y"];
            int stride_y = m->params()["stride_y"];
            if (last_match_node->name() == m->name() && stride_y == 1)
            {
                ret = ret && kernel_y == last_match_node->inputs()[0]->shape().h();
            }
            else
            {
                ret = ret && kernel_y == stride_y;
            }
        }
    }

    return ret;
}

static inline void check_child(NodePtr& curr_n, std::vector<NodePtr> &match_list)
{
    bool ret = true;
    // has child
    ret = ret && curr_n->children().size() == 1;
    if (ret)
    {
        NodePtr curr_child = curr_n->children()[0];
        // child node
        ret = ret && curr_child->type() == OpType_Pooling;
        ret = ret && check_curr_param(curr_child->params());
        if (ret)
        {
            match_list.push_back(curr_child);
            check_child(curr_child, match_list);
        }
    }
}

static inline bool creteria(NodePtr n, std::vector<NodePtr> &match_list)
{
    bool ret = true;
    match_list.clear();
    // base node
    ret = ret && n->type() == OpType_Pooling;
    ret = ret && check_curr_param(n->params());
    if (ret)
    {
        match_list.push_back(n);
        check_child(n, match_list); // child node
        // check condition
        ret = ret && check_fuse_condition(match_list);
    }
    return ret;
}

bool fuse_pooling(GraphPtr g)
{
    bool is_changed = false;
    auto nodes = g->nodes();
    std::vector<NodePtr> match_list = {};
    for (auto &&n : nodes)
    {
        if (!creteria(n, match_list))
        {
            continue;
        }

        match_list[0]->params()["kernel_x"] = match_list[0]->inputs()[0]->shape().w();
        match_list[0]->params()["kernel_y"] = match_list[0]->inputs()[0]->shape().h();
        match_list[0]->params()["stride_x"] = 1;
        match_list[0]->params()["stride_y"] = 1;

        auto out = match_list[match_list.size() - 1]->outputs()[0];
        match_list[0]->remove_output(0);
        match_list[0]->add_output(out);
        for (size_t i = 1; i < match_list.size(); ++i)
        {
            g->remove_node(match_list[i]);
        }

        is_changed = true;
    }
    return is_changed;
}
}; // namespace aipubt
