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
 * @Author: Samuel Han
 * @Date: 2022-04-11 14:24:01
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-04-22 11:08:59
 */

#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include <algorithm>

namespace aipubt
{

// we can always fuse two slice to single one.
bool fuse_slice(GraphPtr g)
{
    std::vector<GraphMatcherPattern> patterns = {
        {{{"0", OpType_Slice}, {"1", OpType_Slice}}, {{"0", "1"}}},
    };

    bool is_changed = false;
    for (auto &pattern : patterns)
    {
        GraphMatcher matcher(pattern);
        matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);
        auto matched = matcher.match(g);
        while (matched.size())
        {
            for (auto &&m : matched)
            {
                auto slice0 = m[0];
                auto slice1 = m[1];
                std::vector<int> start0 = slice0->params()["begin"];
                std::vector<int> start1 = slice1->params()["begin"];
                std::vector<int> end0 = slice0->params()["end"];
                std::vector<int> end1 = slice1->params()["end"];
                std::vector<int> stride0 = slice0->params()["strides"];
                std::vector<int> stride1 = slice1->params()["strides"];

                std::vector<int> start, end, stride;
                for (uint32_t i = 0; i < start0.size(); ++i)
                {
                    start.push_back(start0[i] + start1[i] * stride0[i]);
                    end.push_back(end0[i] - ((end0[i] - start0[i] + stride0[i] - 1) / stride0[i] - end1[i]) * stride0[i]);
                    stride.push_back(stride0[i] * stride1[i]);
                }
                slice1->params()["begin"] = start;
                slice1->params()["end"] = end;
                slice1->params()["strides"] = stride;
                slice1->remove_input(0);
                slice1->add_input(slice0->inputs()[0]);
                MODIFY_NODE(m[1]);
                if (m[0]->children().size() == 0 && !m[0]->outputs()[0]->is_io_tensor())
                {
                    g->remove_node(m[0]);
                }
                is_changed = true;
            }
            matched = matcher.match(g);
        }
    }
    return is_changed;
}

} // namespace aipubt