//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Wuyang Zhao
 * @Date: 2025-05-19 16:50:53
 */
#include "builder/lowering/lowering.h"
#include <builder/plugin.h>
#include <gcompiler_api.h>
#include <utils/utils.h>

namespace aipubt
{
class InternalLayoutReshapeTrans : public BuilderOpPlugin
{
public:
    const GraphMatcherPattern get_graph_pattern()
    {
        GraphMatcherPattern pattern = {
            {{"n0", OpType_LayoutConvertor}, {"n1", OpType_Reshape}, {"n2", OpType_Transpose}},
            {{"n0", "n1"}, {"n1", "n2"}}};
        return pattern;
    }
    uint32_t get_score()
    {
        return 100;
    }
    bool set_target(const char *target)
    {
        return true;
    };
    bool check_params(BaseNodePtrList nodes)
    {
        NodePtr trans_node = get_real_node(nodes[2]->inputs()[0]->consumers()[0]);
        NodePtr reshape_node = get_real_node(nodes[1]->inputs()[0]->consumers()[0]);
        auto inp = nodes[0]->inputs()[0];
        auto outp = trans_node->outputs()[0];
        auto data_layout = nodes[0]->inputs()[0]->shape().layout();
        if (reshape_node->is_dynamic() ||
            nodes[1]->inputs()[0]->shape().layout() != DataLayout_NHWC ||
            nodes[1]->inputs()[0]->shape().layout() != nodes[1]->outputs()[0]->shape().layout() ||
            inp->shape().layout() != outp->shape().layout() ||
            (data_layout != DataLayout_NCHWC16 && data_layout != DataLayout_NCHWC32))
        {
            return false;
        }

        bool memlayout_unchanged = false;
        int d_layout = (inp->shape().layout() == DataLayout_NCHWC32) ? 32 : 16;
        std::vector<int> in_shape = inp->shape();
        std::vector<int> reshape_out_shape = nodes[1]->outputs()[0]->shape();
        std::vector<int> out_shape = outp->shape();
        std::vector<int> trans_perm = trans_node->params()["perm"];
        if (in_shape[0] == 1 &&
            (trans_perm[0] == 2 && trans_perm[3] == 3) &&
            (in_shape[2] * in_shape[3] == out_shape[2] * out_shape[3]) &&
            (reshape_out_shape[3] % d_layout == 0 && reshape_out_shape[2] * reshape_out_shape[3] == d_layout * in_shape[1]))
        {
            if (reshape_out_shape[0] == 1 || reshape_out_shape[1] == 1)
            {
                memlayout_unchanged = true;
            }
            else if (trans_perm[1] == 0 && trans_perm[2] == 1)
            {
                memlayout_unchanged = true;
            }
        }
        return memlayout_unchanged;
    }
    BuilderParams generate_params(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        return BuilderParams();
    }
    const std::string generate_code_name(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        return "";
    }
    void memory_remapping(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        auto out = sgnode->outputs()[0];
        auto inp = sgnode->inputs()[0];
        out->memory_offset()->relative_offset = 0;
        out->memory_offset()->set_base_offset(inp->memory_offset());
    }
};

REGISTER_PLUGIN(InternalLayoutReshapeTrans, 0x01010000, BUILDERPLUGIN_TYPE | GBPlugin_Internal);
} // namespace aipubt
