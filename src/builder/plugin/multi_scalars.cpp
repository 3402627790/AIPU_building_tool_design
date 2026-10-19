//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020-2026 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Kyle Huang
 * @Date: 2026-01-08 16:50:53
 */
#include <builder/emitter/code_emitter.h>
#include <builder/plugin.h>
#include <core/node.h>

namespace aipubt
{
class MultiScalars : public BuilderOpPlugin
{
private:
    CodeEmitter m_emitter;

public:
    const GraphMatcherPattern get_graph_pattern()
    {
        GraphMatcherPattern pattern;
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
        return true;
    }
    void setup(BaseNodePtr node, BaseNodePtrList match_list)
    {
        // INFO() << "Fuse scalar setup";
        m_emitter.emit_func(match_list, node, node->name());
        m_emitter.compiler_Driver(node->name());
    }
    BuilderParams generate_params(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        auto ro = BuilderParams();
        ro.push_back(sgnode->get_attr<TensorPtr>(Constants::descriptorbase));
        return ro;
    }
    BuilderParams generate_descriptor(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        auto ro = BuilderParams();
        auto args = m_emitter.args(sgnode->name());
        for (auto arg : args)
        {
            // INFO() << "fuse scalar ro: " << arg->name();
            ro.push_back(arg);
        }
        return ro;
    }
    const std::string generate_code_name(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        return m_emitter.code_name(sgnode->name());
    }
    const std::string generate_kernel(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        return m_emitter.kernel_name(sgnode->name());
    }
};

REGISTER_PLUGIN(MultiScalars, 0x01010000, BUILDERPLUGIN_TYPE);
} // namespace aipubt
