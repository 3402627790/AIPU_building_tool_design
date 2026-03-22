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
 * @Author: Neo Wang 
 * @Date: 2020-01-29 14:24:38 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-10-15 17:58:49
 */
#include <builder/plugin.h>
#include <gcompiler_api.h>

#include <string>
#include <vector>
using namespace aipubt;
class EmptyPlugin : public BuilderOpPlugin
{
public:
    const GraphMatcherPattern get_graph_pattern()
    {
        INFO() << m_name << "::get_graph_pattern()";
        return std::pair<
            std::vector<std::pair<std::string, int>>,
            std::vector<std::pair<std::string, std::string>>>();
    }
    uint32_t get_score()
    {
        INFO() << m_name << "::get_score()";
        return 0;
    };
    bool set_target(const char *target)
    {
        INFO() << m_name << "::set_target(), target=" << target;
        return true;
    };
    BuilderParams generate_params(BaseNodePtr node, BaseNodePtrList &nodes)
    {
        INFO() << m_name << "::generate_params()";
        return BuilderParams();
    }
    const std::string generate_code_name(BaseNodePtr node, BaseNodePtrList &nodes)
    {
        return "";
    }

protected:
    std::string m_name = "EmptyPlugin";
};

class EmptyPlugin2 : public EmptyPlugin
{
};

REGISTER_PLUGIN(EmptyPlugin, 0x10000000, 0x00);
REGISTER_PLUGIN(EmptyPlugin2, 0x10000002, 0x00);