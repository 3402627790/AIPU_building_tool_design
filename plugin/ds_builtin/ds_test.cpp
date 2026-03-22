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
 * @Author: Lang Hu
 * @Date: 2024-06-04 14:35:03
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-06-04 14:35:03
 */
#include <builder/plugin.h>
#include <gcompiler_api.h>
#include <utils/utils.h>

namespace aipubt
{

class DSTestPlugin : public BuilderOpPlugin
{
public:
    const GraphMatcherPattern get_graph_pattern()
    {
        return {{{"n1", OpType_Testing}}, {}};
    }
    const std::string get_layer_type_name()
    {
        return "Testing";
    }
    uint32_t get_score()
    {
        return 0;
    }
    bool set_target(const char *target)
    {
        if (utils::startswith(target, "X2"))
        {
            m_target_name = target;
            return true;
        }
        return false;
    }
    bool check_params(BaseNodePtrList nodes)
    {
        return true;
    }
    void setup(BaseNodePtr node, BaseNodePtrList match_list)
    {
        node->set_attr("keeping_layout", true);
        LayoutCombinations valid_in = {
            {DataLayout_Flat},
        };
        LayoutCombinations valid_out = {
            {DataLayout_Flat},
        };
        node->set_attr("valid_input_layouts", valid_in);
        node->set_attr("valid_output_layouts", valid_out);
    }
    BuilderParams generate_params(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        BuilderParams ro;
        return ro;
    }
    BuilderParams generate_descriptor(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        std::vector<ReadOnlyParam> desc, desc_param, desc_act;
        if (desc_param.size())
        {
            sgnode->set_attr(Constants::descriptor_param, desc_param);
        }
        if (desc_act.size())
        {
            sgnode->set_attr(Constants::descriptor_act, desc_act);
        }
        return desc;
    }
    const std::string generate_code_name(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        std::string code_name = "aipubt_internal_lib.o";
        DEBUG() << "generate_code_name: " << code_name;
        return code_name;
    }
    const std::string generate_kernel(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        std::string kernel_name;
        DEBUG() << "generate_kernel: " << kernel_name;
        return kernel_name;
    }

private:
    std::string m_target_name;
};

// name, version, type
REGISTER_PLUGIN(DSTestPlugin, 0x01010000, BUILDERPLUGIN_TYPE | GBPlugin_DS);
} // namespace aipubt
