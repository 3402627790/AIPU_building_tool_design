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
 * @Date: 2025-12-01 13:56:31
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2025-12-01 13:56:31
 */
#include "opt_plugin.h"

namespace aipubt
{
class DSBTSwitch : public BuilderOpPlugin
{
public:
    const GraphMatcherPattern get_graph_pattern()
    {
        return {{{"n1", OpType_Switch}}, {}};
    }
    const std::string get_layer_type_name()
    {
        return "Switch";
    }
    uint32_t get_score()
    {
        return 0;
    }
    bool set_target(const char *target)
    {
        if (utils::startswith(target, "X3"))
        {
            m_target_name = target;
            return true;
        }
        return false;
    }
    bool check_params(BaseNodePtrList nodes)
    {
        auto node = nodes[0];
        if (node->inputs().size() < 1 || node->outputs().size() != 1)
        {
            return false;
        }
        auto inp0 = node->inputs()[0];
        for (size_t i = 0; i < node->inputs().size(); ++i)
        {
            auto t = node->inputs()[i];
            if (t->type() != inp0->type() || t->shape() != inp0->shape())
            {
                return false;
            }
        }
        auto out = node->outputs()[0];
        if (inp0->type() != out->type() || inp0->shape() != out->shape())
        {
            return false;
        }
        return true;
    }
    void setup(BaseNodePtr node, BaseNodePtrList match_list)
    {
        LayoutCombinations valid_in = {
            std::vector<DataLayout>(node->inputs().size(), DataLayout_Flat),
            std::vector<DataLayout>(node->inputs().size(), DataLayout_NHWC),
            std::vector<DataLayout>(node->inputs().size(), DataLayout_NCHWC16),
            std::vector<DataLayout>(node->inputs().size(), DataLayout_NCHWC32),
        };
        LayoutCombinations valid_out = {
            {DataLayout_Flat},
            {DataLayout_NHWC},
            {DataLayout_NCHWC16},
            {DataLayout_NCHWC32},
        };
        node->set_attr("keeping_layout", true);
        node->set_attr("valid_input_layouts", valid_in);
        node->set_attr("valid_output_layouts", valid_out);
    }
    BuilderParams generate_params(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        BuilderParams ro;
        ro.push_back((uint32_t)sgnode->inputs().size());
        ro.push_back(sgnode->outputs()[0]);
        ro.push_back(sgnode->get_attr<TensorPtr>(Constants::descriptorbase));
        ro.push_back(sgnode->get_attr<TensorPtr>(Constants::descriptor_param_base));
        ro.push_back(sgnode->get_attr<TensorPtr>(Constants::descriptor_act_base));
        return ro;
    }
    BuilderParams generate_descriptor(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        BuilderParams desc, desc_param, desc_act;
        for (auto t : sgnode->inputs())
        {
            desc.push_back(t);
            desc.push_back(t->shape().symbol().size() * Tensor::dtype_size(t->type()));
        }
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
        std::string code_name = "switch.o";
        DEBUG() << "generate_code_name: " << code_name;
        return code_name;
    }
    const std::string generate_kernel(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        std::string kernel_name = "dsbt_switch";
        DEBUG() << "generate_kernel: " << kernel_name;
        return kernel_name;
    }

private:
    std::string m_target_name;
};

class BTSwitch : public DSBTSwitch
{
};

// name, version, type
REGISTER_PLUGIN(BTSwitch, 0x01010000, BUILDERPLUGIN_TYPE);
REGISTER_PLUGIN(DSBTSwitch, 0x01010000, BUILDERPLUGIN_TYPE | GBPlugin_DS);
} // namespace aipubt
