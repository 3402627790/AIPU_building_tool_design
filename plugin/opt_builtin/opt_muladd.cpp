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
 * @Date: 2024-11-11 17:16:53
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-11-11 17:16:53
 */
#include "opt_plugin.h"

namespace aipubt
{
class DSBTMulAdd : public BuilderOpPlugin
{
public:
    const GraphMatcherPattern get_graph_pattern()
    {
        return {{{"n1", OpType_Mul}, {"n2", OpType_Add}}, {{"n1", "n2"}}};
    }
    const std::string get_layer_type_name()
    {
        return "MulAdd";
    }
    uint32_t get_score()
    {
        if (check_plugin_option("ENABLE_BTMULADD_PLUGIN"))
        {
            return SINGLE_PLUGIN_SCORE;
        }
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
        // if (!check_plugin_option("ENABLE_BTMULADD_PLUGIN"))
        // {
        //     return false;
        // }

        for (auto n : nodes)
        {
            auto in0 = n->inputs()[0];
            auto in1 = n->inputs()[1];
            if (in0->type() != Tensor::FP16 || in1->type() != Tensor::FP16)
            {
                return false;
            }
            if (in0->shape().dim() < in1->shape().dim() ||
                in0->shape().size() < in1->shape().size())
            {
                std::swap(in0, in1);
            }
            if (in0->shape().dim() != 4 && in1->shape().dim() != 1)
            {
                return false;
            }
            if ((n->type() == OpType_Mul && in1->shape()[0] != 1) ||
                (n->type() == OpType_Add && in1->shape()[0] != in0->shape()[3]))
            {
                return false;
            }
        }
        return true;
    }
    void setup(BaseNodePtr node, BaseNodePtrList match_list)
    {
        node->set_attr("keeping_layout", false);
        LayoutCombinations valid_in = {
            {DataLayout_Flat, DataLayout_Flat, DataLayout_Flat},
        };
        LayoutCombinations valid_out = {
            {DataLayout_NHWC},
        };
        node->set_attr("valid_input_layouts", valid_in);
        node->set_attr("valid_output_layouts", valid_out);
    }
    BuilderParams generate_params(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        auto in0 = sgnode->inputs()[0];
        auto in1 = sgnode->inputs()[1];
        auto in2 = sgnode->inputs()[2];
        auto out = sgnode->outputs()[0];
        if (in0->shape().dim() < in1->shape().dim() ||
            in0->shape().size() < in1->shape().size())
        {
            std::swap(in0, in1);
        }
        auto in0_shape = in0->shape().symbol();
        Expr d0 = in0_shape[0];
        for (int i = 1; i < in0_shape.dim() - 1; ++i)
        {
            d0 *= in0_shape[i];
        }
        Expr d1 = in0_shape[-1];
        BuilderParams ro;
        ro.push_back(in0);
        ro.push_back(in1);
        ro.push_back(in2);
        ro.push_back(out);
        ro.push_back(d0);
        ro.push_back(d1);
        return ro;
    }
    BuilderParams generate_descriptor(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        BuilderParams desc, desc_param, desc_act;
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
        std::string code_name = "muladd.o";
        DEBUG() << "generate_code_name: " << code_name;
        return code_name;
    }
    const std::string generate_kernel(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        std::string kernel_name = "muladd_c_fp16";
        DEBUG() << "generate_kernel: " << kernel_name;
        return kernel_name;
    }

private:
    std::string m_target_name;
};

class BTMulAdd : public DSBTMulAdd
{
};

// name, version, type
REGISTER_PLUGIN(BTMulAdd, 0x01010000, BUILDERPLUGIN_TYPE);
REGISTER_PLUGIN(DSBTMulAdd, 0x01010000, BUILDERPLUGIN_TYPE | GBPlugin_DS);
} // namespace aipubt
