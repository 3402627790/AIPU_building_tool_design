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
// This entire notice must be reproduced on all copies of this file and copies
// of this file may only be made by a person if such person is permitted to do
// so under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Samule Han
 * @Date: 2020-03-13 15:50:28
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-10-15 15:21:50
 */
#include <builder/plugin.h>
#include <core/optype.h>
#include <fstream>
#include <gcompiler_api.h>
#include <map>
#include <string>
#include <vector>

#define MAJOR_VERSION 2
#define MINOR_VERSION 0

#ifndef PLUGIN_BUILDNUM
#define PLUGIN_BUILDNUM 1
#endif //!PLUGIN_BUILDNUM

#define VERSION_NUMBER ((((MAJOR_VERSION)&0xff) << 24) | (((MINOR_VERSION)&0xff) << 16) | (PLUGIN_BUILDNUM & 0xffff))

namespace aipubt
{
typedef std::vector<DataLayout> LayoutList;
typedef std::vector<LayoutList> LayoutCombinations;

//1. register a op name: NewOp
REGISTER_CUSTOMIZE_OPTYPE(NewOp);

class LUTPlugin : public BuilderOpPlugin
{
public:
    const GraphMatcherPattern get_graph_pattern()
    {
        return {{{"n1", (OpType)OpType_NewOp}}, {}};
    }
    uint32_t get_score() { return 10; };
    bool check_params(BaseNodePtrList nodes) { return true; }
    bool set_target(const char *target)
    {
        m_target_name = target;
        return true;
    };
    void setup(BaseNodePtr node, BaseNodePtrList match_list)
    {
        node->set_attr("keeping_layout", true);
        LayoutCombinations valid_in = {
            {DataLayout_NCHWC32},
            {DataLayout_NCHWC16},
            {DataLayout_NHWC},
            {DataLayout_NCHW},
        };
        LayoutCombinations valid_out = {
            {DataLayout_NCHWC32},
            {DataLayout_NCHWC16},
            {DataLayout_NHWC},
            {DataLayout_NCHW},
        };
        node->set_attr("valid_input_layouts", valid_in);
        node->set_attr("valid_output_layouts", valid_out);
    }
    BuilderParams generate_params(BaseNodePtr sgnode,
                                  BaseNodePtrList &nodes)
    {
        auto node = nodes[0];
        BuilderParams ro;
        int input_shape_dim = node->inputs()[0]->shape().dim();
        int start_idx = input_shape_dim < 4 ? 0 : 1;
        for (int i = 0; i < 3; i++)
        {
            if (i < input_shape_dim)
            {
                ro.push_back(node->inputs()[0]->shape()[start_idx + i]);
            }
            else
            {
                ro.push_back(1);
            }
        }
        ro.push_back(sgnode->constants()[node->name() + "/lut"]);
        ro.push_back(node->inputs()[0]);
        ro.push_back(node->outputs()[0]);
        return ro;
    }
    const std::string generate_code_name(BaseNodePtr sgnode, BaseNodePtrList &nodes) override
    {
        auto node = nodes[0];
        std::string activation_method = node->params()["method"];
        std::string name;
        if (activation_method == "SIGMOID")
        {
            name = "sigmoid";
        }
        else if (activation_method == "TANH")
        {
            name = "tanh";
        }
        name += ".bin";
        //if your bin file name is follow the suffix-target rule:
        // for zy_v1 target: would add a suffix: _zy_sc002/_zy_sc003, for qian add _z1_1104
        // you can use the function complete_text_code_name to complete your code name

        return complete_text_code_name(name);
        // else you need deal with the bin filename(function name) by your self.
        if (m_target_name == "Z1_1104")
        {
            name = "qian_1104_" + name;
        }
        else
        {
            //other cases
        }
        return name;
    }

private:
    std::string m_target_name;
};
// name, version, pluginType
REGISTER_PLUGIN(LUTPlugin, VERSION_NUMBER, BUILDERPLUGIN_TYPE);

}; // namespace aipubt
