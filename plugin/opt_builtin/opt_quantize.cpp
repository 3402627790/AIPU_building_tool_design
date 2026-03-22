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
#include "opt_plugin.h"
#include <core/fp16.h>

namespace aipubt
{
class BTQuantize : public BuilderOpPlugin
{
public:
    const GraphMatcherPattern get_graph_pattern()
    {
        return {{{"n1", OpType_Quantize}}, {}};
    }
    const std::string get_layer_type_name()
    {
        return "Quantize";
    }
    uint32_t get_score()
    {
        return 10000;
    }
    bool set_target(const char *target)
    {
        if (utils::startswith(target, "X3"))
        {
            return true;
        }
        return false;
    }
    bool check_params(BaseNodePtrList nodes)
    {
        auto n = nodes[0];
        if (n->inputs()[0]->type() != Tensor::FP16 || n->outputs()[0]->type() != Tensor::INT16)
        {
            return false;
        }
        return true;
    }
    void setup(BaseNodePtr node, BaseNodePtrList match_list)
    {
        node->set_attr("keeping_layout", true);
        LayoutCombinations valid_in = {};
        LayoutCombinations valid_out = {};
        node->set_attr("valid_input_layouts", valid_in);
        node->set_attr("valid_output_layouts", valid_out);
    }
    BuilderParams generate_params(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        auto total_size = sgnode->inputs()[0]->shape().symbol().size() * Tensor::dtype_size(sgnode->outputs()[0]->type());
        auto tec_size = total_size.floordiv(4) & 0xfffffe00;
        auto tail = total_size - tec_size * 3;
        float scale = sgnode->outputs()[0]->quantization().scale();
        int zp = sgnode->outputs()[0]->quantization().offset();
        const std::map<std::string, int> rdmod = {
            {"ROUND_TO_EVEN", 0},
            {"TRUNC", 1},
            {"CEIL", 2},
            {"FLOOR", 3}};
        int mode = 0;
        if (nodes[0]->params().count("round_mode"))
        {
            std::string m = nodes[0]->params()["round_mode"];
            if (rdmod.count(m))
            {
                mode = rdmod.at(m);
            }
            else
            {
                WARN() << "unsupport rounding mode:" << m;
            }
        }
        float nzp = 0.f - zp;
        half scale16 = scale;
        BuilderParams ro;
        ro.push_back(sgnode->inputs()[0]);
        ro.push_back(sgnode->outputs()[0]);
        ro.push_back(scale);
        ro.push_back(scale16);
        ro.push_back(nzp);
        ro.push_back(tec_size);
        ro.push_back(tail);
        ro.push_back(mode);
        return ro;
    }
    const std::string generate_code_name(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        std::string code_name = "quantize.o";
        return code_name;
    }
    const std::string generate_kernel(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        return "quantize_kernel";
    }
};

class DSBTQuantize : public BTQuantize
{
};
// name, version, type
REGISTER_PLUGIN(BTQuantize, 0x01010000, BUILDERPLUGIN_TYPE);
REGISTER_PLUGIN(DSBTQuantize, 0x01010000, BUILDERPLUGIN_TYPE | GBPlugin_DS);
} // namespace aipubt
