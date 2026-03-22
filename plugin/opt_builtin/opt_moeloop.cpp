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
 * @Date: 2026-03-03 10:44:06
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2026-03-03 10:44:06
 */
#include "opt_plugin.h"
#include <core/graph.h>
#include <core/ifnode.h>
#include <core/node.h>
#include <core/subgraph.h>

namespace aipubt
{
inline NodePtr get_real_node(NodePtr n)
{
    auto sgn = SubGraphNode::from_node(n);
    if (sgn)
    {
        auto nodes = sgn->get_real_node();
        if (nodes.size() == 1)
        {
            return nodes[0];
        }
    }
    return n;
}

class DSBTMoELoop : public BuilderOpPlugin
{
public:
    const GraphMatcherPattern get_graph_pattern()
    {
        return {{{"n1", OpType_Loop}, {"n2", OpType_Crop}}, {{"n1", "n2"}}};
    }
    const std::string get_layer_type_name()
    {
        return "MoELoop";
    }
    uint32_t get_score()
    {
        if (check_plugin_option("ENABLE_BTMOELOOP_PLUGIN"))
        {
            return FUSED_PLUGIN_SCORE;
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
        if (!check_plugin_option("ENABLE_BTMOELOOP_PLUGIN"))
        {
            return false;
        }
        auto n = nodes[0];
        if (n->inputs().size() != 6 || n->outputs().size() != 2)
        {
            return false;
        }
        auto inp = n->inputs()[3];
        if (inp->type() != Tensor::FP16 ||
            inp->shape().dim() != 2 || inp->shape()[1] % 16 != 0)
        {
            return false;
        }
        if (inp->shape()[0] > 524287)
        {
            return false;
        }
        return true;
    }
    void setup(BaseNodePtr node, BaseNodePtrList match_list)
    {
        node->set_attr("keeping_layout", false);
        LayoutCombinations valid_in = {{
            DataLayout_Flat,
            DataLayout_Flat,
            DataLayout_Flat,
            DataLayout_Flat,
            DataLayout_Flat,
            DataLayout_Flat,
        }};
        LayoutCombinations valid_out = {{
            DataLayout_Flat,
            DataLayout_Flat,
        }};
        node->set_attr("valid_input_layouts", valid_in);
        node->set_attr("valid_output_layouts", valid_out);

        node->set_attr(Constants::mmu_support, false);
        node->set_attr(Constants::sram_free, false);
        uint32_t ws_size = 3 * node->inputs()[3]->mem_size();
        DEBUG() << "ws_size: " << ws_size;
        node->set_attr("workspace_size", ws_size);
    }
    BuilderParams generate_params(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        int mid_channel = sgnode->get_attr<int>("mid_channel");
        int split_offset = sgnode->get_attr<int>("split_offset");
        int param_offset = sgnode->get_attr<int>("param_offset");
        int max_aiff_hw0 = sgnode->get_attr<int>("max_aiff_hw0");
        int max_aiff_hw1 = sgnode->get_attr<int>("max_aiff_hw1");
        int split_num0 = sgnode->get_attr<int>("split_num0");
        int split_num1 = sgnode->get_attr<int>("split_num1");
        DEBUG() << "mid_channel: " << mid_channel;
        DEBUG() << "split_offset: " << split_offset;
        DEBUG() << "param_offset: " << param_offset;
        DEBUG() << "max_aiff_hw0: " << max_aiff_hw0;
        DEBUG() << "max_aiff_hw1: " << max_aiff_hw1;
        DEBUG() << "split_num0: " << split_num0;
        DEBUG() << "split_num1: " << split_num1;
        auto input = sgnode->inputs()[3];
        BuilderParams ro;
        ro.push_back(sgnode->inputs()[0]);
        ro.push_back(input);
        ro.push_back(sgnode->inputs()[4]);
        ro.push_back(sgnode->inputs()[5]);
        auto out_name = nodes[1]->outputs()[0]->name();
        for (auto t : sgnode->outputs())
        {
            if (t->name() == out_name)
            {
                ro.push_back(t);
                break;
            }
        }
        auto in_shape = input->shape().symbol();
        ro.push_back(in_shape[0]);
        ro.push_back(in_shape[1]);
        ro.push_back(mid_channel);
        ro.push_back(split_offset);
        ro.push_back(param_offset);
        ro.push_back(max_aiff_hw0);
        ro.push_back(max_aiff_hw1);
        ro.push_back(split_num0);
        ro.push_back(split_num1);
        ro.push_back(sgnode->constants()["shape_param"]);
        ro.push_back(sgnode->constants()["desc_offset"]);
        ro.push_back(sgnode->get_attr<TensorPtr>(Constants::descriptorbase));
        ro.push_back(sgnode->get_attr<TensorPtr>(Constants::descriptor_param_base));
        ro.push_back(sgnode->get_attr<TensorPtr>(Constants::descriptor_act_base));
        ro.push_back(sgnode->get_attr<TensorPtr>("workspace"));
        return ro;
    }
    BuilderParams generate_descriptor(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        auto node = nodes[0];
        auto sg = SubGraphNode::from_node(node)->subgraph();
        std::vector<NodePtr> ifs;
        for (auto n : sg->nodes())
        {
            if (get_real_node(n)->type() == OpType_Switch)
            {
                for (auto t : n->inputs())
                {
                    ifs.push_back(t->node());
                }
                break;
            }
        }
        int expert_num = ifs.size();
        DEBUG() << "expert_num: " << expert_num;
        auto target = m_target_name;
        sgnode->constants().clear();
        BuilderParams desc;
        std::vector<uint32_t> offsets(expert_num * 8, 0);
        std::vector<uint32_t> oc_split0, oc_split1, shape_params;
        int mid_channel = 0;
        int max_aiff_hw0 = 0, max_aiff_hw1 = 0;
        int split_num0 = 0, split_num1 = 0;
        for (int i = 0; i < expert_num; ++i)
        {
            auto ifn = std::dynamic_pointer_cast<IfNode>(ifs[i]);
            auto head = ifn->head();
            auto sep = ifn->sep();
            auto children = head->outputs()[2]->consumers();
            while (children.size() == 1)
            {
                children = children[0]->children();
            }
            NodePtr fc0, act0, fc1, mul, fc2;
            for (auto c : children)
            {
                std::vector<NodePtr> fcs;
                NodePtr act = nullptr;
                auto p = c;
                while (p != sep)
                {
                    auto t = get_real_node(p)->type();
                    if (t == OpType_Convolution)
                    {
                        fcs.push_back(p);
                    }
                    else if (t == OpType_Activation)
                    {
                        act = p;
                    }
                    else if (t == OpType_Eltwise)
                    {
                        mul = p;
                    }
                    p = p->children()[0];
                }
                if (act)
                {
                    fc0 = fcs[0];
                    act0 = act;
                }
                else
                {
                    fc1 = fcs[0];
                }
                fc2 = fcs[1];
            }
            std::vector<NodePtr> sgns = {fc0, act0, fc1, mul, fc2};
            for (size_t j = 0; j < sgns.size(); ++j)
            {
                auto sgn = sgns[j];
                auto n = get_real_node(sgn);
                DEBUG() << "processing node: " << n->name() << "...";
                BuilderParams d;
                std::vector<TensorPtr> constants;
                std::vector<uint32_t> params;
                if (n->type() == OpType_Convolution)
                {
                    size_t max_aiff_hw = 0;
                    size_t split_num = 0;
                    d = gen_conv_desc(sgn, n, target, max_aiff_hw, split_num, constants, params);
                    if (sgn == fc2)
                    {
                        oc_split1 = params;
                        max_aiff_hw1 = max_aiff_hw;
                        split_num1 = split_num;
                    }
                    else
                    {
                        oc_split0 = params;
                        max_aiff_hw0 = max_aiff_hw;
                        split_num0 = split_num;
                    }
                }
                else if (n->type() == OpType_Activation)
                {
                    d = gen_lut_desc(sgn, n, target, constants);
                }
                else if (n->type() == OpType_Eltwise)
                {
                    d = gen_eltwise_desc(sgn, n, target, params);
                    shape_params = params;
                }
                for (auto t : constants)
                {
                    auto name = n->name() + "/" + t->name();
                    DEBUG() << "push weight: " << name << " " << t->mem_size();
                    sgnode->constants()[name] = t;
                }
                offsets[i * 8 + j] = desc.size();
                desc.insert(desc.end(), d.begin(), d.end());
            }
            mid_channel = fc0->outputs()[0]->shape().c_total();
        }
        auto params = oc_split0;
        int split_offset = params.size();
        params.insert(params.end(), oc_split1.begin(), oc_split1.end());
        int param_offset = params.size();
        params.insert(params.end(), shape_params.begin(), shape_params.end());
        sgnode->set_attr("mid_channel", mid_channel);
        sgnode->set_attr("split_offset", split_offset);
        sgnode->set_attr("param_offset", param_offset);
        sgnode->set_attr("max_aiff_hw0", max_aiff_hw0);
        sgnode->set_attr("max_aiff_hw1", max_aiff_hw1);
        sgnode->set_attr("split_num0", split_num0);
        sgnode->set_attr("split_num1", split_num1);
        set_zerocopy_params(sgnode, params, "shape_param");
        set_zerocopy_params(sgnode, offsets, "desc_offset");
        BuilderParams desc_param, desc_act;
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
        std::string code_name = "moeloop.o";
        DEBUG() << "generate_code_name: " << code_name;
        return code_name;
    }
    const std::string generate_kernel(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        std::string kernel_name = "dsbt_moeloop";
        DEBUG() << "generate_kernel: " << kernel_name;
        return kernel_name;
    }

private:
    std::string m_target_name;
};

class BTMoELoop : public DSBTMoELoop
{
};

// name, version, type
REGISTER_PLUGIN(BTMoELoop, 0x01010000, BUILDERPLUGIN_TYPE);
REGISTER_PLUGIN(DSBTMoELoop, 0x01010000, BUILDERPLUGIN_TYPE | GBPlugin_DS);
} // namespace aipubt
