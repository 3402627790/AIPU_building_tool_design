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
 * @Author: Kyle Huang
 * @Date: 2026-1-7 10:04:00
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2026-1-7 10:04:00
 */
 #include "builder/plugin.h"
 #include "core/config.h"
 #include "core/core.h"
 #include "core/plugin.h"
 #include "core/plugin_loader.h"
 #include "passes.h"
 
 namespace aipubt
 {
 static inline bool criteria(NodePtr n)
 {
     if (n->type() == OpType_SubGraph && n->attrs().count("is_scalar_plugin"))
     {
         if (!(bool)n->attrs()["is_scalar_plugin"])
         {
             ERROR();
         }
         return true;
     }
     if (
         n->type() != OpType_Activation &&
         n->type() != OpType_Cast &&
         n->type() != OpType_Reshape &&
         n->type() != OpType_InternalConstant &&
         n->type() != OpType_Gather &&
         n->type() != OpType_Eltwise &&
         n->type() != OpType_Logical)
     {
         return false;
     }
 
     // INFO() << "check type before: " << n;
     // for (auto i : n->inputs())
     // {
     //     INFO() << "input shape: " << i->shape();
     // }
     // for (auto i : n->inputs())
     // {
     //     INFO() << "output shape: " << i->shape();
     // }
 
     if (n->type() != OpType_Reshape)
     {
         bool unquantifiable = false;
         if (n->params().count("unquantifiable"))
         {
             unquantifiable = n->params()["unquantifiable"];
         }
 
         bool ignore_scale_zp = false;
         if (n->params().count("ignore_scale_zp"))
         {
             ignore_scale_zp = n->params()["ignore_scale_zp"];
         }
 
         if (!unquantifiable && !ignore_scale_zp)
         {
             return false;
         }
     }
 
     // INFO() << "check type after: " << n;
     // for (auto i : n->inputs())
     // {
    //     INFO() << "input shape: " << i->shape();
    // }
    // for (auto i : n->inputs())
    // {
    //     INFO() << "output shape: " << i->shape();
    // }

    // for (auto i : n->inputs())
    for (size_t k = 0; k < n->inputs().size(); k++)
    {
        auto i = n->inputs()[k];
        uint32_t shape = 1;
        if (n->type() == OpType_Gather && k == 0)
        {
            continue;
        }

        for (auto s : i->shape())
        {
            shape *= s;
        }
        if (shape != 1)
        {
            return false;
        }
    }

    for (auto o : n->outputs())
    {
        uint32_t shape = 1;
        for (auto s : o->shape())
        {
            shape *= s;
        }
        if (shape != 1)
        {
            return false;
        }
    }

    return true;
}

static bool is_top(NodePtr n)
{
    if (!criteria(n))
    {
        return false;
    }

    for (auto p : n->parents())
    {
        if (criteria(p))
        {
            return false;
        }
    }
    return true;
}

static void get_ancester_nodes(NodePtr n, std::set<NodePtr> &ancester_nodes)
{
    for (auto p : n->parents())
    {
        if (contains(ancester_nodes, p))
        {
            continue;
        }
        // INFO() << "parent node: " << p;
        ancester_nodes.insert(p);
        get_ancester_nodes(p, ancester_nodes);
    }
}

static bool valid_other_input(NodePtr n, const std::set<NodePtr> &scalar_nodes)
{
    if (n->outputs().size() > 1)
    {
        return false;
    }

    //input is not access to scalar_nodes
    std::set<NodePtr> ancester_nodes;
    get_ancester_nodes(n, ancester_nodes);
    for (auto n : ancester_nodes)
    {
        if (contains(scalar_nodes, n))
        {
            return false;
        }
    }

    return true;
}

static void get_scalar_nodes(NodePtr n, std::set<NodePtr> &scalar_nodes)
{
    scalar_nodes.insert(scalar_nodes.begin(), n);
    for (auto c : n->children())
    {
        if (criteria(c))
        {
            if (c->parents().size() == 1 || n->children().size() == 1)
            {
                get_scalar_nodes(c, scalar_nodes);
            }
            else if (c->parents().size() == 2)
            {
                //if child has 2 input then need to check another node
                NodePtr other_input;
                for (auto p : c->parents())
                {
                    if (p != n)
                    {
                        other_input = p;
                    }
                }
                std::set<NodePtr> branch_nodes;
                if (valid_other_input(other_input, scalar_nodes)) //ancester nodes in scalar_nodes
                {
                    for (auto n : branch_nodes)
                    {
                        scalar_nodes.insert(n);
                    }
                    get_scalar_nodes(c, scalar_nodes);
                }
            }
        }
    }
}

bool fuse_multi_scalars(GraphPtr g)
{
    BuilderOpPluginInfo *scalar_plugin = nullptr;
    PluginLoader *plugin_loader = PluginLoader::get_instance();
    auto plugins = ((*plugin_loader)[BUILDERPLUGIN_TYPE]);
    for (auto p : plugins)
    {
        BuilderOpPluginInfo *plugin = reinterpret_cast<BuilderOpPluginInfo *>(p);
        if (plugin->name == "MultiScalars")
        {
            scalar_plugin = plugin;
            break;
        }
    }

    if (!scalar_plugin)
    {
        return false;
    }

    for (int cnt = 0; cnt < 2; cnt++)
    {

        std::set<NodePtr> merged_nodes;
        auto nodes = g->nodes();
        for (auto n : nodes)
        {
            std::set<NodePtr> scalar_nodes;
            if (contains(merged_nodes, n))
            {
                continue;
            }
            if (!is_top(n))
            {
                continue;
            }
            // INFO() << "top node: " << n;
            get_scalar_nodes(n, scalar_nodes);
            if (scalar_nodes.size() > 1) //generate scalar plugin
            {
                std::vector<NodePtr> scalar_nodes_vec(scalar_nodes.begin(), scalar_nodes.end());
                auto sg_merged = SubGraph::create(g, scalar_nodes_vec);
                auto sg_merged_node = sg_merged->as_node(true, true);
                sg_merged_node->set_unified_view(false);
                sg_merged_node->set_attr("is_scalar_plugin", true);
                for (auto n : scalar_nodes_vec)
                {
                    merged_nodes.insert(n);
                }
            }
        }
    }

    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        if (n->type() == OpType_SubGraph && n->attrs().count("is_scalar_plugin"))
        {
            if (!(bool)n->attrs()["is_scalar_plugin"])
            {
                ERROR();
            }
            // INFO() << "scalar plugin: " << n;

            //flattern & unfold
            std::vector<NodePtr> scalar_nodes_vec;
            auto sgn = SubGraphNode::from_node(n);
            auto sg = sgn->subgraph();
            if (sgn)
            {
                sgn->flat();
                scalar_nodes_vec = sgn->unfold();
            }

            auto scalar_merged = SubGraph::create(g, scalar_nodes_vec);
            scalar_merged->topological_sort();
            scalar_nodes_vec = scalar_merged->nodes();

            auto node = scalar_merged->as_node(true, false);

            node->set_unified_view(true);
            node->set_attr("_match_list", scalar_nodes_vec);
            node->set_attr(Constants::plugin_attrs_name, scalar_plugin);
            auto error_code = g->global_tensors()[Constants::error_code];
            node->set_attr(Constants::error_code, error_code);
            auto opt = scalar_plugin->init_handle();
            BaseNodePtrList mm;
            for (auto mn : scalar_merged->nodes())
            {
                mm.push_back(std::static_pointer_cast<BaseNode>(mn));
            }

            // for(auto i : node->inputs())
            // {
            //     INFO() << "input: " << i->name();
            // }

            // for(auto o : node->outputs())
            // {
            //     INFO() << "output: " << o->name();
            // }

            opt->setup(std::static_pointer_cast<BaseNode>(node), mm);
        }
    }

    return true;
}
} // namespace aipubt