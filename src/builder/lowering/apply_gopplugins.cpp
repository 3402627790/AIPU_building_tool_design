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
 * @Date: 2020-01-29 14:27:03
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-08-26 11:11:24
 */
#include "builder/gop_plugin.h"
#include "builder/lowering/lowering.h"
#include "builder/plugin.h"
#include "builder/plugin/ds_code_wrapper.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/constants.h"
#include "core/graph.h"
#include "core/graph_matcher.h"
#include "core/logger.h"
#include "core/node.h"
#include "core/ops/ops.h"
#include "core/plugin_loader.h"
#include "core/subgraph.h"
#include <algorithm>
#include <set>
#include <unistd.h>
#include <vector>

namespace aipubt
{
struct ImplInfo
{
    GopImplPtr impl;
    std::vector<NodePtr> nodes;
};

static const std::vector<GopPluginInfo *> &get_gopplugins()
{

    auto target = get_target()->name();
    static std::vector<GopPluginInfo *> ret;

    PluginLoader *m_plugin_loader = PluginLoader::get_instance();
    pid_t pid = getpid();
    DEBUG() << "target " << target.c_str() << " PID: " << pid;

    auto m_plugins = ((*m_plugin_loader)[GOPPLUGIN_TYPE]);
    INFO() << "plugins: " << m_plugins.size();

    for (auto plugin_ : m_plugins)
    {
        GopPluginInfo *plugin = reinterpret_cast<GopPluginInfo *>(plugin_);
        auto opt = plugin->init_handle();
        std::string name = plugin->name;

        // INFO() << "plugin name " << name;
        // INFO() << "set target " << target;
        if (opt->set_target(target.c_str()))
        {
            // DEBUG() << "set target " << target->name().c_str() << " for " << name << " pid: " << pid;
            ret.push_back(plugin);
        }
    }

    return ret;
}

static std::vector<ImplInfo> plugin_match(GopPluginInfo *plugin, NodePtr n)
{
    std::vector<ImplInfo> impl_info;
    bool is_ds_plugin = plugin->plugin_type & GBPlugin_DS;
    if (is_ds_plugin)
    {
        return impl_info;
    }

    auto opt = plugin->init_handle();
    auto patterns = opt->patterns();
    for (auto pattern : patterns)
    {
        INFO() << "node type: " << n->type();
        INFO() << "pattern type: " << pattern.nodes[0].second;

        GraphMatcherPattern gpattern = std::make_pair(pattern.nodes, pattern.edges);
        GraphMatcher matcher(gpattern);
        matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
        auto nodes = matcher.match_nodes(n);
        if (nodes.size())
        {
            auto impl = opt->get_impl(nodes, pattern);
            ImplInfo info{impl, nodes};
            impl_info.push_back(info);
        }
    }

    return impl_info;
}

static ImplInfo get_best_impl(NodePtr n, std::set<GopPluginInfo *> &plugins)
{
    std::vector<ImplInfo> candidates;
    for (auto plugin : plugins)
    {
        auto infos = plugin_match(plugin, n);
        for (auto info : infos)
        {
            candidates.push_back(info);
        }
    }

    if (!candidates.size())
    {
        ERROR() << "no plugin found";
    }
    auto max_it = std::max_element(candidates.begin(), candidates.end(),
                                   [](const ImplInfo a, const ImplInfo b) {
                                       if (a.nodes.size() > b.nodes.size())
                                       {
                                           return false;
                                       }
                                       else if (a.nodes.size() < b.nodes.size())
                                       {
                                           return true;
                                       }

                                       return a.impl->score() < b.impl->score();
                                   });

    ImplInfo info = *max_it;
    return info;
}

static bool apply_gopplugin_(GraphPtr g,
                             const std::map<int, std::set<GopPluginInfo *>> &plugin_map)
{
    auto error_code = g->global_tensors()[Constants::error_code];
    bool unified_view = true;
    const char *_unified_view = getenv(Constants::SUBGRAPH_VIEW);
    if (_unified_view)
    {
        unified_view = (atoi(_unified_view) != 0);
    }

    for (auto n : g->nodes())
    {
        INFO() << n->name() << " type: " << n->type();
    }

    auto nodes = g->nodes();
    std::set<NodePtr> fused;
    for (auto n : nodes)
    {
        if (n->type() == OpType_Input)
        {
            continue;
        }
        if (contains(fused, n))
        {
            continue;
        }
        if (!plugin_map.count(n->type()))
        {
            continue;
        }
        auto plugins = plugin_map.at(n->type());
        auto info = get_best_impl(n, plugins);
        if (!info.impl)
        {
            ERROR() << "no plugin impl for " << n->name();
            return false;
        }
        INFO() << "get best impl for " << n;

        auto sg_nodes = info.nodes;
        if (BuilderConfig::get<bool>(Constants::enable_ds))
        {
            auto it = std::find_if(info.nodes.begin(), info.nodes.end(), [](NodePtr n) {
                return n->is_dynamic();
            });
            if (it != info.nodes.end())
            {
                auto out_name = g->get_valid_tensor_name("paramupdate/output");
                TensorPtr out = std::make_shared<Tensor>(out_name, TensorShape{0}, Tensor::UINT8);
                auto update_node = Node::create(g->get_valid_node_name("paramupdate"), OpType_ParamUpdate);
                update_node->add_output(out);
                update_node->set_attr("_match_list", std::vector<NodePtr>{update_node});
                update_node->set_attr(Constants::plugin_impl, std::make_shared<GopDSCodeImpl>());
                g->add_node(update_node);
                fused.insert(update_node);

                auto dummy_node = ops::dummy({out})[0]->node();
                sg_nodes.push_back(dummy_node);
                fused.insert(dummy_node);
            }
        }

        auto sg = SubGraph::create(g, sg_nodes);
        auto node = sg->as_node(true, false); // no need sg input
        node->set_unified_view(unified_view);
        node->set_attr("_match_list", info.nodes);
        node->set_attr(Constants::plugin_impl, info.impl);
        info.impl->update_prefer_cfg("prefer_cfg", true);
        fused.insert(info.nodes.begin(), info.nodes.end());
    }

    return true;
}

static std::map<int, std::set<GopPluginInfo *>> get_plugin_map(const std::vector<GopPluginInfo *> &plugins)
{
    std::map<int, std::set<GopPluginInfo *>> plugin_map;
    for (auto plugin : plugins)
    {
        auto opt = plugin->init_handle();
        auto patterns = opt->patterns();
        for (auto pattern : patterns)
        {
            auto type = pattern.nodes[0].second;
            plugin_map[type].insert(plugin);
        }
    }
    return plugin_map;
}

bool apply_gopplugins(GraphPtr g)
{
    auto plugins = get_gopplugins();
    INFO() << "total plugin size: " << plugins.size();

    std::map<int, std::set<GopPluginInfo *>> plugin_map = get_plugin_map(plugins);

    auto ret = apply_gopplugin_(g, plugin_map);

    return ret;
}
} // namespace aipubt
