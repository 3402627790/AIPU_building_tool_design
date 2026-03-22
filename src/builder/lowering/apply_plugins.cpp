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
#include "builder/lowering/lowering.h"
#include "builder/plugin.h"
#include "core/config.h"
#include "core/constants.h"
#include "core/graph.h"
#include "core/graph_matcher.h"
#include "core/logger.h"
#include "core/node.h"
#include "core/plugin_loader.h"
#include "core/subgraph.h"
#include <algorithm>
#include <builder/target.h>
#include <set>
#include <unistd.h>
#include <vector>
namespace aipubt
{
void add_log_callback(const std::string &name, std::function<void(const std::string &)> cb, int level);
void remove_log_callback(const std::string &name);

static const std::vector<BuilderOpPluginInfo *> &get_plugins()
{
    static std::string m_target;
    auto target = get_target()->name();
    bool clear = false;
    if (m_target != target)
    {
        clear = true;
        m_target = target;
    }
    static std::vector<BuilderOpPluginInfo *> ret;
    if (clear)
    {
        ret.clear();
    }
    if (ret.size())
    {
        return ret;
    }
    PluginLoader *m_plugin_loader = PluginLoader::get_instance();
    pid_t pid = getpid();
    DEBUG() << "target " << m_target.c_str() << " PID: " << pid;
    auto disable_plugins = BuilderConfig::get<std::vector<std::string>>(Constants::disable_plugins);
    // auto enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);

    auto m_plugins = ((*m_plugin_loader)[BUILDERPLUGIN_TYPE]);
    for (auto plugin_ : m_plugins)
    {
        BuilderOpPluginInfo *plugin = reinterpret_cast<BuilderOpPluginInfo *>(plugin_);
        // if (enable_ds && !(plugin->plugin_type & GBPlugin_DS))
        // {
        //     continue;
        // }
        // if (target->isainfo().type < ISA_ZYv5 && plugin->plugin_type & GBPlugin_DS)
        // {
        //     continue;
        // }
        auto opt = plugin->init_handle();
        std::string name = plugin->name;
        if (contains(disable_plugins, name))
        {
            DEBUG() << "Disbaled Plugin:" << name;
            continue;
        }
        if (opt->set_target(m_target.c_str()))
        {
            pid = getpid();
            // DEBUG() << "set target " << target->name().c_str() << " for " << name << " pid: " << pid;
            ret.push_back(plugin);
        }
    }
    std::sort(ret.begin(), ret.end(), [](BuilderOpPluginInfo *p1, BuilderOpPluginInfo *p2) -> bool {
        auto p1_score = p1->init_handle()->get_score();
        auto p2_score = p2->init_handle()->get_score();
        if (p1_score == p2_score)
        {
            auto p1_type = p1->plugin_type & GBPlugin_DS;
            auto p2_type = p2->plugin_type & GBPlugin_DS;
            if (p1_type != p2_type) // non-dynamic shape plugin comes first
            {
                return p1_type < p2_type;
            }
        }
        return p1_score > p2_score;
    });
    return ret;
}

static std::vector<BuilderOpPluginInfo *> get_plugins_filtered(std::vector<BuilderOpPluginInfo *> &plugins,
                                                               unsigned int plugin_flag = 0)
{
    std::vector<BuilderOpPluginInfo *> plugins_filtered;
    if (plugin_flag)
    {
        for (auto plugin : plugins)
        {
            if (plugin->plugin_type & plugin_flag)
            {
                plugins_filtered.push_back(plugin);
            }
        }
        return plugins_filtered;
    }
    else
    {
        return plugins;
    }
}

// true:  allow fuse
// false: do NOT allow fuse
static bool check_io_tensor(std::vector<NodePtr> nodes)
{
    for (auto n : nodes)
    {
        for (auto o : n->outputs())
        {
            // block the fusion if the io_tensor would be disappear
            // (consumers.size() > 0 and all the consumers are internal)
            if (o->is_io_tensor() && o->consumers().size() > 0)
            {
                bool all_internal_consumers = true;
                for (auto c : o->consumers())
                {
                    if (!contains(nodes, c))
                    {
                        all_internal_consumers = false;
                        break;
                    }
                }
                if (all_internal_consumers)
                {
                    return false;
                }
            }
        }
    }
    return true;
}

static bool check_ds_plugin(const std::vector<NodePtr> &nodes, BuilderOpPluginInfo *plugin)
{
    if (BuilderConfig::get<bool>(Constants::enable_ds))
    {
        bool is_dynamic = false;
        for (auto n : nodes)
        {
            if (n->is_dynamic())
            {
                is_dynamic = true;
                break;
            }
        }
        bool is_ds_plugin = plugin->plugin_type & GBPlugin_DS;
        if ((is_dynamic && !is_ds_plugin) || (!is_dynamic && is_ds_plugin))
        {
            return false;
        }
    }
    return true;
}

static std::vector<std::vector<NodePtr>> filter_pattern(const std::vector<std::vector<NodePtr>> &matches, BuilderOpPluginInfo *plugin)
{
    std::vector<std::vector<NodePtr>> filtered;
    if (matches.size() == 0)
    {
        return filtered;
    }
    if (plugin->name == "HwaConvEltwise" || plugin->name == "HwaConvEltwisePooling" || plugin->name == "HwaCropConvEltwise")
    {
        uint32_t conv_idx, eltwise_idx;
        if (plugin->name == "HwaConvEltwise" || plugin->name == "HwaConvEltwisePooling")
        {
            conv_idx = 0;
            eltwise_idx = 1;
        }
        else // HwaCropConvEltwise
        {
            conv_idx = 1;
            eltwise_idx = 2;
        }
        std::map<NodePtr, uint32_t> elt_node_index;
        int filter_index;
        for (size_t i = 0; i < matches.size(); ++i)
        {
            auto pattern = matches[i];
            auto elt_node = pattern[eltwise_idx];
            filter_index = -1;
            for (size_t j = 0; j < filtered.size(); ++j)
            {
                auto filtered_elt_node = filtered[j][eltwise_idx];
                if (filtered_elt_node == elt_node)
                {
                    filter_index = j;
                }
            }

            if (filter_index >= 0)
            {
                auto conv_new = pattern[conv_idx];
                auto conv_old = filtered[filter_index][conv_idx];
                // same channel will use the first one to make sure no mismatch
                if (conv_new->inputs()[0]->shape().c() == conv_old->inputs()[0]->shape().c())
                {
                    if (conv_new == elt_node->parents()[0])
                    {
                        filtered[filter_index] = pattern;
                    }
                }
                // different channel will use the larger one to optimize the perforamcne
                else if (conv_new->inputs()[0]->shape().c() > conv_old->inputs()[0]->shape().c())
                {
                    filtered[filter_index] = pattern;
                }
            }
            else
            {
                filtered.push_back(pattern);
            }
        }
    }
    else
    {
        filtered = matches;
    }
    return filtered;
}

static bool apply_plugins_(GraphPtr graph,
                           std::vector<GraphPtr> all_graphs,
                           std::vector<BuilderOpPluginInfo *> &plugins)
{
    auto error_code = graph->global_tensors()[Constants::error_code];
    bool unified_view = true;
    const char *_unified_view = getenv(Constants::SUBGRAPH_VIEW);
    if (_unified_view)
    {
        unified_view = (atoi(_unified_view) != 0);
    }

    for (auto g : all_graphs)
    {
        // each subgraph using its own workspace, since for Multicore, the workspace is private
        // for single core, each subgraph's workspace should share main graph's workspace
        std::set<NodePtr> fused_nodes;
        for (auto plugin : plugins)
        {
            auto opt = plugin->init_handle();
            auto pattern = opt->get_graph_pattern();
            if (pattern.first.size())
            {
                GraphMatcher gm(pattern.first, pattern.second);
                auto gm_mode = gm.mode() | GraphMatcher::GreedyMode;
                if (plugin->plugin_type & GBPlugin_Internal)
                    gm_mode = gm_mode | GraphMatcher::SingleOpSubgraphMode;
                gm.set_mode(gm_mode);
                auto matches = gm.match(g);
                matches = filter_pattern(matches, plugin);
                for (auto m : matches)
                {
                    if (!check_io_tensor(m))
                    {
                        continue;
                    }
                    if (!check_ds_plugin(m, plugin))
                    {
                        continue;
                    }
                    if (m.size() == 1 && m[0]->type() == OpType_LayoutConvertor)
                    {
                        continue;
                    }
                    bool has_merged = false;
                    BaseNodePtrList mm;
                    for (auto mn : m)
                    {
                        mm.push_back(std::static_pointer_cast<BaseNode>(mn));
                        if (fused_nodes.count(mn))
                        {
                            has_merged = true;
                            break;
                        }
                    }
                    if (has_merged)
                    {
                        continue;
                    }
                    std::vector<std::string> warns, errors;
                    if (m.size() == 1)
                    {
                        auto capture_warn = [&warns](const std::string &msg) {
                            warns.push_back(msg);
                        };
                        auto capture_error = [&errors](const std::string &msg) {
                            errors.push_back(msg);
                        };
                        add_log_callback("_tw", capture_warn, _WARN_LEVEL);
                        add_log_callback("_te", capture_error, _ERROR_LEVEL);
                    }
                    bool res = opt->check_params(mm);
                    if (m.size() == 1)
                    {
                        m[0]->set_attr(Constants::build_warn_log, warns);
                        m[0]->set_attr(Constants::build_error_log, errors);
                    }
                    if (BuilderConfig::get<bool>(Constants::warn_all))
                    {
                        for (auto msg : m[0]->get_attr<std::vector<std::string>>(Constants::build_warn_log))
                        {
                            std::cout << msg << std::endl;
                            GCompilerLogger::increase_warn_count();
                        }
                    }
                    remove_log_callback("_tw");
                    remove_log_callback("_te");
                    if (res)
                    {
                        auto sg = SubGraph::create(g, m);
                        auto node = sg->as_node(true, false); // no need sg input
                        node->set_unified_view(unified_view);
                        node->set_attr("_match_list", m);
                        node->set_attr(Constants::plugin_attrs_name, plugin);
                        node->set_attr(Constants::error_code, error_code);
                        opt->setup(std::static_pointer_cast<BaseNode>(node), mm);
                        fused_nodes.insert(m.begin(), m.end());
                    }
                }
            }
        }
    }
    return true;
}

bool apply_plugins(GraphPtr g)
{
    auto plugins = get_plugins();
    auto all_graphs = get_all_graphs(g);
    return apply_plugins_(g, all_graphs, plugins);
}

bool apply_internal_plugins(GraphPtr g)
{
    auto plugins = get_plugins();
    std::vector<GraphPtr> all_graphs = {g};
    plugins = get_plugins_filtered(plugins, GBPlugin_Internal);
    return apply_plugins_(g, all_graphs, plugins);
}
} // namespace aipubt
