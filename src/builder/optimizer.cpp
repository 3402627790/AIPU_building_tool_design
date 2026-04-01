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
 * @Last Modified time: 2023-11-15 16:35:34
 */
#include "builder/optimizer.h"
#include "builder/lowering/lowering.h"
#include "builder/plugin.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/constants.h"
#include "core/graph_matcher.h"
#include "core/logger.h"
#include "core/node.h"
#include "core/subgraph.h"

namespace aipubt
{
BuilderOpPlugin *BuilderOptimizer::get_single_op_plugin(OpType t, uint32_t flag)
{
    auto m_plugins = ((*PluginLoader::get_instance())[BUILDERPLUGIN_TYPE]);
    flag = flag >> 16;
    for (auto plugin_ : m_plugins)
    {
        auto plugin = reinterpret_cast<BuilderOpPluginInfo *>(plugin_);
        if ((plugin->plugin_type >> 16) != flag)
        {
            continue;
        }
        auto opt = plugin->init_handle();
        auto p = opt->get_graph_pattern();
        if (p.second.size() == 0 && p.first.size() == 1 && p.first[0].second == t)
        {
            return opt;
        }
    }
    return nullptr;
}

BuilderOpPlugin *BuilderOptimizer::get_builder_op_plugin_by_name(const std::string &name, uint32_t flag)
{
    auto m_plugins = ((*PluginLoader::get_instance())[BUILDERPLUGIN_TYPE]);
    flag = flag >> 16;
    for (auto plugin_ : m_plugins)
    {
        auto plugin = reinterpret_cast<BuilderOpPluginInfo *>(plugin_);
        if ((plugin->plugin_type >> 16) != flag)
        {
            continue;
        }
        if (plugin->name != name)
        {
            continue;
        }
        return plugin->init_handle();
    }
    return nullptr;
}

void BuilderOptimizer::optimize(GraphPtr graph)
{
    auto output_tensors = graph->get_output_tensors();

    auto enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);
    if (enable_ds)
    {
        pooling_to_globalpool(graph);
        update_dynamic_max_shape(graph);
    }
    init_symbol_shape(graph);
    infer_symbol_shape(graph);

    update_disable_passes();
    lower_before_all(graph);
    lower_before_tiling(graph);
    if (BuilderConfig::get<bool>(Constants::show_raw_graph))
    {
        graph->to_svg("graph_raw.svg");
    }

    if (enable_ds)
    {
        divide_graph(graph);
        if (BuilderConfig::exist(Constants::show_divided_graph) &&
            BuilderConfig::get<bool>(Constants::show_divided_graph))
        {
            graph->to_svg("graph_divided.svg");
        }
    }

    auto all_graphs = graph->get_all_graphs();
    if (enable_ds)
    {
        auto it = std::find(all_graphs.begin(), all_graphs.end(), graph);
        if (it != all_graphs.end())
        {
            all_graphs.erase(it);
        }
    }
    for (auto g : all_graphs)
    {
        lower_before_optimize(g);
    }
    infer_symbol_shape(graph);

    unfold_graph(graph);
    lower_after_unfold(graph);

    lower_before_plugin(graph);
    if (enable_ds)
    {
        all_graphs = {graph};
    }

    apply_plugins(graph);

    for (auto g : all_graphs)
    {
        lower_after_optimize(g);
    }
    apply_internal_plugins(graph);

    // update subgraph real nodes
    for (auto n : graph->nodes())
    {
        if (SubGraphNode::is_subgraph_node(n))
        {
            auto sg = SubGraphNode::from_node(n)->subgraph();
            std::dynamic_pointer_cast<SubGraph>(sg)->update_real_nodes();
        }
    }

    for (auto t : output_tensors)
    {
        if (!graph->has_tensor(t))
        {
            ERROR() << "output tensor " << t->name() << " is not in graph";
        }
    }
}
} // namespace aipubt
