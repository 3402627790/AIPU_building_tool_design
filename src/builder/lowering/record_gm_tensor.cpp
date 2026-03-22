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
 * @Author: Kyle huang
 * @Date: 2024-1-23 17:58:05
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-08-21 16:39:25
 */
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/constants.h"
#include "core/core.h"
#include "core/plugin_loader.h"
#include "utils/utils.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

namespace aipubt
{

void record_gm_tensor(GraphPtr g)
{
    std::string load_file = BuilderConfig::get<std::string>(Constants::load_prior_in_gm);
    std::map<std::string, int> prior_in_gm;
    if ((load_file != ""))
    {
        if (!utils::file_exists(load_file))
        {
            WARN() << "doesn't exist";
            return;
        }
        std::stringstream buffer;
        std::ifstream ifile(load_file);
        buffer << ifile.rdbuf();
        utils::JsonObject cfg_graph_info = utils::JsonObject::parse(buffer.str());
        auto obj = cfg_graph_info.obj();
        ifile.close();
        for (auto cfg_node : obj)
        {
            // INFO() << "cfg node: " << cfg_node.first;
            auto cfg_info = cfg_node.second;
            int index = cfg_info["prior_in_gm"];
            if (index >= 0)
            {
                INFO() << "cfg node: " << cfg_node.first << " is priori in gm";
                prior_in_gm[cfg_node.first] = index;
            }
        }

        auto m_nodes = g->nodes();
        for (auto &&n : m_nodes)
        {
            auto it = prior_in_gm.find(n->name());
            if (it != prior_in_gm.end())
            {
                // n->set_attr("prior_in_gm", it->second);
                int index = it->second;
                auto t = n->outputs()[index];
                t->set_attr("prior_in_gm", true);
            }
        }
    }
    else if (BuilderConfig::get<bool>(Constants::dump_prior_in_gm))
    {
        utils::JsonObject node_info;
        utils::JsonObject graph_info;
        auto m_nodes = g->nodes();
        for (auto &&n : m_nodes)
        {

            // node_info["node_name"] = m_node->name();
            for (size_t i = 0; i < n->outputs().size(); ++i)
            {
                auto t = n->outputs()[i];
                if (t->attrs().count("prior_in_gm"))
                {
                    node_info["prior_in_gm"] = i;
                }
                else
                {
                    node_info["prior_in_gm"] = -1;
                }
            }
            graph_info[n->name()] = node_info;
        }
        std::ofstream os("prior_in_gm.json");
        if (os.is_open())
        {
            os << graph_info;
            os << std::endl;
            os.close();
        }
    }
    else if (BuilderConfig::get<bool>(Constants::softmax_prior_in_gm))
    {
        for (auto n : g->nodes())
        {
            // auto nodes = n->get_real_node();
            // if (nodes.size() != 1)
            // {
            //     continue;
            // }

            // if (nodes[0]->type() != OpType_Softmax)
            // {
            //     continue;
            // }
            // auto concat_node = n->children()[0];
            // auto matmul1 = concat_node->children()[0];
            // if (matmul1->type() != OpType_MatMul)
            // {
            //     continue;
            // }

            // if (concat_node->type() != OpType_Concat && concat_node->type() != OpType_InternalConcatenated)
            // {
            //     continue;
            // }

            BuilderOpPluginInfo *plugin = n->get_attr(Constants::plugin_attrs_name);
            if (plugin && plugin->name == "HwaSoftmax" && n->parents().size() == 1 && n->children().size() == 1)
            {
                // step 1: deal matmul0
                auto node_before_softmax = n->parents()[0];
                BuilderOpPluginInfo *plugin_m0 = node_before_softmax->get_attr(Constants::plugin_attrs_name);
                if (plugin_m0 && plugin_m0->name == "HwaCropMatmul")
                {
                    node_before_softmax->inputs()[0]->set_attr("prior_in_gm", true);
                    node_before_softmax->inputs()[1]->set_attr("prior_in_gm", true);
                    node_before_softmax->outputs()[0]->set_attr("prior_in_gm", true);
                }

                // step 2: deal matmul1
                auto node_after_softmax = n->children()[0];
                BuilderOpPluginInfo *plugin_m1 = node_after_softmax->get_attr(Constants::plugin_attrs_name);
                if (plugin_m1 && plugin_m1->name == "HwaMatmul")
                {
                    auto matmul1 = node_after_softmax;
                    if (matmul1->inputs()[0] == n->outputs()[0])
                    {
                        matmul1->inputs()[1]->set_attr("prior_in_gm", true);
                    }
                    else
                    {
                        matmul1->inputs()[0]->set_attr("prior_in_gm", true);
                    }

                    n->inputs()[0]->set_attr("prior_in_gm", true);
                    n->outputs()[0]->set_attr("prior_in_gm", true);
                }
                else if (node_after_softmax->type() == OpType_Concat || node_after_softmax->type() == OpType_InternalConcatenated)
                {
                    auto matmul1 = node_after_softmax->children()[0];
                    BuilderOpPluginInfo *plugin_2 = matmul1->get_attr(Constants::plugin_attrs_name);
                    if (plugin_2 && plugin_2->name == "HwaMatmul")
                    {
                        if (matmul1->inputs()[0] == node_after_softmax->outputs()[0])
                        {
                            matmul1->inputs()[1]->set_attr("prior_in_gm", true);
                        }
                        else
                        {
                            matmul1->inputs()[0]->set_attr("prior_in_gm", true);
                        }

                        n->inputs()[0]->set_attr("prior_in_gm", true);
                        n->outputs()[0]->set_attr("prior_in_gm", true);
                    }
                }

                // auto n_p = n->parents()[0];
                // BuilderOpPluginInfo *p_plugin = n_p->get_attr(Constants::plugin_attrs_name);

                // auto n_c = n->children()[0];
                // BuilderOpPluginInfo *c_plugin = n_c->get_attr(Constants::plugin_attrs_name);

                // if (p_plugin->name == "HwaMatmulBn" && c_plugin->name == "HwaMatmul")
                // {
                //     n->inputs()[0]->set_attr("prior_in_gm", true);
                //     n->outputs()[0]->set_attr("prior_in_gm", true);
                //     n_p->inputs()[1]->set_attr("prior_in_gm", true);
                //     n_c->inputs()[1]->set_attr("prior_in_gm", true);
                // }
            }
        }
    }
    return;
}

}; // namespace aipubt
