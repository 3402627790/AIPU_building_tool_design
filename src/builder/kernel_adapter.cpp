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
 * @Date: 2026-04-24 14:27:33
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2026-04-24 14:27:33
 */
#include "builder/kernel_adapter.h"
#include "builder/ds_code_generator_cx.h"
#include "core/config.h"

namespace aipubt
{

bool KernelAdapter::set_graph(GraphPtr g)
{
    m_graph = g;
    if (!g)
    {
        return false;
    }
    return true;
}

void KernelAdapter::record_tensors(NodePtr n, const BuilderParams &params)
{
    for (auto &p : params)
    {
        if (p.const_tensor && !contains(m_const_tensors, p.const_tensor)) //is constant
        {
            m_const_tensors.push_back(p.const_tensor);
        }
        else if (p.tensor && p.tensor->attrs().count(Constants::is_workspace_tensor)) //is workspace
        {
            m_ws_map[n].insert(p.tensor);
        }
        else if (p.surface)
        {
            m_surface_tensors[p.surface] = n;
        }
    }
}

void KernelAdapter::kernel_adapt()
{
    m_graph->walk_on_node([](NodePtr n) {
        if (n->attrs().count(Constants::plugin_impl))
        {
            auto impl = n->get_attr<GopImplPtr>(Constants::plugin_impl);
            auto infos = impl->kernel_infos();
            std::vector<PlaceholderPtr> param_bases;
            std::vector<PlaceholderPtr> stream_bases;
            for (size_t i = 0; i < infos.size(); ++i)
            {
                param_bases.push_back(std::make_shared<Placeholder>());
                stream_bases.push_back(std::make_shared<Placeholder>());
            }
            n->set_attr(Constants::kernel_infos, infos);
            n->set_attr(Constants::param_bases, param_bases);
            n->set_attr(Constants::stream_bases, stream_bases);
        }
    });

    if (BuilderConfig::get<bool>(Constants::enable_ds))
    {
        DSCodeGeneratorCPS code_gen;
        code_gen.generate(m_graph);
    }

    for (auto s : m_graph->nodes())
    {
        auto sgn = SubGraphNode::from_node(s);
        if (sgn)
        {
            auto sg = sgn->subgraph();
            for (auto n : sg->nodes())
            {
                if (n->type() == OpType_Input || n->type() == OpType_SubGraphInput)
                {
                    continue;
                }

                if (n->attrs().count(Constants::plugin_impl))
                {
                    auto plugin_impl = n->get_attr<GopImplPtr>(Constants::plugin_impl);
                    auto kernel_infos = n->get_attr<KernelInfos>(Constants::kernel_infos);
                    auto lib_name = plugin_impl->kernel_cpsbin();
                    if (lib_name == "GB_DEFAULT_PATH")
                    {
                        ERROR();
                    }
                    else if (lib_name == "")
                    {
                        lib_name = "GB_DEFAULT_PATH";
                    }
                    for (auto info : kernel_infos)
                    {
                        INFO() << "kernel for " << n << " " << info.kernel_name;
                        m_kernel_lib[info.kernel_name] = lib_name;
                        record_tensors(n, info.args);

                        for (auto &arg : info.args)
                        {
                            if (arg.param_list)
                            {
                                auto &params = arg.param_list->params();
                                for (auto &p : params)
                                {
                                    if (p.param_list || p.surface)
                                    {
                                        ERROR() << "paramlist cannot contain paramlist or surface";
                                    }
                                }
                                record_tensors(n, params);
                            }
                        }
                    }
                }
                else
                {
                    ERROR() << n << " has no plugin_impl";
                }
            }
        }
    }
}

std::vector<ConstTensorPtr> KernelAdapter::get_const_tensors()
{
    return m_const_tensors;
}

std::map<NodePtr, std::set<TensorPtr>> KernelAdapter::get_ws_map()
{
    return m_ws_map;
}

std::map<TensorPtr, NodePtr> KernelAdapter::get_surface_tensors()
{
    return m_surface_tensors;
}

std::map<std::string, std::string> KernelAdapter::get_kernel_lib()
{
    return m_kernel_lib;
}

} // namespace aipubt