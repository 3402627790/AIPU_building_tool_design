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

namespace aipubt
{

bool KernelAdapter::set_graph(GraphPtr g)
{
    m_graph = g;
    if (!g)
    {
        return true;
    }
    return false;
}
void KernelAdapter::kernel_adapt()
{
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
                    auto kernel_infos = plugin_impl->kernel_infos();
                    n->set_attr(Constants::kernel_infos, kernel_infos);
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
                        // kernel_names.insert(info.kernel_name);
                        m_kernel_lib[info.kernel_name] = lib_name;
                        for (auto arg : info.args)
                        {
                            if (arg.const_tensor) //is constant
                            {
                                m_const_tensors.insert(arg.const_tensor);
                            }
                            else if (arg.tensor && arg.tensor->attrs().count(Constants::is_workspace_tensor)) //is workspace
                            {
                                m_ws_tensors[arg.tensor] = n;
                            }
                            else if (arg.surface)
                            {
                                m_surface_tensors[arg.surface] = n;
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
std::set<ConstTensorPtr> KernelAdapter::get_const_tensors()
{
    return m_const_tensors;
}

std::map<TensorPtr, NodePtr> KernelAdapter::get_ws_tensors()
{
    return m_ws_tensors;
}

std::map<TensorPtr, NodePtr> KernelAdapter::get_surface_tensors()
{
    return m_surface_tensors;
}

std::map<std::string, std::string> KernelAdapter::get_kernel_lib()
{
    return m_kernel_lib;
}

}; // namespace aipubt