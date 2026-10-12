//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020-2026 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Kyle Huang
 * @Date: 2026-04-24 14:27:29
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2026-04-24 14:27:29
 * */
#ifndef BUILDER_KERNEL_ADAPTOR_H
#define BUILDER_KERNEL_ADAPTOR_H

#include "builder/gop_plugin.h"
#include "core/constants.h"
#include "core/graph.h"
#include "core/node.h"
#include "core/plugin.h"
#include "core/subgraph.h"

namespace aipubt
{
class KernelAdapter
{
public:
    bool set_graph(GraphPtr g);
    void kernel_adapt();
    std::vector<TensorPtr> get_const_tensors();
    std::map<NodePtr, std::set<TensorPtr>> get_ws_map();
    const std::map<std::string, std::string> &get_device_lib() const;
    const std::map<std::string, std::string> &get_host_lib() const;

private:
    void record_tensors(NodePtr n, const BuilderParams &params);

    GraphPtr m_graph;
    std::vector<TensorPtr> m_const_tensors;
    std::map<NodePtr, std::set<TensorPtr>> m_ws_map;
    std::map<std::string, std::string> m_device_lib;
    std::map<std::string, std::string> m_host_lib;
};
} // namespace aipubt
#endif //!BUILDER_KERNEL_ADAPTOR_H