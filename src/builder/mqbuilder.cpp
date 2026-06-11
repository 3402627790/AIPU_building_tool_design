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
 * @Date: 2026-04-09 14:23:41
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2026-04-09 14:23:41
 */

#include "builder/mqbuilder.h"
#include "builder/assembler.h"
#include "builder/builder.h"
#include "builder/code_generator.h"
#include "builder/gop_plugin.h"
#include "builder/kernel_adapter.h"
#include "builder/load_balancer.h"
#include "builder/mq_ir_cache.h"
#include "builder/object_generator.h"
#include "builder/optimizer.h"
#include "builder/soms_scheduler.h"
#include "core/config.h"
#include "core/graph.h"
#include "core/tensor.h"
#include "statistic/perf_estimator.h"
#include "transform/passes.h"

namespace aipubt
{

GraphPtr load_mq_ir_cache(const std::string &dir);
void save_mq_ir_cache(GraphPtr graph, const std::string &dir);

MqBuilder::MqBuilder(GraphPtr graph) : m_graph(graph)
{
    m_elf = std::make_shared<CpsElf>();
}

GraphPtr MqBuilder::get_graph()
{
    return m_graph;
}

CpsElfPtr MqBuilder::get_elf()
{
    return m_elf;
}

std::map<TensorPtr, uint32_t> MqBuilder::get_tensor_index_map()
{
    return m_tensor_index_map;
}

bool MqBuilder::build()
{
    bool loaded_from_cache = false;
    auto mq_load_ir = BuilderConfig::get<std::string>("mq_load_ir");
    if (!mq_load_ir.empty())
    {
        auto cached_graph = load_mq_ir_cache(mq_load_ir);
        if (!cached_graph)
        {
            ERROR() << "mq_load_ir: failed to load cache from " << mq_load_ir;
            return false;
        }
        else
        {
            m_graph = cached_graph;
            loaded_from_cache = true;
        }
    }

    m_graph->global_tensors().clear();
    TensorPtr sp = std::make_shared<Tensor>("stack", TensorShape({8192}), Tensor::UINT8);
    m_graph->global_tensors()["stack"] = sp;
    TensorPtr print = std::make_shared<Tensor>("print", TensorShape({8192}), Tensor::UINT8);
    m_graph->global_tensors()["print"] = print;
    TensorPtr param = std::make_shared<Tensor>("param", TensorShape({0}), Tensor::UINT8);
    m_graph->global_tensors()["param"] = param;
    TensorPtr stream0 = std::make_shared<Tensor>("stream0", TensorShape({0}), Tensor::UINT8);
    m_graph->global_tensors()["stream0"] = stream0;
    TensorPtr stream1 = std::make_shared<Tensor>("stream1", TensorShape({0}), Tensor::UINT8);
    m_graph->global_tensors()["stream1"] = stream1;
    TensorPtr stream2 = std::make_shared<Tensor>("stream2", TensorShape({0}), Tensor::UINT8);
    m_graph->global_tensors()["stream2"] = stream2;
    TensorPtr stream3 = std::make_shared<Tensor>("stream3", TensorShape({0}), Tensor::UINT8);
    m_graph->global_tensors()["stream3"] = stream3;

    config_dynamic_shape(m_graph);

    if (!loaded_from_cache)
    {
        BuilderOptimizer optimizer;
        optimizer.optimize_mq(m_graph);
        if (!m_graph->check_valid_name())
        {
            ERROR() << "check valid name failed";
        }

        auto mq_save_ir = BuilderConfig::get<std::string>("mq_save_ir");
        if (!mq_save_ir.empty())
        {
            save_mq_ir_cache(m_graph, mq_save_ir);
        }
    }

    PerfEstimator perf;
    perf(m_graph);

    aipubt::BuilderConfig::set(Constants::single_core_opt, true);
    LoadBalancer load_balancer;
    load_balancer.set_graph(m_graph);
    load_balancer.schedule();
    auto levels = load_balancer.get_levels();

    set_layer_id(m_graph);

    //kernel adaptor
    KernelAdapter kernel_adapter;
    kernel_adapter.set_graph(m_graph);
    kernel_adapter.kernel_adapt();
    auto const_tensors = kernel_adapter.get_const_tensors();
    auto ws_map = kernel_adapter.get_ws_map();
    auto surface_tensors = kernel_adapter.get_surface_tensors();
    auto kernel_lib_names = kernel_adapter.get_kernel_lib();

    SOMSScheduler scheduler;
    scheduler.set_graph(m_graph);
    scheduler.set_levels(levels);
    scheduler.set_gm_size({0});
    scheduler.set_ws(ws_map);
    scheduler.disable_sm();
    std::set<TensorPtr> not_reuse_tensor;
    std::set<TensorPtr> skip_tensors;
    std::set<TensorPtr> input_tensors;
    std::set<TensorPtr> output_tensors;
    for (auto t : m_graph->get_input_tensors())
    {
        input_tensors.insert(t);
    }
    for (auto t : m_graph->get_output_tensors())
    {
        output_tensors.insert(t);
    }
    for (auto gt : m_graph->global_tensors())
    {
        if (gt.second && gt.second->mem_size())
        {
            not_reuse_tensor.insert(gt.second);
        }
    }
    if (BuilderConfig::get<bool>("dump_all_tensors"))
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
                    for (auto out : n->outputs())
                    {
                        if (!contains(output_tensors, out) && !contains(input_tensors, out))
                        {
                            output_tensors.insert(out);
                        }
                    }
                }
            }
        }
    }

    auto data_slots = scheduler.mem_allocate(not_reuse_tensor, input_tensors, output_tensors, skip_tensors);
    for (unsigned int i = 0; i < data_slots.size(); ++i)
    {
        data_slots[i]->buffer_id = i;
    }
    //TODO constant allocate
    for (auto t : const_tensors)
    {
        INFO() << "constant tensor: " << t->name();
    }
    auto const_slots = scheduler.const_allocate(const_tensors);

    //code gen
    CodeGenerator code_gen;
    code_gen.set_kernel_lib(kernel_lib_names);
    code_gen.code_gen(m_elf);

    //Assembler
    Assembler assembler;
    assembler.set_graph(m_graph);
    assembler.set_elf(m_elf);
    assembler.assembly();
    auto streams = assembler.get_streams();
    auto param_offset_table = assembler.get_param_offset_table();
    auto &params = assembler.get_params();
    m_tensor_index_map = assembler.get_tensor_index_map();
    auto const_index_map = assembler.get_const_index_map();

    //gen elf
    ObjectGenerator object_generator;
    object_generator.set_streams(streams);
    object_generator.set_param_offset_table(param_offset_table);
    object_generator.set_params(params);
    object_generator.set_tensor_index_map(m_tensor_index_map);
    object_generator.set_const_index_map(const_index_map);
    object_generator.set_slots(data_slots);
    object_generator.set_graph(m_graph);
    object_generator.set_constant(const_tensors);
    object_generator.gen(m_elf);

    return true;
}

MqBuilder::~MqBuilder()
{
}
} // namespace aipubt
