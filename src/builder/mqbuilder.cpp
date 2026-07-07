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
#include "builder/code_generator.h"
#include "builder/gop_plugin.h"
#include "builder/kernel_adapter.h"
#include "builder/load_balancer.h"
#include "builder/lowering/lowering.h"
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
    std::map<TensorPtr, uint32_t> tensor_index_map = m_graph->attrs()["tensor_index_map"];
    return tensor_index_map;
}

std::vector<BuildPhase> MqBuilder::create_phases()
{
    return {
        {"optimize", BuildStage::OPTIMIZE, [this]() { return phase_optimize(); }},
        {"schedule", BuildStage::SCHEDULE, [this]() { return phase_schedule(); }},
        {"kernel_adapt", BuildStage::KERNEL_ADAPT, [this]() { return phase_kernel_adapt(); }},
        {"mem_alloc", BuildStage::MEM_ALLOC, [this]() { return phase_mem_alloc(); }},
        {"code_gen", BuildStage::CODE_GEN, [this]() { return phase_code_gen(); }},
        {"assemble", BuildStage::ASSEMBLE, [this]() { return phase_assemble(); }},
        {"gen_obj", BuildStage::GEN_OBJ, [this]() { return phase_gen_obj(); }},
    };
}

bool MqBuilder::build()
{
    m_graph->global_tensors().clear();
    TensorPtr sp = std::make_shared<Tensor>("stack", TensorShape({8192}), Tensor::UINT8);
    m_graph->global_tensors()["stack"] = sp;
    TensorPtr print = std::make_shared<Tensor>("print", TensorShape({8192}), Tensor::UINT8);
    m_graph->global_tensors()["print"] = print;

    auto phases = create_phases();
    BuildStage save_stage = OPTIMIZE;
    BuildStage load_stage = (BuildStage)((int)save_stage + 1);

    for (const auto &phase : phases)
    {
        auto mq_load_ir = BuilderConfig::get<std::string>("mq_load_ir");
        if (!mq_load_ir.empty())
        {
            if (phase.stage < load_stage)
            {
                continue;
            }
            else if (phase.stage == load_stage)
            {
                m_graph = load_mq_ir_cache(mq_load_ir);
                if (!m_graph)
                {
                    ERROR() << "mq_load_ir: failed to load cached MQ IR from " << mq_load_ir;
                    return false;
                }
                if (!load_mq_ir_cache_attributes(m_graph, mq_load_ir))
                {
                    ERROR() << "mq_load_ir: failed to load cached MQ IR attributes from " << mq_load_ir;
                    return false;
                }
            }
        }

        INFO() << "====== executing phase: " << phase.name << " ======";
        if (!phase.execute())
        {
            ERROR() << "phase failed: " << phase.name;
            return false;
        }

        auto mq_save_ir = BuilderConfig::get<std::string>("mq_save_ir");
        if (!mq_save_ir.empty() && phase.stage == save_stage)
        {
            save_mq_ir_cache(m_graph, mq_save_ir);
            break;
        }
    }
    return true;
}

bool MqBuilder::phase_optimize()
{
    BuilderOptimizer optimizer;
    optimizer.optimize_mq(m_graph);
    if (!m_graph->check_valid_name())
    {
        ERROR() << "check valid name failed";
        return false;
    }

    return true;
}

bool MqBuilder::phase_schedule()
{
    PerfEstimator perf;
    perf(m_graph);

    LoadBalancer load_balancer;
    BuilderConfig::set(Constants::load_balance_strategy, 0);
    BuilderConfig::set(Constants::disable_dependency, true);
    int mq_num = BuilderConfig::get<int>(Constants::set_mq_num);
    load_balancer.set_core(mq_num);
    load_balancer.set_graph(m_graph);
    load_balancer.schedule();
    auto levels = load_balancer.get_levels();
    m_graph->set_attr("levels", levels);
    return true;
}

bool MqBuilder::phase_kernel_adapt()
{
    set_layer_id(m_graph);

    //kernel adaptor
    KernelAdapter kernel_adapter;
    kernel_adapter.set_graph(m_graph);
    kernel_adapter.kernel_adapt();
    auto const_tensors = kernel_adapter.get_const_tensors();
    m_graph->set_attr("const_tensors", const_tensors);
    auto ws_map = kernel_adapter.get_ws_map();
    m_graph->set_attr("ws_map", ws_map);
    auto surface_tensors = kernel_adapter.get_surface_tensors();
    m_graph->set_attr("surface_tensors", surface_tensors);
    auto kernel_lib_names = kernel_adapter.get_kernel_lib();
    m_graph->set_attr("kernel_lib_names", kernel_lib_names);
    return true;
}

bool MqBuilder::phase_mem_alloc()
{
    SOMSScheduler scheduler;
    scheduler.set_graph(m_graph);
    std::vector<std::vector<NodePtr>> levels = m_graph->attrs()["levels"];
    scheduler.set_levels(levels);
    scheduler.set_gm_size({0});
    std::map<NodePtr, std::set<TensorPtr>> ws_map = m_graph->attrs()["ws_map"];
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
    m_graph->set_attr("data_slots", data_slots);
    std::vector<ConstTensorPtr> const_tensors = m_graph->attrs()["const_tensors"];
    auto const_slots = scheduler.const_allocate(const_tensors);
    m_graph->set_attr("const_slots", const_slots);

    return true;
}

bool MqBuilder::phase_code_gen()
{
    CodeGenerator code_gen;
    std::map<std::string, std::string> kernel_lib_names = m_graph->attrs()["kernel_lib_names"];
    code_gen.set_kernel_lib(kernel_lib_names);
    code_gen.code_gen(m_elf);

    return true;
}

bool MqBuilder::phase_assemble()
{
    //Assembler
    Assembler assembler;
    assembler.set_graph(m_graph);
    assembler.set_elf(m_elf);
    std::vector<std::vector<NodePtr>> levels = m_graph->attrs()["levels"];
    assembler.set_level(levels);
    assembler.assembly();
    auto streams = assembler.get_streams();
    m_graph->set_attr("streams", streams);
    auto param_offset_table = assembler.get_param_offset_table();
    m_graph->set_attr("param_offset_table", param_offset_table);
    auto params = assembler.get_params();
    m_graph->set_attr("params", params);
    auto tensor_index_map = assembler.get_tensor_index_map();
    m_graph->set_attr("tensor_index_map", tensor_index_map);
    auto const_index_map = assembler.get_const_index_map();
    m_graph->set_attr("const_index_map", const_index_map);

    return true;
}

bool MqBuilder::phase_gen_obj()
{
    //gen elf
    ObjectGenerator object_generator;
    std::vector<std::vector<task_t>> streams = m_graph->attrs()["streams"];
    object_generator.set_streams(streams);
    std::map<uint32_t, uint32_t> param_offset_table = m_graph->attrs()["param_offset_table"];
    object_generator.set_param_offset_table(param_offset_table);
    BuilderParams params = m_graph->attrs()["params"];
    object_generator.set_params(params);
    std::map<TensorPtr, uint32_t> tensor_index_map = m_graph->attrs()["tensor_index_map"];
    object_generator.set_tensor_index_map(tensor_index_map);
    std::map<ConstTensorPtr, uint32_t> const_index_map = m_graph->attrs()["const_index_map"];
    object_generator.set_const_index_map(const_index_map);
    std::vector<MemoryObjectPtr> data_slots = m_graph->attrs()["data_slots"];
    object_generator.set_slots(data_slots);
    object_generator.set_graph(m_graph);
    std::vector<ConstTensorPtr> const_tensors = m_graph->attrs()["const_tensors"];
    object_generator.set_constant(const_tensors);
    object_generator.gen(m_elf);

    return true;
}

MqBuilder::~MqBuilder()
{
}
} // namespace aipubt
