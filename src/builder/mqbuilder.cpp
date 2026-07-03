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

#include <cctype>

namespace aipubt
{
namespace
{

void reset_mq_global_tensors(GraphPtr m_graph)
{
    if (!m_graph)
    {
        return;
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
}

bool parse_mq_ir_stage(const std::string &stage_name, BuildStage &stage)
{
    std::string normalized_stage;
    for (auto c : stage_name)
    {
        if (c != '\r' && c != '\n')
        {
            normalized_stage.push_back((char)std::tolower((unsigned char)c));
        }
    }

    if (normalized_stage.empty() || normalized_stage == "optimize")
    {
        stage = BuildStage::OPTIMIZE;
        return true;
    }
    if (normalized_stage == "schedule")
    {
        stage = BuildStage::SCHEDULE;
        return true;
    }

    ERROR() << "unsupported mq_ir_stage: " << stage_name << ", expected optimize or schedule";
    return false;
}

std::string mq_ir_stage_name(BuildStage stage)
{
    if (stage == BuildStage::SCHEDULE)
    {
        return "schedule";
    }
    return "optimize";
}

} // namespace

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
    return m_cache.tensor_index_map;
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
    reset_mq_global_tensors(m_graph);

    auto phases = create_phases();
    auto mq_save_ir = BuilderConfig::get<std::string>(Constants::mq_save_ir);
    auto mq_load_ir = BuilderConfig::get<std::string>(Constants::mq_load_ir);

    BuildStage save_stage = BuildStage::OPTIMIZE;
    if (BuilderConfig::exist(Constants::mq_ir_stage) &&
        !parse_mq_ir_stage(BuilderConfig::get<std::string>(Constants::mq_ir_stage), save_stage))
    {
        return false;
    }

    BuildStage cache_stage = save_stage;
    if (!mq_load_ir.empty())
    {
        std::string stage_name = BuilderConfig::exist(Constants::mq_ir_stage) ?
                                     BuilderConfig::get<std::string>(Constants::mq_ir_stage) :
                                     load_mq_ir_cache_stage(mq_load_ir);
        if (!parse_mq_ir_stage(stage_name, cache_stage))
        {
            return false;
        }
    }
    BuildStage load_stage = (BuildStage)((int)cache_stage + 1);

    for (const auto &phase : phases)
    {
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
                reset_mq_global_tensors(m_graph);
                if (cache_stage == BuildStage::SCHEDULE &&
                    !load_mq_ir_cache_schedule(m_graph, mq_load_ir, m_cache.levels))
                {
                    ERROR() << "mq_load_ir: failed to load cached MQ schedule from " << mq_load_ir;
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

        if (!mq_save_ir.empty() && phase.stage == save_stage)
        {
            save_mq_ir_cache(m_graph, mq_save_ir, mq_ir_stage_name(save_stage));
            if (save_stage == BuildStage::SCHEDULE && !save_mq_ir_cache_schedule(m_cache.levels, mq_save_ir))
            {
                ERROR() << "mq_save_ir: failed to save MQ schedule to " << mq_save_ir;
                return false;
            }
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
    m_cache.levels = load_balancer.get_levels();

    return true;
}

bool MqBuilder::phase_kernel_adapt()
{
    set_layer_id(m_graph);

    //kernel adaptor
    KernelAdapter kernel_adapter;
    kernel_adapter.set_graph(m_graph);
    kernel_adapter.kernel_adapt();
    m_cache.const_tensors = kernel_adapter.get_const_tensors();
    m_cache.ws_map = kernel_adapter.get_ws_map();
    m_cache.surface_tensors = kernel_adapter.get_surface_tensors();
    m_cache.kernel_lib_names = kernel_adapter.get_kernel_lib();

    return true;
}

bool MqBuilder::phase_mem_alloc()
{
    SOMSScheduler scheduler;
    scheduler.set_graph(m_graph);
    scheduler.set_levels(m_cache.levels);
    scheduler.set_gm_size({0});
    scheduler.set_ws(m_cache.ws_map);
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

    m_cache.data_slots = scheduler.mem_allocate(not_reuse_tensor, input_tensors, output_tensors, skip_tensors);
    for (unsigned int i = 0; i < m_cache.data_slots.size(); ++i)
    {
        m_cache.data_slots[i]->buffer_id = i;
    }

    // for (auto t : m_cache.const_tensors)
    // {
    //     INFO() << "constant tensor: " << t->name();
    // }
    m_cache.const_slots = scheduler.const_allocate(m_cache.const_tensors);

    return true;
}

bool MqBuilder::phase_code_gen()
{
    CodeGenerator code_gen;
    code_gen.set_kernel_lib(m_cache.kernel_lib_names);
    code_gen.code_gen(m_elf);

    return true;
}

bool MqBuilder::phase_assemble()
{
    //Assembler
    Assembler assembler;
    assembler.set_graph(m_graph);
    assembler.set_elf(m_elf);
    assembler.set_level(m_cache.levels);
    assembler.assembly();
    m_cache.streams = assembler.get_streams();
    m_cache.param_offset_table = assembler.get_param_offset_table();
    m_cache.params = assembler.get_params();
    m_cache.tensor_index_map = assembler.get_tensor_index_map();
    m_cache.const_index_map = assembler.get_const_index_map();

    return true;
}

bool MqBuilder::phase_gen_obj()
{
    //gen elf
    ObjectGenerator object_generator;
    object_generator.set_streams(m_cache.streams);
    object_generator.set_param_offset_table(m_cache.param_offset_table);
    object_generator.set_params(m_cache.params);
    object_generator.set_tensor_index_map(m_cache.tensor_index_map);
    object_generator.set_const_index_map(m_cache.const_index_map);
    object_generator.set_slots(m_cache.data_slots);
    object_generator.set_graph(m_graph);
    object_generator.set_constant(m_cache.const_tensors);
    object_generator.gen(m_elf);

    return true;
}

MqBuilder::~MqBuilder()
{
}
} // namespace aipubt
