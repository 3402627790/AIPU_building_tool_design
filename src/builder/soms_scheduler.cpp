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
 * @Date: 2024-07-15 14:27:29
 * @Last Modified by: kyle.huang
 * @Last Modified time: 2024-08-15 10:57:30
 */
#include "soms_scheduler.h"
#include "builder/lowering/hash_constant.h"
#include "builder/target.h"
#include "core/config.h"
#include "sram_optimize.h"
#include <algorithm>
#include <chrono>
#include <set>
#include <unordered_set>
#include <vector>
namespace aipubt
{
inline size_t aligned(size_t s, size_t align = 256)
{
    return (s + align - 1) & ~(align - 1);
}

// static NodePtr find_actual_output(NodePtr n)
// {
//     auto g = n->graph();
//     auto sg = std::dynamic_pointer_cast<SubGraph>(g);
//     auto s = sg->subgraphnode();
//     auto t = n->outputs()[0];
//     auto t_p = std::dynamic_pointer_cast<ProxyTensor>(t);
//     TensorPtr s_input;
//     if (t_p)
//     {
//         auto proxied_t_p = t_p->proxied_tensor();
//         for (auto i : s->inputs())
//         {
//             if (i == proxied_t_p)
//             {
//                 s_input = i;
//             }
//         }
//     }
//     else
//     {
//         ERROR();
//     }

//     NodePtr ret;
//     auto p_stream = s_input->node();
//     if (p_stream)
//     {
//         auto sgn = SubGraphNode::from_node(p_stream);

//         if (sgn)
//         {
//             auto sg = sgn->subgraph();
//             for (auto n : sg->nodes())
//             {
//                 for (auto o : n->outputs())
//                 {
//                     auto o_p = std::dynamic_pointer_cast<ProxyTensor>(o);
//                     if (o_p)
//                     {
//                         auto proxied_o_p = o_p->proxied_tensor();
//                         if (s_input == proxied_o_p)
//                         {
//                             ret = n;
//                         }
//                     }
//                 }
//             }
//         }
//         else
//         {
//             ERROR();
//         }
//     }
//     else
//     {
//         ERROR();
//     }

//     if (!ret)
//     {
//         ERROR();
//     }
//     return ret;
// }

int SOMSScheduler::get_mem_id(TensorPtr t)
{
    auto slice = std::dynamic_pointer_cast<TensorSlice>(t);
    if (slice)
    {
        return slice->mem_id();
    }
    return -1;
}

std::vector<MemoryObjectPtr> SOMSScheduler::const_allocate(const std::vector<TensorPtr> &const_tensor)
{
    auto get_mem_size = [](TensorPtr t) -> uint64_t {
        auto ptr = t->memory_offset()->get_base_object();
        auto size = std::max(ptr->size, (uint64_t)t->mem_size());
        return size;
    };

    //reduce_constant
    std::map<std::string, ConstTensorPtr> constant_hash;
    uint64_t reduce_size = 0;
    for (auto constant : const_tensor)
    {
        std::string hash = utils::sha256_hex(constant->data<void>(), constant->mem_size());
        if (constant_hash.count(hash))
        {
            auto same_const = constant_hash[hash];
            reduce_size += constant->mem_size();
            same_const->memory_offset()->foot_print += constant->memory_offset()->foot_print;
            constant->memory_offset()->foot_print = same_const->memory_offset()->foot_print;
            // constant->share_data_from(*constant_hash[hex]);
            constant->memory_offset()->set_base_offset(same_const->memory_offset()->get_base_object());
        }
        else
        {
            constant_hash[hash] = constant;
        }
    }
    if (reduce_size > 0)
    {
        INFO() << "Reduce constants memory size: " << utils::get_human_readable_size(reduce_size);
    }

    std::set<MemoryObjectPtr> mem_ptr;
    std::map<MemoryObjectPtr, size_t> memory_size;
    std::map<MemoryObjectPtr, std::set<MemoryObjectPtr>> slot_map;
    for (auto constant : const_tensor)
    {
        auto ptr = constant->memory_offset()->get_base_object();
        if (ptr->m_base)
        {
            continue; //allocate root blob once for shared blobs
        }
        auto id = get_mem_id(constant);
        ptr->set_mem_id(id);
        mem_ptr.insert(ptr);
        memory_size[ptr] = get_mem_size(constant);
    }

    //allocate_constant
    std::set<MemoryObjectPtr> assigned_ptr;
    std::set<MemoryObjectPtr> unassigned_ptr;
    std::set<MemoryObjectPtr> total_assigned_ptr;
    std::map<MemoryObjectPtr, uint64_t> offset_map;
    if (m_numa_cfg.mem_size().size() > 0)
    {
        for (size_t i = 0; i < m_numa_cfg.mem_size().size(); ++i)
        {
            MemoryObjectPtr numa_s = std::make_shared<MemoryObject>(0);
            numa_s->size = m_numa_cfg.mem_size()[i];
            if (numa_s->size == 0)
            {
                continue;
            }
            numa_s->align = 0;
            numa_s->set_mem_id(i);
            auto used_size = offset_assignment(numa_s, memory_size, std::map<MemoryObjectPtr, std::vector<MemoryObjectPtr>>(), orderbydefault, true, 1, Linear, mem_ptr, assigned_ptr, unassigned_ptr, offset_map, total_assigned_ptr);
            if (used_size)
            {
                numa_s->size = used_size;
                slot_map[numa_s] = assigned_ptr;
            }
            mem_ptr = unassigned_ptr;
            assigned_ptr.clear();
            unassigned_ptr.clear();
        }
    }

    MemoryObjectPtr const_s = std::make_shared<MemoryObject>(0);
    const_s->size = 0xe0000000;
    const_s->align = 0;
    const_s->set_mem_id(-1);
    const_s->set_buf_type(MemoryObject::type::constant);
    auto used_size = offset_assignment(const_s, memory_size, std::map<MemoryObjectPtr, std::vector<MemoryObjectPtr>>(), orderbydefault, true, 1, Linear, mem_ptr, assigned_ptr, unassigned_ptr, offset_map, total_assigned_ptr);
    if (used_size)
    {
        const_s->size = used_size;
        slot_map[const_s] = assigned_ptr;
    }
    if (unassigned_ptr.size())
    {
        ERROR();
    }

    int cnt = 0;
    std::vector<MemoryObjectPtr> slots;
    for (auto s : slot_map)
    {
        // DEBUG() << "slot ";
        if (s.first->size == 0)
        {
            ERROR() << "slot size is empty";
        }
        slots.push_back(s.first);
        s.first->foot_print = 0;
        s.first->buffer_id = cnt++;
        for (auto m : s.second)
        {
            // DEBUG() << at.t->name() << " offset " << at.offset << " size " << at.size;
            auto ptr = m;
            ptr->set_base_offset(s.first);
            ptr->relative_offset = offset_map[m];
            s.first->foot_print += ptr->foot_print;
        }
    }

    return slots;
}

std::vector<MemoryObjectPtr> SOMSScheduler::mem_allocate(const std::set<TensorPtr> &not_reuse_tensor, const std::set<TensorPtr> &inputs, const std::set<TensorPtr> &keeps_tensors, const std::set<TensorPtr> &skip_tensors)
{
    if (!m_graph)
    {
        return std::vector<MemoryObjectPtr>();
    }

    auto start = std::chrono::high_resolution_clock::now();

    if (BuilderConfig::get<std::string>(Constants::load_prior_in_gm) != "" || BuilderConfig::get<bool>(Constants::softmax_prior_in_gm))
    {
        m_need_prior_gm = true;
    }

    for (auto &pair : m_ws_map)
    {
        if (m_need_prior_gm)
        {
            break;
        }
        for (auto &t : pair.second)
        {
            if (t->is_prior_tensor())
            {
                m_need_prior_gm = true;
                break;
            }
        }
    }

    // for (auto n : m_graph->nodes())
    // {
    //     auto sgn = SubGraphNode::from_node(n);
    //     if (sgn)
    //     {
    //         auto sg = sgn->subgraph();
    //         for (auto sg_n : sg->nodes())
    //         {
    //             if (sg_n->type() == OpType_SubGraphInput)
    //             {
    //                 auto out = find_actual_output(sg_n);
    //                 sg_n->set_attr(Constants::proxy_node, NodeRef(out));
    //             }
    //         }
    //     }
    // }

    std::set<MemoryObjectPtr> skip_mem;
    for (auto t : skip_tensors)
    {
        // INFO() << "skip tensors: " << t->name();
        auto ptr = t->memory_offset()->get_base_object();
        skip_mem.insert(ptr);
        // tensor_map[ptr] = t;
    }

    // assign sm
    if (m_enable_sm)
    {
        auto sm_mem = allocate_sm(skip_mem);
        for (auto m : sm_mem)
        {
            if (!contains(skip_mem, m)) //may have skip tensors
            {
                skip_mem.insert(m);
                // tensor_map[ptr] = t;
            }
        }
    }
    // std::map<MemoryObjectPtr, TensorPtr> tensor_map; //for debug only

    std::set<MemoryObjectPtr> not_reuse_mem;
    for (auto t : not_reuse_tensor)
    {
        auto ptr = t->memory_offset()->get_base_object();
        if (!contains(skip_mem, ptr)) //may have skip tensors
        {
            skip_mem.insert(ptr);
            not_reuse_mem.insert(ptr);
            // tensor_map[ptr] = t;
            m_tensor_map[ptr] = t;
        }
    }

    for (auto t : inputs)
    {
        auto ptr = t->memory_offset()->get_base_object();
        if (contains(skip_mem, ptr)) //may have skip tensors
        {
            // WARN() << "skip mem! " << t->name();
            continue;
        }
        m_input_mem.insert(ptr);
    }

    for (auto t : keeps_tensors)
    {
        auto ptr = t->memory_offset()->get_base_object();
        if (contains(skip_mem, ptr)) //may have skip tensors
        {
            // WARN() << "skip mem! " << t->name();
            continue;
        }
        m_keep_mem.insert(ptr);
    }

    auto all_mems = mem_live_init(skip_mem);

    auto ti1 = std::chrono::high_resolution_clock::now();
    auto du1 = std::chrono::duration_cast<std::chrono::microseconds>(ti1 - start);
    INFO() << "[EVAL] init time t1: " << du1.count() << " us";

    auto conflict_map = gen_mem_conflict(all_mems);

    auto ti2 = std::chrono::high_resolution_clock::now();
    auto du2 = std::chrono::duration_cast<std::chrono::microseconds>(ti2 - ti1);
    INFO() << "[EVAL] unsafe check time t2: " << du2.count() << " us";

    m_slots = mem_assignment(conflict_map, all_mems, not_reuse_mem);

    //offset allocation for many object
    auto ti3 = std::chrono::high_resolution_clock::now();
    auto du3 = std::chrono::duration_cast<std::chrono::microseconds>(ti3 - ti2);
    INFO() << "[EVAL] mem assignment time t3: " << du3.count() << " us";

    // for private buffer like workspace, we need set the attributes for later bss generation
    for (auto n : m_graph->nodes())
    {
        auto sgn = SubGraphNode::from_node(n);
        if (sgn)
        {
            auto ws = sgn->subgraph()->global_tensors()[Constants::workspace];
            ws->memory_offset()->base_offset = nullptr;
            {
                std::vector<MemoryObjectPtr> pb;
                pb.push_back(ws->memory_offset()->get_base_object());
                n->attrs()[Constants::private_buffers] = pb;
            }
        }
    }

    // INFO() << "footprint of gm: " << utils::get_human_readable_size(gm_s->foot_print);
    // exit(0);

    return m_slots;
}

void SOMSScheduler::update_sm(GraphPtr g, std::set<MemoryObjectPtr> &sm_mem)
{
    for (auto node : g->nodes())
    {
        if (node->type() == OpType_Loop || node->type() == OpType_If)
        {
            auto sgn = SubGraphNode::from_node(node);
            update_sm(sgn->subgraph(), sm_mem);
        }

        for (auto t : node->outputs())
        {
            auto ptr = t->memory_offset()->get_base_object();
            if (ptr->m_base == 0xf8000000)
            {
                // INFO() << t->name() << " " << t->node() << " is in sram";
                sm_mem.insert(ptr);
            }
        }
    }
}

void SOMSScheduler::sm_skip_tensors(GraphPtr g, std::set<TensorPtr> &skip_tensors, const std::set<MemoryObjectPtr> &skip_mem)
{
    for (auto node : g->nodes())
    {
        if (node->type() == OpType_Loop || node->type() == OpType_If)
        {
            auto sgn = SubGraphNode::from_node(node);
            sm_skip_tensors(sgn->subgraph(), skip_tensors, skip_mem);
        }
        for (auto t : node->outputs())
        {
            auto ptr = t->memory_offset()->get_base_object();
            if (contains(skip_mem, ptr))
            {
                skip_tensors.insert(t);
            }
        }
    }
}

std::set<MemoryObjectPtr> SOMSScheduler::allocate_sm(std::set<MemoryObjectPtr> &skip_mem)
{
    std::set<MemoryObjectPtr> sm_mem;
    for (auto level : m_levels)
    {
        for (auto n : level)
        {
            std::set<TensorPtr> sg_keeps_tensors;
            auto sgn = std::static_pointer_cast<SubGraphNode>(n);
            for (auto node : sgn->subgraph()->nodes())
            {
                for (auto o : node->outputs())
                {
                    auto o_p = std::dynamic_pointer_cast<ProxyTensor>(o);
                    if (o_p)
                    {
                        sg_keeps_tensors.insert(o);
                        // INFO() << "keep tensors: " << o->name();
                    }
                }
            }

            std::set<TensorPtr> skip_tensors;
            sm_skip_tensors(sgn->subgraph(), skip_tensors, skip_mem);

            auto reserve_orders = sram_optimize(sgn->subgraph(), sg_keeps_tensors, skip_tensors);
            if (reserve_orders.size())
            {
                sgn->subgraph()->topological_sort(reserve_orders);
            }

            update_sm(sgn->subgraph(), sm_mem);
        }
    }
    return sm_mem;
}

NodePtr SOMSScheduler::update_gen_consume(GraphPtr g, int &node_stamp, std::set<MemoryObjectPtr> &all_mems, const std::set<MemoryObjectPtr> &skip_mem)
{
    auto update = [this, &skip_mem, &all_mems](TensorPtr t, NodePtr generator, std::vector<NodePtr> consumers, int mem_id) -> void {
        auto ptr = t->memory_offset()->get_base_object();
        // INFO() << "SG tensor " <<t->name() << " node " << t->node()->name() << " ptr " << ptr;
        if (generator->type() != OpType_SubGraphInput)
        {
            if (generator->type() != OpType_Loop && generator->type() != OpType_If)
            {
                m_p_generators[ptr].push_back(generator);
            }
        }
        for (auto c : consumers)
        {
            if (c->type() != OpType_Loop && c->type() != OpType_If)
            {
                m_p_consumers[ptr].push_back(c);
            }
        }
        if (t->is_prior_tensor())
        {
            ptr->set_mem_flag(MemoryObject::flag::prior_in_gm);
            INFO() << "prior in gm " << t->name();
        }
        if (!contains(skip_mem, ptr))
        {
            all_mems.insert(ptr);
            m_tensor_map[ptr] = t;
        }
        ptr->set_mem_id(mem_id);
    };

    NodePtr last_node;
    for (auto n : g->nodes())
    {
        // INFO() << "node stamp: " << n->name() << " is " << node_stamp;
        n->set_attr(Constants::mem_clock, node_stamp);
        last_node = n;
        node_stamp = node_stamp + 1;
        if (n->type() == OpType_Loop || n->type() == OpType_If)
        {
            auto sgn = SubGraphNode::from_node(n);
            last_node = update_gen_consume(sgn->subgraph(), node_stamp, all_mems, skip_mem);

            for (auto t : n->outputs())
            {
                auto ptr = t->memory_offset()->get_base_object();
                if (ptr->get_attr() == MemoryObject::attr::loop_scan)
                {
                    // INFO() << "loop scan generator of " << t->name() << " is " << n->name();
                    m_p_generators[ptr].push_back(n);
                }
            }

            for (auto t : n->inputs())
            {
                auto ptr = t->memory_offset()->get_base_object();
                if (ptr->get_attr() == MemoryObject::attr::loop_scan)
                {
                    if (!last_node)
                    {
                        ERROR();
                    }
                    // INFO() << "loop scan consumer of " << t->name() << " is " << n->name();
                    m_p_consumers[ptr].push_back(last_node);
                }
            }
        }

        if (m_ws_map.count(n))
        {
            auto ws_set = m_ws_map[n];
            for (auto ws : ws_set)
            {
                update(ws, n, {n}, -1);
            }
        }

        for (auto t : n->outputs())
        {
            NumaTensorInfo info; // default dist_mode is UMA
            if (t->attrs().count(Constants::numa))
            {
                info = t->get_attr<NumaTensorInfo>(Constants::numa);
            }
            if (info.dist_mode == NumaDistMode::UMA)
            {
                update(t, n, t->consumers(), -1);
            }
            else if (info.dist_mode == NumaDistMode::LOCAL_SINGLE || info.dist_mode == NumaDistMode::REPLICATED)
            {
                ERROR() << "unsupoort";
            }
            else if (info.dist_mode == NumaDistMode::GRID_CENTRIC || info.dist_mode == NumaDistMode::BLOCK_CENTRIC)
            {
                for (size_t i = 0; i < info.slices.size(); ++i)
                {
                    auto s = info.slices[i];
                    update(s, n, t->consumers(), s->mem_id());
                }
            }
            else
            {
                ERROR() << "unsupport mode";
            }
        }
    }
    return last_node;
}

void SOMSScheduler::update_stream(GraphPtr g, const std::set<MemoryObjectPtr> &skip_mem, NodePtr stream)
{
    for (auto n : g->nodes())
    {
        m_streams[n] = stream;
        if (n->type() == OpType_Loop || n->type() == OpType_If)
        {
            auto sgn = SubGraphNode::from_node(n);
            update_stream(sgn->subgraph(), skip_mem, stream);
        }
    }
}

void SOMSScheduler::check_all_mems(GraphPtr g, const std::set<MemoryObjectPtr> &all_mems, const std::set<MemoryObjectPtr> &skip_mem)
{
    for (auto n : g->nodes())
    {
        if (n->type() == OpType_Loop || n->type() == OpType_If)
        {
            auto sgn = SubGraphNode::from_node(n);
            check_all_mems(sgn->subgraph(), all_mems, skip_mem);
        }

        for (auto t : n->outputs())
        {
            NumaTensorInfo info; // default dist_mode is UMA
            if (t->attrs().count(Constants::numa))
            {
                info = t->get_attr<NumaTensorInfo>(Constants::numa);
            }

            if (info.dist_mode == NumaDistMode::UMA)
            {
                auto ptr = t->memory_offset()->get_base_object();
                if (!contains(all_mems, ptr) && !contains(skip_mem, ptr))
                {
                    ERROR() << "tensor not allocate " << t->name();
                }
            }
            else if (info.dist_mode == NumaDistMode::LOCAL_SINGLE || info.dist_mode == NumaDistMode::REPLICATED)
            {
                ERROR() << "unsupoort";
            }
            else if (info.dist_mode == NumaDistMode::GRID_CENTRIC || info.dist_mode == NumaDistMode::BLOCK_CENTRIC)
            {
                for (size_t i = 0; i < info.slices.size(); ++i)
                {
                    auto s = info.slices[i];
                    auto ptr = s->memory_offset()->get_base_object();
                    if (!contains(all_mems, ptr) && !contains(skip_mem, ptr))
                    {
                        ERROR() << "tensor not allocate " << s->name();
                    }
                }
            }
            else
            {
                ERROR() << "unsupport mode";
            }
        }
    }
}

std::set<MemoryObjectPtr> SOMSScheduler::mem_live_init(const std::set<MemoryObjectPtr> &skip_mem)
{
    std::set<MemoryObjectPtr> all_mems;
    int node_stamp = 0;
    for (size_t i = 0; i < m_levels.size(); ++i)
    {
        auto level = m_levels[i];
        for (auto n : level)
        {
            n->set_attr(Constants::node_level, i);
            auto sgn = SubGraphNode::from_node(n);
            if (sgn)
            {
                update_gen_consume(sgn->subgraph(), node_stamp, all_mems, skip_mem);
            }
            else
            {
                INFO() << "not assign " << n->name();
            }
        }
    }

    for (size_t i = 0; i < m_levels.size(); ++i)
    {
        auto nodes = m_levels[i];
        for (auto n : nodes)
        {
            auto sgn = SubGraphNode::from_node(n);
            if (sgn)
            {
                update_stream(sgn->subgraph(), skip_mem, n);
            }
            else
            {
                INFO() << "not assign " << n->name();
            }
        }
    }

    // check if all mems
    for (size_t i = 0; i < m_levels.size(); ++i)
    {
        auto level = m_levels[i];
        for (auto n : level)
        {
            auto sgn = SubGraphNode::from_node(n);
            if (sgn)
            {
                check_all_mems(sgn->subgraph(), all_mems, skip_mem);
            }
        }
    }

    return all_mems;
}

std::map<MemoryObjectPtr, std::vector<MemoryObjectPtr>> SOMSScheduler::gen_mem_conflict(const std::set<MemoryObjectPtr> &all_mems)
{
    std::map<MemoryObjectPtr, std::vector<MemoryObjectPtr>> conflict_map;
    std::set<MemoryObjectPtr> unsafe_mems = all_mems;
    for (auto m1 : all_mems)
    {
        int reason = 0;
        int32_t l1_s = get_start_level(m1);
        int32_t l1_e = get_end_level(m1);

        unsafe_mems.erase(m1);
        for (auto m2 : unsafe_mems)
        {
            int32_t l2_s = get_start_level(m2);
            int32_t l2_e = get_end_level(m2);

            // INFO() << "check safe: " << m_tensor_map[m1]->name() << " and " << m_tensor_map[m2]->name();
            if (l1_s > l2_s)
            {
                if (l1_s > l2_e)
                {
                    continue; // No conflict
                }
                else if (l1_s == l2_e)
                {
                    if (anc_stream_reuse(m1, m2))
                    {
                        continue; // No conflict
                    }
                }
                reason = 1;
            }
            else if (l1_s < l2_s)
            {
                if (l1_e < l2_s)
                {
                    continue; // No conflict
                }
                else if (l1_e == l2_s)
                {
                    if (anc_stream_reuse(m2, m1))
                    {
                        continue; // No conflict
                    }
                }
                reason = 1;
            }
            else
            { // l1_s == l2_s
                if ((l1_s == l1_e) && (l2_s < l2_e))
                {
                    if (anc_stream_reuse(m2, m1))
                    {
                        continue;
                    }
                    reason = 1;
                }
                else if ((l1_s < l1_e) && (l2_e == l2_s))
                {
                    if (anc_stream_reuse(m1, m2))
                    {
                        continue;
                    }
                    reason = 1;
                }
                else if ((l1_s == l1_e) && (l2_s == l2_e))
                {
                    if (same_stream(m1, m2))
                    {
                        if (same_stream_reuse(m1, m2))
                        {
                            continue; // No conflict
                        }
                        reason = 3;
                    }
                    else
                    {
                        if (anc_stream_reuse(m1, m2))
                        {
                            continue; // No conflict
                        }
                        else if (anc_stream_reuse(m2, m1))
                        {
                            continue; // No conflict
                        }
                        reason = 2;
                    }
                }
                else
                {
                    reason = 1;
                }
            }

            // INFO() << "unsafe: " << m_tensor_map[m1]->name() << " and " << m_tensor_map[m2]->name();
            conflict_map[m1].push_back(m2);
            conflict_map[m2].push_back(m1);
            if (aipubt::BuilderConfig::get<bool>(Constants::show_soms_bottleneck))
            {
                m_conflict_reason[make_pair(m1, m2)] = reason;
                m_conflict_reason[make_pair(m2, m1)] = reason;
            }
        }
    }

    return conflict_map;
}

static void print_child(NodePtr n, const std::map<uint64_t, MemoryObjectPtr> &mem_map, const std::map<MemoryObjectPtr, uint64_t> &offset_map, const std::map<MemoryObjectPtr, TensorPtr> &m_tensor_map, const std::map<MemoryObjectPtr, size_t> &mem_size, const std::map<std::pair<MemoryObjectPtr, MemoryObjectPtr>, uint32_t> &m_conflict_reason)
{
    uint32_t id = n->attrs()["mem_id"];
    auto mem = mem_map.at(id);
    auto mem_offset = offset_map.at(mem);
    auto t = m_tensor_map.at(mem);
    for (auto c : n->children())
    {
        uint32_t c_id = c->attrs()["mem_id"];
        auto c_mem = mem_map.at(c_id);
        auto c_offset = offset_map.at(c_mem);
        auto c_t = m_tensor_map.at(c_mem);
        auto reason = 0;
        if (m_conflict_reason.count(make_pair(mem, c_mem)))
        {
            reason = m_conflict_reason.at(make_pair(mem, c_mem));
        }
        std::string reason_str = reason == 0 ? "no reason" : reason == 1 ? "data parallel"
                                                         : reason == 2   ? "multi core parallel"
                                                         : reason == 3   ? "same core overlap"
                                                                         : "";
        INFO() << t->name() << "(0x" << std::hex << mem_size.at(mem) << ") offset 0x" << mem_offset << " child: " << c_t->name() << "(0x" << mem_size.at(c_mem) << ") offset: 0x" << c_offset << " conflict reason: " << reason_str;
        print_child(c, mem_map, offset_map, m_tensor_map, mem_size, m_conflict_reason);
    }
}

uint64_t SOMSScheduler::offset_assignment(MemoryObjectPtr pool, const std::map<MemoryObjectPtr, size_t> &memory_size, const std::map<MemoryObjectPtr, std::vector<MemoryObjectPtr>> &conflict_map, AllocOrder s, bool intermediate_only, uint32_t opt_loop, AllocGoal opt_goal, const std::set<MemoryObjectPtr> &mem_ptr, std::set<MemoryObjectPtr> &assigned_ptr, std::set<MemoryObjectPtr> &unassigned_ptr, std::map<MemoryObjectPtr, uint64_t> &offset_map, std::set<MemoryObjectPtr> &total_assigned_ptr)
{
    std::set<MemoryObjectPtr> swapped_vec;

    auto avoid = [&memory_size](MemoryObjectPtr p, const std::set<MemoryObjectPtr> &assigned_mems, const std::unordered_set<MemoryObjectPtr> &conflict_m, const std::map<MemoryObjectPtr, uint64_t> &offset_map_pool) -> std::vector<std::pair<uint64_t, uint64_t>> {
        std::vector<std::pair<uint64_t, uint64_t>> ret;
        for (auto m : assigned_mems)
        {
            if (conflict_m.count(m))
            {
                // INFO() << "unsafe: " << a.t->node()->name();
                ret.push_back(std::make_pair(offset_map_pool.at(m), memory_size.at(m)));
            }
        }
        return ret;
    };

    auto merge = [](const std::vector<std::pair<uint64_t, uint64_t>> &intervals) -> std::vector<std::pair<uint64_t, uint64_t>> {
        std::vector<std::pair<uint64_t, uint64_t>> ret;
        for (auto interval : intervals)
        {
            bool overlap = false;
            for (auto &i : ret)
            {
                if (i.first == interval.first)
                {
                    i.second = std::max(i.second, interval.second);
                    overlap = true;
                }
                else if (i.first > interval.first)
                {
                    if (i.first <= interval.first + interval.second)
                    {
                        uint64_t size1 = i.first - interval.first + i.second;
                        uint64_t size2 = interval.second;
                        i.first = interval.first;
                        i.second = std::max(size1, size2);
                        overlap = true;
                    }
                }
                else
                {
                    if (i.first + i.second >= interval.first)
                    {
                        uint64_t size1 = interval.first - i.first + interval.second;
                        uint64_t size2 = i.second;
                        i.second = std::max(size1, size2);
                        overlap = true;
                    }
                }
            }

            if (!overlap)
            {
                ret.push_back(interval);
            }
        }
        return ret;
    };

    auto fit = [](uint64_t size, uint64_t align, const std::vector<std::pair<uint64_t, uint64_t>> &intervals, uint64_t available_offset) -> uint64_t {
        uint64_t ret = 0;
        uint64_t start = 0;
        for (size_t i = 0; i < intervals.size(); ++i)
        {
            auto interval = intervals[i];
            if (start + size <= interval.first && start >= available_offset)
            {
                ret = start;
                break;
            }
            else
            {
                start = interval.first + interval.second;
                start = aligned(start, align);
            }
        }
        if (start < available_offset)
        {
            start = aligned(available_offset, align);
        }
        ret = start;
        return ret;
    };

    auto switch_bottleneck = [&memory_size, &assigned_ptr, &unassigned_ptr, &swapped_vec](std::vector<MemoryObjectPtr> &sorted, bool flag) -> bool {
        bool found = false;
        MemoryObjectPtr swap_ptr;
        std::vector<MemoryObjectPtr> v;
        if (flag)
        {
            v.assign(assigned_ptr.begin(), assigned_ptr.end());
        }
        else
        {
            v.assign(unassigned_ptr.begin(), unassigned_ptr.end());
        }

        std::stable_sort(v.begin(), v.end(), [&memory_size](MemoryObjectPtr a, MemoryObjectPtr b) {
            uint64_t s1 = memory_size.at(a);
            uint64_t s2 = memory_size.at(b);
            return s1 > s2;
        });

        for (size_t j = 0; j < v.size(); ++j)
        {
            auto ptr = v[j];
            if (contains(swapped_vec, ptr))
            {
                continue;
            }
            found = true;
            swap_ptr = ptr;
        }

        if (!found)
        {
            // INFO() << "not found!";
            return false;
        }

        swapped_vec.insert(swap_ptr);
        auto it = std::find(sorted.begin(), sorted.end(), swap_ptr);
        if (it != sorted.end())
        {
            sorted.erase(it);
            if (flag)
            {
                sorted.push_back(swap_ptr);
            }
            else
            {
                sorted.insert(sorted.begin(), swap_ptr);
            }
        }
        else
        {
            ERROR();
        }

        return true;
    };

    auto gen_bottleneck = [&memory_size](MemoryObjectPtr ptr, uint32_t mem_id, std::map<uint64_t, MemoryObjectPtr> &mem_map, GraphPtr mem_graph, uint64_t offset, const std::vector<std::pair<uint64_t, uint64_t>> &avoid_internals, const std::unordered_set<MemoryObjectPtr> &conflict_list, const std::map<MemoryObjectPtr, uint64_t> &offset_map_pool) -> void {
        mem_id++;
        mem_map[mem_id] = ptr;
        NodePtr new_node = Node::create(std::to_string(mem_id), OpType_NoOp);
        new_node->set_attr("mem_id", mem_id);
        TensorPtr out = std::make_shared<Tensor>(std::to_string(mem_id) + "out", TensorShape(0), Tensor::Dtype::UINT8);
        new_node->add_output(out);
        mem_graph->add_node(new_node);
        // INFO() << "add node " << m_tensor_map[ptr] << " offset: 0x" std::hex << offset;
        for (auto interval : avoid_internals)
        {
            auto end = interval.first + interval.second;
            // INFO() << "end: 0x" << std::hex << end;
            auto end_aligned = aligned(end, ptr->align);
            if (end_aligned == offset)
            {
                for (auto n : mem_graph->nodes())
                {
                    uint32_t id = n->attrs()["mem_id"];
                    auto mem = mem_map[id];
                    auto mem_offset = offset_map_pool.at(mem);
                    auto mem_size = memory_size.at(mem);
                    // INFO() << "id: " << id << m_tensor_map[mem] << " mem_offset: 0x" << std::hex << mem_offset << " size: 0x" << mem_size;
                    if ((mem_offset + (uint64_t)mem_size == end) && conflict_list.count(mem))
                    {
                        n->add_input(out);
                        // INFO() <<  m_tensor_map[mem] << " add input " << m_tensor_map[ptr];
                    }
                    else
                    {
                        continue;
                    }
                }
            }
            else
            {
                continue;
            }
        }
    };

    auto show_bottleneck = [&memory_size](std::map<uint64_t, MemoryObjectPtr> &mem_map, GraphPtr mem_graph, const std::map<MemoryObjectPtr, TensorPtr> &tensor_map, const std::map<std::pair<MemoryObjectPtr, MemoryObjectPtr>, uint32_t> &conflict_reason, const std::map<MemoryObjectPtr, uint64_t> &offset_map_pool) {
        NodePtr longest_n = 0;
        uint64_t logest = 0;
        for (auto n : mem_graph->nodes())
        {
            if (n->parents().size() == 0)
            {
                uint64_t id = n->attrs()["mem_id"];
                auto mem = mem_map.at(id);
                auto mem_offset = offset_map_pool.at(mem);
                if (logest < mem_offset + memory_size.at(mem))
                {
                    logest = mem_offset + memory_size.at(mem);
                    longest_n = n;
                }
            }
        }

        uint64_t id = longest_n->attrs()["mem_id"];
        auto mem = mem_map.at(id);
        auto mem_offset = offset_map_pool.at(mem);
        auto t = tensor_map.at(mem);
        INFO() << "memory bottleneck " << t->name() << " offset 0x" << std::hex << mem_offset;
        print_child(longest_n, mem_map, offset_map_pool, tensor_map, memory_size, conflict_reason);
    };

    std::map<uint64_t, MemoryObjectPtr> mem_map;
    uint64_t used_size = 0;
    uint64_t available_size = pool->size;
    uint64_t available_offset = 0;
    std::vector<MemoryObjectPtr> sorted(mem_ptr.begin(), mem_ptr.end());
    if (s == orderbysize)
    {
        std::stable_sort(sorted.begin(), sorted.end(), [&memory_size, this](MemoryObjectPtr a, MemoryObjectPtr b) {
            uint64_t s1 = memory_size.at(a);
            uint64_t s2 = memory_size.at(b);
            return s1 == s2 ? m_tensor_map[a]->name() > m_tensor_map[b]->name() : s1 > s2;
        });
    }
    else if (s == orderbylife)
    {
        auto stream = m_levels[0][0];
        std::stable_sort(sorted.begin(), sorted.end(), [this, stream](MemoryObjectPtr a, MemoryObjectPtr b) {
            uint64_t t1, t2;
            t1 = end_time(a, stream) - start_time(a, stream);
            t2 = end_time(b, stream) - start_time(b, stream);
            return t1 == t2 ? m_tensor_map[a]->name() > m_tensor_map[b]->name() : t1 < t2;
        });
    }
    else if (s == orderbyfp)
    {
        auto stream = m_levels[0][0];
        std::stable_sort(sorted.begin(), sorted.end(), [this, stream](MemoryObjectPtr a, MemoryObjectPtr b) {
            uint64_t f1 = a->foot_print;
            uint64_t f2 = b->foot_print;
            return f1 == f2 ? m_tensor_map[a]->name() > m_tensor_map[b]->name() : f1 > f2;
        });
    }
    bool flag = false;
    std::map<MemoryObjectPtr, uint64_t> best_offset_map;
    uint64_t best_fp = 0;
    // uint32_t best_i = 0;
    for (size_t i = 0; i < opt_loop; ++i)
    {
        GraphPtr mem_graph = Graph::create();
        int mem_id = 0;
        std::map<uint64_t, MemoryObjectPtr> mem_map;
        std::set<MemoryObjectPtr> assigned_ptr_pool;
        std::set<MemoryObjectPtr> unassigned_ptr_pool;
        uint64_t fp_pool = 0;
        uint64_t used_size_pool = 0;
        std::map<MemoryObjectPtr, uint64_t> offset_map_pool;

        if (m_need_prior_gm)
        {
            std::stable_sort(sorted.begin(), sorted.end(), [](MemoryObjectPtr a, MemoryObjectPtr b) {
                if (a->get_mem_flag() == MemoryObject::flag::prior_in_gm)
                {
                    // INFO() << "prior in gm: " << m_tensor_map[a]->name();
                    return true;
                }
                return false;
            });
        }

        for (auto ptr : sorted)
        {
            if (contains(assigned_ptr_pool, ptr))
            {
                WARN() << "already assigned";
                continue;
            }

            if (intermediate_only && (contains(m_keep_mem, ptr) || contains(m_input_mem, ptr)))
            {
                unassigned_ptr_pool.insert(ptr);
                continue;
            }

            if (pool->get_mem_id() != ptr->get_mem_id())
            {
                unassigned_ptr_pool.insert(ptr);
                continue;
            }

            // INFO() << "assigning " << m_tensor_map[ptr]->name();

            available_size = pool->size;
            available_offset = 0;

            uint64_t mem_size = 0;
            if (memory_size.count(ptr))
            {
                mem_size = memory_size.at(ptr);
                if (mem_size > available_size)
                {
                    unassigned_ptr_pool.insert(ptr);
                    continue;
                }
            }
            else
            {
                ERROR() << "no mem size";
            }

            uint64_t offset;
            if (opt_goal == Linear)
            {
                offset = used_size_pool;
            }
            else
            {
                std::unordered_set<MemoryObjectPtr> conflict_list;
                if (conflict_map.count(ptr))
                {
                    for (const auto &item : conflict_map.at(ptr))
                    {
                        conflict_list.insert(item);
                    }
                }
                auto avoid_internals = avoid(ptr, assigned_ptr_pool, conflict_list, offset_map_pool);

                std::stable_sort(avoid_internals.begin(), avoid_internals.end(), [](std::pair<uint32_t, uint32_t> a, std::pair<uint32_t, uint32_t> b) {
                    return a.first < b.first;
                });

                // for (auto a : avoid_internals)
                // {
                //     INFO() << "avoid1: " << a.first << " " << a.second;
                // }

                avoid_internals = merge(avoid_internals);

                // for (auto a : avoid_internals)
                // {
                //     INFO() << "avoid2: " << a.first << " " << a.second;
                // }

                offset = fit(mem_size, ptr->align, avoid_internals, available_offset);

                if (aipubt::BuilderConfig::get<bool>(Constants::show_soms_bottleneck))
                {
                    gen_bottleneck(ptr, ++mem_id, mem_map, mem_graph, offset, avoid_internals, conflict_list, offset_map_pool);
                }
            }
            uint64_t total_size = offset + mem_size;
            if (total_size <= available_size)
            {
                if (used_size_pool < total_size)
                    used_size_pool = total_size;
                // INFO() << "total size for slot " << i << ": 0x" << std::hex << slots_used_size[i];
                fp_pool += ptr->foot_print;
                if (offset == 0)
                {
                    if (ptr->align > pool->align)
                    {
                        pool->align = ptr->align;
                    }
                }
                // INFO() << m_tensor_map[ptr]->name() << " assigned to offset 0x" << std::hex << offset << " size 0x" << mem_size;
                if (offset_map_pool.count(ptr))
                {
                    ERROR();
                }
                offset_map_pool[ptr] = offset;
                assigned_ptr_pool.insert(ptr);
                if (total_assigned_ptr.count(ptr))
                {
                    ERROR() << "duplicate assignment";
                }
            }
            else
            {
                unassigned_ptr_pool.insert(ptr);
            }
        }

        if (opt_goal == MaxFp)
        {
            if (best_fp == 0 || best_fp < fp_pool)
            {
                best_fp = fp_pool;
                best_offset_map = offset_map_pool; //TODO
                used_size = used_size_pool;
                assigned_ptr = assigned_ptr_pool;
                unassigned_ptr = unassigned_ptr_pool;
                // best_i = i;
            }
        }
        else //MinSpace
        {
            if (used_size == 0 || used_size > used_size_pool)
            {
                best_fp = fp_pool;
                best_offset_map = offset_map_pool; //TODO
                used_size = used_size_pool;
                assigned_ptr = assigned_ptr_pool;
                unassigned_ptr = unassigned_ptr_pool;
                // best_i = i;
            }
        }

        // INFO() << "loop " << i << " fp: " << utils::get_human_readable_size(fp_pool) << " used: 0x" << std::hex << used_size_pool;

        if (!switch_bottleneck(sorted, flag))
        {
            break;
        }

        if (aipubt::BuilderConfig::get<bool>(Constants::show_soms_bottleneck))
        {
            show_bottleneck(mem_map, mem_graph, m_tensor_map, m_conflict_reason, offset_map_pool);
        }
    }

    // DEBUG() << "get best opt " << best_i;

    for (auto pair : best_offset_map)
    {
        if (offset_map.count(pair.first))
        {
            ERROR();
        }
        offset_map[pair.first] = pair.second;
    }

    total_assigned_ptr.insert(assigned_ptr.begin(), assigned_ptr.end());

    return used_size;
}

std::vector<MemoryObjectPtr> SOMSScheduler::mem_assignment(const std::map<MemoryObjectPtr, std::vector<MemoryObjectPtr>> &conflict_map, const std::set<MemoryObjectPtr> &all_mems, const std::set<MemoryObjectPtr> &not_reuse_mem)
{
    std::map<MemoryObjectPtr, size_t> memory_size;
    graph_memory(memory_size, m_graph->nodes());
    for (auto ws_pair : m_ws_map)
    {
        auto ws_set = ws_pair.second;
        for (auto ws : ws_set)
        {
            auto ws_ptr = ws->memory_offset()->get_base_object();
            memory_size[ws_ptr] = ws->mem_size();
        }
    }

    for (auto gt : m_graph->global_tensors())
    {
        if (gt.second)
        {
            auto ptr = gt.second->memory_offset()->get_base_object();
            if (!memory_size.count(ptr))
            {
                memory_size[ptr] = gt.second->mem_size();
            }
            else
            {
                ERROR();
            }
        }
    }

    std::map<MemoryObjectPtr, uint64_t> offset_map;
    std::set<MemoryObjectPtr> mem_ptr = all_mems;
    std::set<MemoryObjectPtr> assigned_ptr;
    std::set<MemoryObjectPtr> unassigned_ptr;
    std::map<MemoryObjectPtr, std::set<MemoryObjectPtr>> slot_map;
    std::set<MemoryObjectPtr> total_assigned_ptr;
    //for gm
    if (m_gm_size.size() && m_gm_size[0] > 0)
    {
        // DEBUG() << "assign mem gm";
        MemoryObjectPtr gm_s = std::make_shared<MemoryObject>(0); //slot 0 gm
        gm_s->size = m_gm_size[0];
        gm_s->align = 0x1000;
        gm_s->set_mem_id(-1);

        uint32_t loop_cnt = BuilderConfig::get<uint32_t>(Constants::soms_gm_loop);

        AllocOrder order = m_graph->nodes().size() > 1 ? orderbysize : orderbylife;

        auto used_size = offset_assignment(gm_s, memory_size, conflict_map, order, true, loop_cnt, MaxFp, mem_ptr, assigned_ptr, unassigned_ptr, offset_map, total_assigned_ptr);

        if (m_need_prior_gm)
        {
            for (auto s : unassigned_ptr)
            {
                if (s->get_mem_flag() == MemoryObject::flag::prior_in_gm)
                {
                    WARN() << "prior tensor " << m_tensor_map[s]->name() << " is not assigned gm";
                }
            }
        }

        //shrink gm
        if (!BuilderConfig::get<bool>(Constants::enable_gm_enhance) && !BuilderConfig::get<bool>(Constants::enable_dynamic_gm))
        {
            gm_s->size = used_size;
        }

        if (gm_s->size)
        {
            m_gm_region_slots.push_back(gm_s);
            m_gm_size[0] = gm_s->size;
            slot_map[gm_s] = assigned_ptr;
        }

        mem_ptr = unassigned_ptr;
        assigned_ptr.clear();
        unassigned_ptr.clear();
    }

    if (m_numa_cfg.mem_size().size() > 0)
    {
        for (size_t i = 0; i < m_numa_cfg.mem_size().size(); ++i)
        {

            MemoryObjectPtr numa_s = std::make_shared<MemoryObject>(0);
            numa_s->size = m_numa_cfg.mem_size()[i];
            if (numa_s->size == 0)
            {
                continue;
            }
            numa_s->align = 0;
            numa_s->set_mem_id(i);
            auto used_size = offset_assignment(numa_s, memory_size, conflict_map, orderbysize, true, 1, MinSpace, mem_ptr, assigned_ptr, unassigned_ptr, offset_map, total_assigned_ptr);

            if (used_size)
            {
                numa_s->size = used_size;
                slot_map[numa_s] = assigned_ptr;
            }

            mem_ptr = unassigned_ptr;
            assigned_ptr.clear();
            unassigned_ptr.clear();
        }
    }

    if (BuilderConfig::get<uint64_t>(Constants::ocm_size) > 0)
    {
        // DEBUG() << "assign mem ocm";
        MemoryObjectPtr ocm_s = std::make_shared<MemoryObject>(0);
        uint64_t ocm_size = BuilderConfig::get<uint64_t>(Constants::ocm_size);
        ocm_s->size = ocm_size;
        ocm_s->align = 0x1000;
        ocm_s->set_mem_id(-1);
        offset_assignment(ocm_s, memory_size, conflict_map, orderbysize, true, 1, MinSpace, mem_ptr, assigned_ptr, unassigned_ptr, offset_map, total_assigned_ptr);
        slot_map[ocm_s] = assigned_ptr;
        mem_ptr = unassigned_ptr;
        assigned_ptr.clear();
        unassigned_ptr.clear();
    }

    while (!mem_ptr.empty())
    {
        // DEBUG() << "assign mem ddr";
        MemoryObjectPtr ddr_s = std::make_shared<MemoryObject>(0);
        ddr_s->size = 0x400000000000; //max size for ddr
        ddr_s->align = 0;
        ddr_s->set_mem_id(-1);
        auto used_size = offset_assignment(ddr_s, memory_size, conflict_map, orderbysize, false, 1, MinSpace, mem_ptr, assigned_ptr, unassigned_ptr, offset_map, total_assigned_ptr);
        ddr_s->size = used_size;
        slot_map[ddr_s] = assigned_ptr;
        mem_ptr = unassigned_ptr;
        assigned_ptr.clear();
        unassigned_ptr.clear();
        if (used_size == 0)
        {
            ERROR() << "some data can not be assigned";
            break;
        }
    }

    for (auto ptr : not_reuse_mem)
    {
        if (!memory_size.count(ptr) || memory_size[ptr] == 0)
        {
            continue;
        }
        // DEBUG() << "assign mem not reuse";
        MemoryObjectPtr ddr_s = std::make_shared<MemoryObject>(0);
        ddr_s->size = memory_size[ptr];
        ddr_s->align = 0;
        mem_ptr.insert(ptr);
        offset_assignment(ddr_s, memory_size, conflict_map, orderbysize, false, 1, MinSpace, mem_ptr, assigned_ptr, unassigned_ptr, offset_map, total_assigned_ptr);
        slot_map[ddr_s] = assigned_ptr;
        assigned_ptr.clear();
        mem_ptr.clear();
        if (unassigned_ptr.size())
        {
            ERROR();
        }
    }

    int cnt = 0;
    std::vector<MemoryObjectPtr> slots;
    for (auto s : slot_map)
    {
        // DEBUG() << "slot ";
        if (s.first->size == 0)
        {
            ERROR() << "slot size is empty";
        }
        slots.push_back(s.first);
        s.first->buffer_id = cnt++;
        s.first->foot_print = 0;
        for (auto m : s.second)
        {
            // DEBUG() << at.t->name() << " offset " << at.offset << " size " << at.size;
            auto ptr = m;
            ptr->set_base_offset(s.first);
            ptr->relative_offset = offset_map[m];
            s.first->foot_print += ptr->foot_print;
        }
    }

    return slots;
}

int32_t SOMSScheduler::get_start_level(MemoryObjectPtr m)
{
    if (m_start_level_cache.find(m) != m_start_level_cache.end())
    {
        return m_start_level_cache[m];
    }

    int level;
    if (contains(m_input_mem, m))
    {
        level = 0;
    }
    else
    {
        level = 65535;
        auto nodes = src_stream(m);
        if (nodes.size())
        {
            for (auto n : nodes)
            {
                level = std::min(level, (int32_t)n->attrs()[Constants::node_level]);
            }
        }
        else
        {
            level = 0;
        }
    }

    m_start_level_cache[m] = level;

    return level;
}

int32_t SOMSScheduler::get_end_level(MemoryObjectPtr m)
{
    if (m_end_level_cache.find(m) != m_end_level_cache.end())
    {
        return m_end_level_cache[m];
    }

    int level;
    if (contains(m_keep_mem, m))
    {
        level = 65535;
    }
    else
    {
        level = 0;
        auto nodes = dest_stream(m);
        if (nodes.size() == 0)
        {
            nodes = src_stream(m);
        }
        for (auto n : nodes)
        {
            level = std::max(level, (int32_t)n->attrs()[Constants::node_level]);
        }
    }

    m_end_level_cache[m] = level;

    return level;
}

void SOMSScheduler::find_anc_streams(NodePtr n, std::set<NodePtr> &ancestors, const uint32_t &level)
{
    if (contains(ancestors, n))
    {
        return;
    }
    ancestors.insert(n);
    for (auto p : n->parents())
    {
        auto p_level = (uint32_t)p->attrs()[Constants::node_level];
        if (p_level < level)
        {
            continue;
        }
        find_anc_streams(p, ancestors, level);
    }
    return;
}

std::set<NodePtr> SOMSScheduler::anc_streams(NodePtr n, uint32_t level)
{
    auto key = std::make_pair(n, level);
    if (m_anc_streams_cache.find(key) != m_anc_streams_cache.end())
    {
        return m_anc_streams_cache[key];
    }

    std::set<NodePtr> ancestors;
    find_anc_streams(n, ancestors, level);
    auto it = ancestors.find(n);
    if (it != ancestors.end())
    {
        ancestors.erase(it);
    }
    else
    {
        ERROR();
    }
    m_anc_streams_cache[key] = ancestors;
    return ancestors;
}

bool SOMSScheduler::is_output_tensor(MemoryObjectPtr m)
{
    auto in = src_stream(m);
    auto out = dest_stream(m);

    if (in.size() > 1 || out.size() > 1)
    {
        return true;
    }

    if (in.size() == 1 && out.size() == 1 && in[0] != out[0])
    {
        return true;
    }

    return false;
}

std::vector<NodePtr> SOMSScheduler::src_stream(MemoryObjectPtr m)
{
    if (m_src_streams_cache.find(m) != m_src_streams_cache.end())
    {
        return m_src_streams_cache[m];
    }

    std::vector<NodePtr> ret;
    auto ptr = m;
    auto generators = m_p_generators[ptr];
    for (auto n : generators)
    {
        auto s = m_streams[n];
        if (!contains(ret, s))
        {
            ret.push_back(s);
        }
    }

    m_src_streams_cache[m] = ret;
    return ret;
}
std::vector<NodePtr> SOMSScheduler::dest_stream(MemoryObjectPtr m)
{
    if (m_dest_streams_cache.find(m) != m_dest_streams_cache.end())
    {
        return m_dest_streams_cache[m];
    }

    std::vector<NodePtr> ret;
    auto ptr = m;
    auto consumers = m_p_consumers[ptr];
    for (auto c : consumers)
    {
        auto s = m_streams[c];
        if (!contains(ret, s))
        {
            ret.push_back(s);
        }
    }

    m_dest_streams_cache[m] = ret;
    return ret;
}

uint32_t SOMSScheduler::start_time(MemoryObjectPtr m, NodePtr s)
{

    auto key = std::make_pair(s, m);
    if (m_start_time_cache.find(key) != m_start_time_cache.end())
    {
        return m_start_time_cache[key];
    }

    uint32_t stamp;
    if (contains(m_input_mem, m))
    {
        stamp = 0;
    }
    else
    {
        auto ptr = m;
        auto nodes = m_p_generators[ptr];
        bool found = false;
        stamp = 65535;
        for (auto node : nodes)
        {
            auto s_n = m_streams[node];
            if (s_n == s)
            {
                found = true;
                stamp = std::min((uint32_t)node->attrs()[Constants::mem_clock], stamp);
            }
        }
        if (!found)
        {
            stamp = 0;
        }
    }

    m_start_time_cache[key] = stamp;

    return stamp;
}

uint32_t SOMSScheduler::end_time(MemoryObjectPtr m, NodePtr s)
{
    auto key = std::make_pair(s, m);
    if (m_end_time_cache.find(key) != m_end_time_cache.end())
    {
        return m_end_time_cache[key];
    }

    uint32_t stamp;
    if (contains(m_keep_mem, m))
    {
        stamp = 65535;
    }
    else
    {
        auto ptr = m;
        auto nodes = m_p_consumers[ptr];
        if (nodes.size() == 0)
        {
            nodes = m_p_generators[ptr];
        }
        stamp = 0;
        bool found = false;
        for (auto node : nodes)
        {
            auto s_n = m_streams[node];
            if (s_n == s)
            {
                found = true;
                stamp = std::max((uint32_t)node->attrs()[Constants::mem_clock], stamp);
            }
        }
        if (!found)
        {
            stamp = 65535;
        }
    }

    m_end_time_cache[key] = stamp;

    return stamp;
}

//m2 is ancestor stream of m1
bool SOMSScheduler::anc_stream_reuse(MemoryObjectPtr m1, MemoryObjectPtr m2)
{
    if (aipubt::BuilderConfig::get<bool>(Constants::disable_dependency))
    {
        return false;
    }
    // INFO() << "anc_stream_reuse!";

    auto s1_nodes = src_stream(m1);
    std::vector<NodePtr> visited_s1;

    for (auto s1 : s1_nodes)
    {
        if (contains(visited_s1, s1))
        {
            continue;
        }
        visited_s1.push_back(s1);
        uint32_t l1 = (uint32_t)s1->attrs()[Constants::node_level];
        auto a_nodes = anc_streams(s1, l1);

        if (is_output_tensor(m2))
        {
            auto o_streams = dest_stream(m2);
            if (o_streams.size() == 0)
            {
                return false;
            }
            for (auto o_s : o_streams)
            {
                uint32_t l2 = (uint32_t)o_s->attrs()[Constants::node_level];
                if (l1 == l2)
                {
                    if (o_s == s1)
                    {
                        if (end_time(m2, s1) >= start_time(m1, s1))
                        {
                            return false;
                        }
                    }
                    else if (!contains(a_nodes, o_s))
                    {
                        return false;
                    }
                }
                else if (l2 > l1)
                {
                    return false;
                }
            }
        }
        else
        {
            auto in = src_stream(m2);
            auto s2 = in[0];
            uint32_t l2 = (uint32_t)s2->attrs()[Constants::node_level];
            if (l1 == l2)
            {
                if (s2 == s1)
                {
                    if (end_time(m2, s1) >= start_time(m1, s1))
                    {
                        return false;
                    }
                }
                else if (!contains(a_nodes, s2))
                {
                    return false;
                }
            }
            else if (l2 > l1)
            {
                return false;
            }
        }
    }
    return true;
}

bool SOMSScheduler::same_stream_reuse(MemoryObjectPtr m1, MemoryObjectPtr m2)
{
    auto s1_src = src_stream(m1);
    auto s1_dest = dest_stream(m1);
    NodePtr stream;
    if (s1_src.size())
    {
        stream = s1_src[0];
    }
    else
    {
        stream = s1_dest[0];
    }

    if ((end_time(m1, stream) < start_time(m2, stream)) || (start_time(m1, stream) > end_time(m2, stream)))
    {
        return true;
    }

    return false;
}

bool SOMSScheduler::same_stream(MemoryObjectPtr m1, MemoryObjectPtr m2)
{
    auto s1_src = src_stream(m1);
    auto s1_dest = dest_stream(m1);
    auto s2_src = src_stream(m2);
    auto s2_dest = dest_stream(m2);
    NodePtr stream;

    if (s1_src.size())
    {
        stream = s1_src[0];
    }
    else
    {
        stream = s1_dest[0];
    }

    for (auto s : s1_src)
    {
        if (s != stream)
        {
            return false;
        }
    }

    for (auto s : s1_dest)
    {
        if (s != stream)
        {
            return false;
        }
    }

    for (auto s : s2_src)
    {
        if (s != stream)
        {
            return false;
        }
    }

    for (auto s : s2_dest)
    {
        if (s != stream)
        {
            return false;
        }
    }

    return true;
}

void SOMSScheduler::disable_sm()
{
    m_enable_sm = false;
}

void SOMSScheduler::set_ws(std::map<NodePtr, std::set<TensorPtr>> &ws_map)
{
    m_ws_map = ws_map;
}

void SOMSScheduler::set_numacfg(const NumaCfg &numa_cfg)
{
    m_numa_cfg = numa_cfg;
}

}; // namespace aipubt
