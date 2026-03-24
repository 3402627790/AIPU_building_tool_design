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
 * @Date: 2024-07-15 14:27:29
 * @Last Modified by: kyle.huang
 * @Last Modified time: 2024-08-15 10:57:30
 */
#include "soms_scheduler.h"
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

std::vector<MemoryObjectPtr> SOMSScheduler::mem_allocate(const std::set<TensorPtr> &not_reuse_tensor, const std::set<TensorPtr> &inputs, const std::set<TensorPtr> &keeps_tensors, const std::set<TensorPtr> &skip_tensors)
{
    if (!m_graph)
    {
        return std::vector<MemoryObjectPtr>();
    }

    auto start = std::chrono::high_resolution_clock::now();

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

    MemoryObjectPtr gm_s = std::make_shared<MemoryObject>(0); //slot 0 gm
    gm_s->size = m_gm_size[0];
    gm_s->align = 0x1000;

    m_slots = offset_assignment(gm_s, conflict_map, all_mems, not_reuse_mem);
    if (gm_s->size)
    {
        m_gm_region_slots.push_back(gm_s);
        m_gm_size[0] = gm_s->size;
    }

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

NodePtr SOMSScheduler::update_gen_consume(GraphPtr g, int &node_stamp, std::set<MemoryObjectPtr> &all_mems, const std::set<MemoryObjectPtr> &skip_mem, const int core_id)
{
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
            last_node = update_gen_consume(sgn->subgraph(), node_stamp, all_mems, skip_mem, core_id);

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

        for (auto t : n->outputs())
        {
            auto ptr = t->memory_offset()->get_base_object();
            // INFO() << "SG tensor " <<t->name() << " node " << t->node()->name() << " ptr " << ptr;
            if (n->type() != OpType_SubGraphInput)
            {
                if (n->type() != OpType_Loop && n->type() != OpType_If)
                {
                    m_p_generators[ptr].push_back(n);
                }
            }
            for (auto c : t->consumers())
            {
                if (c->type() != OpType_Loop && c->type() != OpType_If)
                {
                    m_p_consumers[ptr].push_back(c);
                }
            }
            if (t->attrs().count("prior_in_gm"))
            {
                ptr->set_mem_flag(MemoryObject::flag::prior_in_gm);
                INFO() << "prior in gm " << t->name();
            }
            if (!contains(skip_mem, ptr))
            {
                if (contains(g->get_output_tensors(), t))
                {
                    m_mem_core_map[ptr] = -1;
                }
                else
                {
                    if (!m_mem_core_map.count(ptr) || m_mem_core_map[ptr] != -1)
                    {
                        m_mem_core_map[ptr] = core_id;
                    }
                }
                all_mems.insert(ptr);
                m_tensor_map[ptr] = t;
            }
            // else
            // {
            //     INFO() << "skip " << t->node()->name() << " ptr " << ptr;
            // }
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
            auto ptr = t->memory_offset()->get_base_object();
            if (!contains(all_mems, ptr) && !contains(skip_mem, ptr))
            {
                ERROR() << "tensor not allocate " << t->name();
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
            int core_id = -1;
            if (n->attrs().count(Constants::virtual_core_id))
            {
                core_id = (int32_t)n->attrs()[Constants::virtual_core_id];
            }
            else
            {
                ERROR() << "empty mem " << n->name();
            }
            auto sgn = SubGraphNode::from_node(n);
            if (sgn)
            {
                update_gen_consume(sgn->subgraph(), node_stamp, all_mems, skip_mem, core_id);
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

    //check if all mems
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

static void print_child(NodePtr n, const std::map<uint32_t, MemoryObjectPtr> &mem_map, const std::map<MemoryObjectPtr, uint32_t> &offset_map, const std::map<MemoryObjectPtr, TensorPtr> &m_tensor_map, const std::map<MemoryObjectPtr, size_t> &mem_size, const std::map<std::pair<MemoryObjectPtr, MemoryObjectPtr>, uint32_t> &m_conflict_reason)
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

std::vector<MemoryObjectPtr> SOMSScheduler::offset_assignment(MemoryObjectPtr gm_s, const std::map<MemoryObjectPtr, std::vector<MemoryObjectPtr>> &conflict_map, const std::set<MemoryObjectPtr> &all_mems, const std::set<MemoryObjectPtr> &not_reuse_mem)
{
    std::map<MemoryObjectPtr, size_t> memory_size;
    std::map<MemoryObjectPtr, uint32_t> offset_map;

    auto avoid = [&memory_size, &offset_map](MemoryObjectPtr p, const std::vector<MemoryObjectPtr> &assigned_mems, const std::unordered_set<MemoryObjectPtr> &conflict_m) -> std::vector<std::pair<uint32_t, uint32_t>> {
        std::vector<std::pair<uint32_t, uint32_t>> ret;
        for (auto m : assigned_mems)
        {
            if (conflict_m.count(m))
            {
                // INFO() << "unsafe: " << a.t->node()->name();
                ret.push_back(std::make_pair(offset_map[m], memory_size[m]));
            }
        }
        return ret;
    };

    auto merge = [](const std::vector<std::pair<uint32_t, uint32_t>> &intervals) -> std::vector<std::pair<uint32_t, uint32_t>> {
        std::vector<std::pair<uint32_t, uint32_t>> ret;
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
                        uint32_t size1 = i.first - interval.first + i.second;
                        uint32_t size2 = interval.second;
                        i.first = interval.first;
                        i.second = std::max(size1, size2);
                        overlap = true;
                    }
                }
                else
                {
                    if (i.first + i.second >= interval.first)
                    {
                        uint32_t size1 = interval.first - i.first + interval.second;
                        uint32_t size2 = i.second;
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

    auto fit = [](uint32_t size, uint32_t align, const std::vector<std::pair<uint32_t, uint32_t>> &intervals, uint32_t available_offset) -> uint32_t {
        uint32_t ret = 0;
        uint32_t start = 0;
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

    graph_memory(memory_size, m_graph->nodes());
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

    std::vector<MemoryObjectPtr> slots;
    std::vector<uint32_t> slots_used_size;
    std::vector<MemoryObjectPtr> sorted(all_mems.begin(), all_mems.end());

    int sg_size = 0;
    for (auto level : m_levels)
    {
        for (auto n : level)
        {
            sg_size++;
        }
    }
    if (sg_size == 1)
    {
        auto stream = m_levels[0][0];
        std::stable_sort(sorted.begin(), sorted.end(), [this, stream](MemoryObjectPtr a, MemoryObjectPtr b) {
            uint32_t t1, t2;
            t1 = end_time(a, stream) - start_time(a, stream);
            t2 = end_time(b, stream) - start_time(b, stream);
            return t1 == t2 ? m_tensor_map[a]->name() > m_tensor_map[b]->name() : t1 < t2;
            // return a->foot_print > b->foot_print;
        });
    }
    else
    {
        std::stable_sort(sorted.begin(), sorted.end(), [&memory_size, this](MemoryObjectPtr a, MemoryObjectPtr b) {
            // return conflict_map[a].size() > conflict_map[b].size();
            uint32_t s1 = memory_size[a];
            uint32_t s2 = memory_size[b];
            return s1 == s2 ? m_tensor_map[a]->name() > m_tensor_map[b]->name() : s1 > s2;
            // return a->foot_print > b->foot_print;
        });
    }

    std::map<MemoryObjectPtr, std::vector<MemoryObjectPtr>> slot_map;

    std::vector<MemoryObjectPtr> assigned_ptr, max_assigned_ptr, unassigned_ptr;
    std::vector<MemoryObjectPtr> gm_mems, max_gm_mems;
    uint32_t gm_used_size = 0, max_gm_used_size = 0;
    uint32_t gm_fp = 0, max_gm_fp = 0;
    std::vector<MemoryObjectPtr> swapped_vec;

    //for gm
    std::map<MemoryObjectPtr, uint32_t> max_offset_map;
    // uint32_t swap_idx = 0;
    // uint32_t swapped_idx = 0;
    // bool restore = false;
    bool flag = true;
    bool found = false;
    uint32_t max_loop = 0;
    uint32_t gm_available_size = gm_s->size;
    uint32_t gm_available_offset = 0;
    uint32_t loop_cnt = BuilderConfig::get<uint32_t>(Constants::soms_gm_loop);
    if (gm_s->size == 0)
    {
        loop_cnt = 1;
    }
    for (size_t i = 0; i < loop_cnt; ++i)
    {
        if (BuilderConfig::get<std::string>(Constants::load_prior_in_gm) != "" || BuilderConfig::get<bool>(Constants::softmax_prior_in_gm))
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

        // INFO() << "gm try: " << i;
        gm_mems.clear();
        gm_fp = 0;
        gm_used_size = 0;
        assigned_ptr.clear();
        unassigned_ptr.clear();
        offset_map.clear();
        for (auto m : sorted)
        {
            auto ptr = m;
            if (contains(assigned_ptr, ptr))
            {
                continue;
            }

            uint32_t mem_size = 0;
            if (memory_size.count(ptr))
            {
                mem_size = memory_size[ptr];
            }
            else
            {
                ERROR() << "no mem size";
            }

            int core_id = m_mem_core_map[ptr];
            if (core_id == -1)
            {
                gm_available_size = gm_s->size;
                gm_available_offset = 0;
            }
            else
            {
                gm_available_size = gm_s->size / Target::get_the_target()->core_num();
                gm_available_offset = core_id * gm_available_size;
            }

            if (mem_size > gm_available_size)
            {
                continue;
            }

            // DEBUG() << "allocating gm " << m_tensor_map[m]->node()->name() << " " << mem_size;
            // std::vector<MemoryObjectPtr> conflict_list;
            std::unordered_set<MemoryObjectPtr> conflict_list;
            if (conflict_map.count(ptr))
            {
                // conflict_list = conflict_map.at(ptr);
                for (const auto &item : conflict_map.at(ptr))
                {
                    conflict_list.insert(item);
                }
            }
            auto avoid_internals = avoid(ptr, gm_mems, conflict_list);

            std::stable_sort(avoid_internals.begin(), avoid_internals.end(), [](std::pair<uint32_t, uint32_t> a, std::pair<uint32_t, uint32_t> b) {
                return a.first < b.first;
            });

            // for(auto a : avoid_internals)
            // {
            //     INFO() << "avoid1: " << a.first << " " << a.second;
            // }

            avoid_internals = merge(avoid_internals);
            // for(auto a : avoid_internals)
            // {
            //     INFO() << "avoid2: " << a.first << " " << a.second;
            // }

            auto offset = fit(mem_size, ptr->align, avoid_internals, gm_available_offset);

            if (offset + mem_size <= gm_available_offset + gm_available_size) // first fit
            {
                // DEBUG() << at.t->node()->name() << " can allocate slot " << i << " offset: " << offset << " size: " << mem_size;

                if (contains(m_keep_mem, ptr) || contains(m_input_mem, ptr)) //input and output will not in gm because bad sync performance
                {
                    // INFO() << "cannot assign to gm " << m_tensor_map[m]->node()->name() << " " << mem_size << " due to input or output";
                    continue;
                }
                // DEBUG() << m_tensor_map[m]->node()->name() << " can allocate gm offset: " << offset << " size: " << mem_size;
                gm_fp += ptr->foot_print;
                if (offset + mem_size > gm_used_size)
                {
                    gm_used_size = offset + mem_size;
                }

                offset_map[ptr] = offset;
                gm_mems.push_back(ptr);
                assigned_ptr.push_back(ptr);
            }
            else
            {
                unassigned_ptr.push_back(ptr);
            }
        }
        if (gm_fp == 0)
        {
            max_gm_fp = gm_fp;
            max_gm_used_size = gm_used_size;
            max_gm_mems = gm_mems;
            max_assigned_ptr = assigned_ptr;
            max_offset_map = offset_map;
            break;
        }

        // INFO() << "loop " << i << " gm_fp: " << utils::get_human_readable_size(gm_fp);
        if (gm_fp > max_gm_fp)
        {
            max_gm_fp = gm_fp;
            max_gm_used_size = gm_used_size;
            max_gm_mems = gm_mems;
            max_assigned_ptr = assigned_ptr;
            max_offset_map = offset_map;
            max_loop = i;
            // swapped_vec.clear();
        }
        // else if (!restore && found)
        // {
        //     // std::swap(sorted[swap_idx], sorted[swapped_idx]);
        //     if (flag)
        //     {
        //         auto element = sorted.back();
        //         sorted.pop_back();
        //         sorted.insert(sorted.begin() + swap_idx, element);
        //         INFO() << "restore back to index " << swap_idx << " due to fp " << utils::get_human_readable_size(gm_fp);
        //     }
        //     else
        //     {
        //         auto element = sorted.front();
        //         sorted.erase(sorted.begin());
        //         sorted.insert(sorted.begin() + swap_idx, element);
        //         INFO() << "restore front to index " << swap_idx << " due to fp " << utils::get_human_readable_size(gm_fp);
        //     }
        //     restore = true;
        //     continue;
        // }

        // restore = false;

        // INFO() << "fp in loop " << i << " " << utils::get_human_readable_size(gm_fp) << " " << assigned_ptr.size();

        flag = !flag;
        MemoryObjectPtr element;
        MemoryObjectPtr swapped;
        uint32_t idx1 = 0;
        // uint32_t idx2 = 0;
        found = false;
        if (flag)
        {
            std::stable_sort(assigned_ptr.begin(), assigned_ptr.end(), [&memory_size, this](MemoryObjectPtr a, MemoryObjectPtr b) {
                // return conflict_map.at(a).size() > conflict_map.at(b).size();
                uint32_t s1 = memory_size[a];
                uint32_t s2 = memory_size[b];
                // return a->foot_print > b->foot_print;
                return s1 == s2 ? m_tensor_map[a]->name() > m_tensor_map[b]->name() : s1 > s2;
            });
            for (size_t j = 0; j < assigned_ptr.size(); ++j)
            {
                auto ptr = assigned_ptr[j];
                if (contains(swapped_vec, ptr))
                {
                    continue;
                }
                found = true;
                idx1 = j;
            }
            if (!found)
            {
                INFO() << "not found!";
                break;
            }
            element = assigned_ptr[idx1];

            swapped_vec.push_back(element);
            auto it = std::find(sorted.begin(), sorted.end(), element);
            if (it != sorted.end())
            {
                // swap_idx = std::distance(sorted.begin(), it);
                sorted.erase(it);
                sorted.push_back(element);
                // INFO() << "swap index " << swap_idx;
            }
            else
            {
                ERROR();
            }
            // INFO() << "put index " << swap_idx << " " << m_tensor_map[element]->node()->name() << " " << memory_size[element] << " to the end " << sorted.size();
        }
        else
        {
            // uint32_t conflict_size = 0;
            std::stable_sort(unassigned_ptr.begin(), unassigned_ptr.end(), [&memory_size, this](MemoryObjectPtr a, MemoryObjectPtr b) {
                // return conflict_map.at(a).size() < conflict_map.at(b).size();
                uint32_t s1 = memory_size[a];
                uint32_t s2 = memory_size[b];
                // return a->foot_print > b->foot_print;
                return s1 == s2 ? m_tensor_map[a]->name() > m_tensor_map[b]->name() : s1 > s2;
            });
            for (size_t j = 0; j < unassigned_ptr.size(); ++j)
            {
                auto ptr = unassigned_ptr[j];
                int core_id = m_mem_core_map[ptr];

                if (core_id == -1)
                {
                    gm_available_size = gm_s->size;
                }
                else
                {
                    gm_available_size = gm_s->size / Target::get_the_target()->core_num();
                }

                if (contains(swapped_vec, ptr))
                {
                    continue;
                }
                if (memory_size[ptr] <= gm_available_size)
                {
                    found = true;
                    idx1 = j;
                }
            }
            if (!found)
            {
                INFO() << "not found!";
                break;
            }
            element = unassigned_ptr[idx1];
            swapped_vec.push_back(element);
            auto it = std::find(sorted.begin(), sorted.end(), element);
            if (it != sorted.end())
            {
                // swap_idx = std::distance(sorted.begin(), it);
                sorted.erase(it);
                sorted.insert(sorted.begin(), element);
            }
            else
            {
                ERROR();
            }
            // INFO() << "put index " << swap_idx << " " << m_tensor_map[element]->node()->name() << " " << memory_size[element] << " to the fronts " << sorted.size();
        }
    }

    gm_fp = max_gm_fp;
    gm_used_size = max_gm_used_size;
    gm_mems = max_gm_mems;
    assigned_ptr = max_assigned_ptr;
    offset_map = max_offset_map;
    DEBUG() << "get max in loop " << max_loop;

    if (gm_used_size != 0)
    {
        slot_map[gm_s] = gm_mems;
    }

    //for ddr

    GraphPtr mem_graph = Graph::create();
    int mem_id = 0;
    std::map<uint32_t, MemoryObjectPtr> mem_map;
    if (sorted.size())
    {
        std::stable_sort(sorted.begin(), sorted.end(), [&memory_size](MemoryObjectPtr a, MemoryObjectPtr b) {
            return memory_size[a] > memory_size[b];
        });

        //for ocm
        MemoryObjectPtr ocm_s = std::make_shared<MemoryObject>(0);
        if (BuilderConfig::get<uint64_t>(Constants::ocm_size))
        {
            uint32_t ocm_size = BuilderConfig::get<uint64_t>(Constants::ocm_size);
            ocm_s->size = ocm_size;
            ocm_s->align = 0x1000;
            slots.push_back(ocm_s);
            slots_used_size.push_back(0);
        }

        MemoryObjectPtr ddr_s = std::make_shared<MemoryObject>(0);
        ddr_s->size = 0xe0000000; //max size for ddr
        ddr_s->align = 0;
        slots.push_back(ddr_s);
        slots_used_size.push_back(0);

        for (auto m : sorted)
        {
            auto ptr = m;
            if (contains(assigned_ptr, ptr))
            {
                continue;
            }

            uint64_t mem_size = 0;
            if (memory_size.count(ptr))
            {
                mem_size = memory_size[ptr];
                if (mem_size >= 0xe0000000)
                {
                    ERROR() << "single tensor larger than 3.5G";
                    return slots;
                }
            }
            else
            {
                ERROR() << "no mem size";
            }
            // DEBUG() << "allocating " << m_tensor_map[m]->node()->name() << " 0x" << std::hex << mem_size;

            // for (auto s : slots)
            for (size_t i = 0; i < slots.size(); ++i)
            {
                auto s = slots[i];
                if (s == ocm_s && (contains(m_keep_mem, ptr) || contains(m_input_mem, ptr)))
                {
                    continue;
                }

                auto assigneed_tensors = slot_map[s];
                std::unordered_set<MemoryObjectPtr> conflict_list;
                if (conflict_map.count(m))
                {
                    // conflict_list = conflict_map.at(m);
                    for (const auto &item : conflict_map.at(m))
                    {
                        conflict_list.insert(item);
                    }
                }
                auto avoid_internals = avoid(m, assigneed_tensors, conflict_list);

                std::stable_sort(avoid_internals.begin(), avoid_internals.end(), [](std::pair<uint32_t, uint32_t> a, std::pair<uint32_t, uint32_t> b) {
                    return a.first < b.first;
                });

                // for(auto a : avoid_internals)
                // {
                //     INFO() << "avoid1: " << a.first << " " << a.second;
                // }

                avoid_internals = merge(avoid_internals);

                // for(auto a : avoid_internals)
                // {
                //     INFO() << "avoid2: " << a.first << " " << a.second;
                // }

                auto offset = fit(mem_size, ptr->align, avoid_internals, 0);

                if (aipubt::BuilderConfig::get<bool>(Constants::show_soms_bottleneck))
                {
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
                                auto mem_offset = offset_map[mem];
                                auto mem_size = memory_size[mem];
                                // INFO() << "id: " << id << m_tensor_map[mem] << " mem_offset: 0x" << std::hex << mem_offset << " size: 0x" << mem_size;
                                if ((mem_offset + (uint32_t)mem_size == end) && conflict_list.count(mem))
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
                }

                uint64_t total_size = offset + mem_size;
                if (total_size < s->size)
                {
                    if (slots_used_size[i] < total_size)
                        slots_used_size[i] = total_size;
                    // INFO() << "total size for slot " << i << ": 0x" << std::hex << slots_used_size[i];
                    if (offset == 0)
                    {
                        if (ptr->align > s->align)
                        {
                            s->align = ptr->align;
                        }
                    }
                    // INFO() << m_tensor_map[m]->name() << " assigned to slot " << i << " offset 0x" << std::hex << offset;
                    // if(offset_map.count(ptr))
                    // {
                    //     ERROR();
                    // }
                    offset_map[ptr] = offset;
                    // if(slot_map.count(ptr))
                    // {
                    //     ERROR();
                    // }
                    slot_map[s].push_back(ptr);
                    assigned_ptr.push_back(ptr);
                    break;
                }

                if (i == slots.size() - 1) //if this is the last slot then will create a new ddr slot)
                {
                    MemoryObjectPtr extra_ddr_s = std::make_shared<MemoryObject>(0);
                    extra_ddr_s->size = 0xe0000000; //max size for ddr
                    extra_ddr_s->align = 0;
                    slots.push_back(extra_ddr_s);
                    slots_used_size.push_back(0);
                }
            }
        }

        for (size_t i = 0; i < slots.size(); ++i)
        {

            if (slots_used_size[i] == 0)
            {
                slots.erase(slots.begin() + i);
                slots_used_size.erase(slots_used_size.begin() + i);
                i--;
            }
            else
            {
                slots[i]->size = slots_used_size[i];
            }
        }
    }

    if (aipubt::BuilderConfig::get<bool>(Constants::show_soms_bottleneck))
    {
        NodePtr longest_n = 0;
        uint32_t logest = 0;
        for (auto n : mem_graph->nodes())
        {
            if (n->parents().size() == 0)
            {
                uint32_t id = n->attrs()["mem_id"];
                auto mem = mem_map.at(id);
                auto mem_offset = offset_map.at(mem);
                if (logest < mem_offset + memory_size[mem])
                {
                    logest = mem_offset + memory_size[mem];
                    longest_n = n;
                }
            }
        }

        uint32_t id = longest_n->attrs()["mem_id"];
        auto mem = mem_map.at(id);
        auto mem_offset = offset_map.at(mem);
        auto t = m_tensor_map.at(mem);
        INFO() << "memory bottleneck " << t->name() << " offset 0x" << std::hex << mem_offset;
        print_child(longest_n, mem_map, offset_map, m_tensor_map, memory_size, m_conflict_reason);
    }

    assigned_ptr.clear();

    for (auto ptr : not_reuse_mem)
    {
        if (contains(assigned_ptr, ptr))
        {
            continue;
        }
        uint32_t mem_size = 0;
        if (memory_size.count(ptr))
        {
            mem_size = memory_size[ptr];
        }
        else
        {
            ERROR();
        }

        if (mem_size == 0)
        {
            WARN() << m_tensor_map[ptr] << " memory size is " << 0;
            continue;
        }

        // DEBUG() << t->name() << " can allocate new slot " << slots.size();
        MemoryObjectPtr s = std::make_shared<MemoryObject>(0);
        s->size = mem_size;
        s->align = ptr->align;
        // if(offset_map.count(ptr))
        // {
        //     ERROR() << m_tensor_map[ptr]->name() << " already assigned";
        // }
        offset_map[ptr] = 0;
        // if(slot_map.count(ptr))
        // {
        //     ERROR();
        // }
        slot_map[s].push_back(ptr);
        slots.push_back(s);

        assigned_ptr.push_back(ptr);
    }

    // for (auto pair : slot_map)
    // {
    //     INFO() << "new slot";
    //     auto vec = pair.second;
    //     for (auto v : vec)
    //     {
    //         if (m_tensor_map[v])
    //         {
    //             DEBUG() << "allocating " << m_tensor_map[v]->name() << " offset 0x" << std::hex << offset_map[v] << " size: 0x" << memory_size[v];
    //         }
    //     }
    // }

    for (auto s : slot_map)
    {
        // DEBUG() << "slot ";
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

    //shrink gm
    if (!BuilderConfig::get<bool>(Constants::enable_gm_enhance) && !BuilderConfig::get<bool>(Constants::enable_dynamic_gm))
    {
        gm_s->size = gm_used_size;
    }

    if (gm_s->size)
    {
        slots.insert(slots.begin(), gm_s);
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

}; // namespace aipubt
