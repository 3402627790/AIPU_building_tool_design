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
 * @Date: 2020-01-29 14:27:33 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-04-12 17:00:35
 */
#include "builder/amat_allocator.h"
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/constants.h"
#include "core/ops/ops.h"
#include "mc_scheduler.h"
#include "mmu_allocator.h"
#include "sram_optimize.h"
#include "utils/utils.h"
#include <cmath>
#include <set>
#include <vector>
namespace aipubt
{
inline size_t aligned(size_t s, size_t align = 256)
{
    return (s + align - 1) & ~(align - 1);
}
static bool compare_tensor_name(const TensorPtr &t1, const TensorPtr &t2)
{
    return t1->name() < t2->name();
}

void MCScheduler::graph_memory_allocate_mc(MemoryAllocator *m_allocator,
                                           std::set<MemoryObjectPtr> &keep,
                                           std::set<TensorPtr> &final_tensors,
                                           std::map<MemoryObjectPtr, int> &consumer_cnt,
                                           std::map<MemoryObjectPtr, size_t> &memory_size,
                                           std::set<MemoryObjectPtr> &allocated)
{
    std::set<TensorPtr> keeping_tensors = final_tensors;
    std::set<MemoryObjectPtr> final_buffers;
    for (auto t : final_tensors)
    {
        final_buffers.insert(t->memory_offset()->get_base_object());
    }
    size_t gm_size = 0;
    for (auto gm : m_gm_size)
    {
        gm_size += gm;
    }
    for (auto level : m_levels)
    {
        // to get the best liveness data, allocate internal buffer first, then allocator the output
        std::vector<MemoryObjectPtr> internal_buffer;
        auto gm_per_core = gm_size;
        if (!BuilderConfig::get<bool>(Constants::disable_separate_gm))
        {
            gm_per_core = gm_size / level.size();
        }
        auto mem_clock = m_allocator->get_clock();
        for (auto n : level)
        {
            if (SubGraphNode::is_subgraph_node(n) && !n->attrs().count(Constants::plugin_attrs_name))
            {
                n->set_attr(Constants::mem_clock, mem_clock);
                std::set<std::string> out_tensors_name;
                // allocate output use main allocator
                for (auto t : n->outputs())
                {
                    out_tensors_name.insert(t->name());
                    auto ptr = t->memory_offset()->get_base_object();
                    if (final_tensors.count(t) || final_buffers.count(ptr))
                    {
                        m_allocator->enable_soc_sram(false); // the output tensor should located on ddr.
                        final_tensors.erase(t);
                    }
                    if (!allocated.count(ptr) && memory_size.count(ptr))
                    {
                        uint32_t real_size = aligned(memory_size[ptr]);
                        if (!BuilderConfig::get<bool>(Constants::disable_separate_gm) && real_size > gm_per_core)
                        {
                            m_allocator->enable_soc_sram(false);
                        }
                        if (t->attrs().count("prior_in_gm"))
                        {
                            ptr->set_mem_flag(MemoryObject::flag::prior_in_gm);
                            INFO() << "prior in gm " << t->name();
                        }
                        m_allocator->allocate(real_size, ptr);
                        allocated.insert(ptr->get_base_object());
                    }
                    t->memory_offset()->get_base_object()->align = t->memory_offset()->get_max_aligned();
                    m_allocator->enable_soc_sram(true);
                }
                auto sgn = std::static_pointer_cast<SubGraphNode>(n);
                std::map<MemoryObjectPtr, int> sg_consumer_cnt;
                MemoryAllocator *allocator = get_mem_allocator();
                //TODO sm allocation both support weight and featuremaps
                auto reserve_orders = sram_optimize(sgn->subgraph(), keeping_tensors);
                if (reserve_orders.size())
                {
                    sgn->subgraph()->topological_sort(reserve_orders);
                    sgn->subgraph()->reset_layer_id();
                }
                std::vector<NodePtr> sg_nodes = sgn->subgraph()->nodes();
                std::set<TensorPtr> skip_tensors;
                for (auto t : sgn->subgraph()->get_output_tensors())
                {
                    if (out_tensors_name.count(t->name()))
                    {
                        skip_tensors.insert(t);
                    }
                }
                MemoryObjectPtr gm = nullptr;
                if (!BuilderConfig::get<bool>(Constants::disable_separate_gm))
                {
                    gm = MemoryObjectPtr(new MemoryObject(0));
                    gm->size = gm_per_core;
                    allocator->add_soc_sram(gm);
                    allocator->enable_soc_sram(true);
                }
                MemoryObjectPtr ocm = nullptr;
                if (BuilderConfig::exist(Constants::ocm_size) && BuilderConfig::get<uint64_t>(Constants::ocm_size) > 0)
                {
                    ocm = MemoryObjectPtr(new MemoryObject(0));
                    ocm->size = BuilderConfig::get<uint64_t>(Constants::ocm_size);
                    allocator->add_soc_sram(ocm, MemoryAllocator::L4);
                    allocator->enable_soc_sram(true);
                }
                graph_consumer(final_tensors, keep, sg_consumer_cnt, sg_nodes);
                graph_memory_allocate(allocator, sg_nodes, keep, final_tensors, sg_consumer_cnt, memory_size, allocated, skip_tensors);
                auto liveness = allocator->get_liveness();
                // INFO() << n << " liveness";
                // for (auto l : liveness)
                // {
                //     INFO() << l.first << " start " << l.second.first << " end " << l.second.second;
                // }
                m_liveness_map.insert(std::make_pair(sgn->subgraph(), liveness));
                allocator->compress_clock();
                // INFO() << n << " compressed liveness";
                // liveness = allocator->get_liveness();
                // for (auto l : liveness)
                // {
                //     INFO() << l.first << " start " << l.second.first << " end " << l.second.second;
                // }
                auto slots = allocator->slots();
                if (gm)
                {
                    m_allocator->enable_soc_sram(true);
                    uint64_t shrink_size = 0;
                    size_t align = 0;
                    bool use_gm = false;
                    for (auto buff : allocator->get_allocated())
                    {
                        if (gm->is_shared_by(buff))
                        {
                            if (buff->align > align)
                            {
                                align = buff->align;
                            }
                            if (buff->offset() + buff->size - gm->offset() > shrink_size)
                            {
                                shrink_size = buff->offset() - gm->offset() + buff->size;
                            }
                            use_gm = true;
                        }
                    }
                    gm->size = shrink_size;
                    gm->align = align;
                    if (use_gm)
                    {
                        m_allocator->allocate(gm->size, gm);
                        m_allocator->enable_soc_sram(false);
                        internal_buffer.push_back(gm);
                    }
                }
                if (ocm)
                {
                    m_allocator->enable_soc_sram(true);
                    uint64_t shrink_size = 0;
                    size_t align = 0;
                    bool use_ocm = false;
                    for (auto buff : allocator->get_allocated())
                    {
                        if (ocm->is_shared_by(buff))
                        {
                            if (buff->align > align)
                            {
                                align = buff->align;
                            }
                            if (buff->offset() + buff->size - ocm->offset() > shrink_size)
                            {
                                shrink_size = buff->offset() - ocm->offset() + buff->size;
                            }
                            use_ocm = true;
                        }
                    }
                    ocm->size = shrink_size;
                    ocm->align = align;
                    if (use_ocm)
                    {
                        m_allocator->allocate(ocm->size, ocm);
                        m_allocator->enable_soc_sram(false);
                        internal_buffer.push_back(ocm);
                    }
                }
                m_allocator->enable_soc_sram(true);
                for (auto s : slots)
                {
                    m_allocator->allocate(s->size, s);
                    internal_buffer.push_back(s);
                }

                delete allocator;
                continue;
            }

            //never reach
            ERROR() << "MC Scheduler found a not Subgraph Node in a level, which should not happen:" << n->name();
        }
        // free internal buffer
        for (auto ptr : internal_buffer)
        {
            // INFO() << "free ptr " << ptr << " size:" << ptr->size;
            m_allocator->free(ptr);
        }
        // free should use level
        for (auto n : level)
        {
            for (auto inp : n->inputs())
            {
                // INFO() << "free " << inp->name();
                MemoryObjectPtr ptr = inp->memory_offset();

                while (!consumer_cnt.count(ptr) && ptr)
                {
                    ptr = ptr->base_offset;
                }
                if (keep.count(ptr))
                {
                    continue;
                }
                if (consumer_cnt.count(ptr))
                {
                    consumer_cnt[ptr]--;
                    if (consumer_cnt[ptr] <= 0)
                    {
                        auto mmu = inp->memory_offset()->get_mmu_object();

                        if (mmu && mmu->next_block)
                        {
                            m_allocator->free(mmu->next_block);
                        }
                        else
                        {
                            m_allocator->free(ptr);
                        }
                    }
                }
                else
                {
                    aipu_assert("never reach!!!!");
                }
            }
            // m_allocator->print();
        }
        m_allocator->clock(); //for top graph each level is considered a clock
    }
    // m_liveness_map.insert(std::make_pair(m_graph, m_allocator->get_liveness()));
    // auto liveness = m_allocator->get_liveness();
    // for (const auto &pair : liveness)
    // {
    //     m_liveness.insert(pair);
    // }
}

std::vector<MemoryObjectPtr> MCScheduler::mem_allocate(const std::set<TensorPtr> &not_reuse_tensor, const std::set<TensorPtr> &inputs, const std::set<TensorPtr> &keeps_tensors, const std::set<TensorPtr> &skip_tensors)
{
    if (!m_graph)
    {
        return std::vector<MemoryObjectPtr>();
    }
    if (m_allocator)
    {
        delete m_allocator;
    }
    m_allocator = get_mem_allocator();

    DEBUG() << "Using CUMP for global memory optimization.";
    // use cump for gm allocation
    for (auto gm_size : m_gm_size)
    {
        auto gm_sram = std::make_shared<MemoryObject>(0);
        gm_sram->size = gm_size;
        gm_sram->align = 0x4000; //smmu requires 16k align
        m_gm_region_slots.push_back(gm_sram);
        m_allocator->add_soc_sram(gm_sram);
    }
    MemoryObjectPtr ocm_buffer = nullptr;
    if (BuilderConfig::exist(Constants::ocm_size) && BuilderConfig::get<uint64_t>(Constants::ocm_size) > 0)
    {
        ocm_buffer = std::make_shared<MemoryObject>(0);
        ocm_buffer->size = BuilderConfig::get<uint64_t>(Constants::ocm_size);
        m_allocator->add_soc_sram(ocm_buffer, MemoryAllocator::L4);
    }

    auto final_tensors = keeps_tensors;
    std::set<TensorPtr> not_reuse(not_reuse_tensor.begin(), not_reuse_tensor.end());
    std::set<MemoryObjectPtr> allocated, keep;
    for (auto t : skip_tensors)
    {
        allocated.insert(t->memory_offset()->get_base_object());
    }
    std::map<MemoryObjectPtr, int> consumer_cnt;
    std::map<MemoryObjectPtr, size_t> memory_size;
    std::vector<NodePtr> end_nodes;
    graph_memory(memory_size, m_graph->nodes());
    graph_consumer(keeps_tensors, keep, consumer_cnt, m_graph->nodes(), false);

    //final_tensors should never free
    for (auto t : final_tensors)
    {
        auto ptr = t->memory_offset()->get_base_object();
        consumer_cnt[ptr] = INT32_MAX; // infinity consumer, never free
    }

    // sort the not_reuse set to make sure allocate is ordered
    std::vector<TensorPtr> not_reuse_sorted(not_reuse.begin(), not_reuse.end());
    std::sort(not_reuse_sorted.begin(), not_reuse_sorted.end(), compare_tensor_name);

    m_allocator->enable_soc_sram(false);

    m_allocator->clock();
    // allocate input tensors
    for (auto t : inputs)
    {
        auto ptr = t->memory_offset()->get_base_object();
        if (!not_reuse.count(t) && !allocated.count(ptr) && memory_size.count(ptr) && !aipubt::BuilderConfig::get<bool>(Constants::enable_custom_smmu))
        {
            uint32_t real_size = memory_size[ptr];
            m_allocator->allocate(aligned(real_size), ptr);
            allocated.insert(ptr->get_base_object());
        }
    }

    for (auto t : not_reuse)
    {
        auto ptr = t->memory_offset()->get_base_object();
        if (!allocated.count(ptr) && memory_size.count(ptr))
        {
            allocated.insert(ptr->get_base_object());
        }
    }

    m_allocator->enable_soc_sram(true);

    graph_memory_allocate_mc(m_allocator, keep, final_tensors, consumer_cnt, memory_size, allocated);
    auto liveness = m_allocator->get_liveness();
    // INFO() << "top liveness";
    // for (auto l : liveness)
    // {
    //     INFO() << l.first << " start " << l.second.first << " end " << l.second.second;
    // }
    m_liveness_map.insert(std::make_pair(m_graph, liveness));
    m_allocator->compress_clock();
    // liveness = m_allocator->get_liveness();
    // INFO() << "top compressed liveness";
    // for (auto l : liveness)
    // {
    //     INFO() << l.first << " start " << l.second.first << " end " << l.second.second;
    // }

    auto slots = m_allocator->slots();

    auto all_buffers = m_allocator->get_allocated();
    auto gm_slots = m_gm_region_slots;
    m_gm_region_slots.clear();

    size_t ft = 0;
    for (auto sram : gm_slots)
    {
        size_t ocm_size = 0;
        bool is_pushed = false;
        for (auto buff : all_buffers)
        {
            if (sram->is_shared_by(buff))
            {
                if (!is_pushed)
                {
                    m_gm_region_slots.push_back(sram);
                    slots.push_back(sram);
                    is_pushed = true;
                }
                if (buff->offset() + buff->size - sram->offset() > ocm_size)
                {
                    ocm_size = buff->offset() - sram->offset() + buff->size;
                }
                ft += buff->foot_print;
            }
        }
        if (ocm_size > 0 && sram->size > ocm_size && !BuilderConfig::get<bool>(Constants::enable_gm_enhance) && !BuilderConfig::get<bool>(Constants::enable_dynamic_gm))
        {
            sram->size = ocm_size;
        }
    }

    INFO() << "with GM optimization reduce footprint:" << utils::get_human_readable_size(ft);
    print_slots_size(slots);
    // workspace_allocate(m_graph);
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
                n->set_attr(Constants::private_buffers, pb);
            }
        }
    }
    // not reuse tensors will not be allocated by allocator, because they are not reuse, and some of them (workspachave a variable size.
    for (auto tensor : not_reuse_sorted)
    {
        if (tensor && tensor->mem_size() > 0)
        {
            auto ptr = tensor->memory_offset()->get_base_object();
            auto slot = std::make_shared<MemoryObject>(0);
            uint32_t real_size = tensor->mem_size();
            if (memory_size.count(ptr))
            {
                real_size = memory_size[ptr];
            }
            slot->size = aligned(real_size);
            slot->align = ptr->align;
            ptr->set_base_offset(slot);
            ptr->relative_offset = 0;
            slots.push_back(slot);
        }
    }
    // move the ocm to the first place of slots
    if (ocm_buffer)
    {
        size_t ocm_size = 0;
        for (auto ptr : all_buffers)
        {
            if (ocm_buffer->is_shared_by(ptr))
            {
                if (ptr->offset() + ptr->size - ocm_buffer->offset() > ocm_size)
                {
                    ocm_size = ptr->offset() - ocm_buffer->offset() + ptr->size;
                }
            }
        }
        for (size_t i = 0; i < slots.size(); ++i)
        {
            if (ocm_buffer->is_shared_by(slots[i]))
            {
                slots.erase(slots.begin() + i);
                break;
            }
        }
        if (ocm_size > 0 && ocm_size < ocm_buffer->size)
        {
            ocm_buffer->size = ocm_size;
            DEBUG() << "ocm size shrink:" << ocm_size << ", original size:" << ocm_buffer->size;
        }
        if (ocm_size > 0)
        {
            slots.insert(slots.begin(), ocm_buffer);
        }
    }

    return slots;
}

}; // namespace aipubt
