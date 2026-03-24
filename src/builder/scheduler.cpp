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
 * @Last Modified time: 2023-02-21 11:16:13
 */
#include "scheduler.h"
#include "amat_allocator.h"
#include "builder/lowering/lowering.h"
#include "builder/profiler.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/constants.h"
#include "mmu_allocator.h"
#include "sram_optimize.h"
#include "statistic/statistic.h"
#include "utils/utils.h"
#include <set>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

namespace aipubt
{
inline size_t aligned(size_t s, size_t align = 256)
{
    return (s + align - 1) & ~(align - 1);
}
static std::set<OpType> ZERO_MEMORY_USE_OP = {
    OpType_Reshape,
    OpType_InternalConcatenated,
    OpType_InternalSpliter,
};

/**
 * @brief Get the external sram object
 *
 * @return std::vector<std::pair<uint32_t, uint32_t>>  list of sram in <address,size>;
 */
std::vector<std::pair<uint32_t, uint32_t>> get_external_sram()
{
    std::vector<std::pair<uint32_t, uint32_t>> list = {};
    size_t alignment;
    auto target = get_target();
    if (target->isainfo().type == ISA_ZY)
    {
        alignment = 1024 * 1024;
    }
    else
    {
        alignment = 16 * 1024;
    }

    static const char *external_sram = getenv(Constants::AIPU_EXTERNAL_SRAM);
    // static const char *external_sram = "0x00100000@0x80000000";
    if (external_sram)
    {
        std::string exsram(external_sram);
        static bool _info_once = false;
        for (auto s : utils::split_string(exsram, ','))
        {
            auto addr_size = utils::split_string(s, '@');
            if (addr_size.size() == 2)
            {
                size_t addr = std::stoul(addr_size[1], nullptr, 0);
                size_t size = std::stoul(addr_size[0], nullptr, 0);
                // if (!_info_once)
                // {
                //     INFO("using external sram with size: %#x Bytes, address: %#x.", size, addr);
                // }
                BuilderConfig::set(Constants::preallocated_soc_sram, true);
                //align the address34
                auto align_addr = (addr + alignment - 1) & (~(alignment - 1));
                int32_t new_size = addr + size - align_addr;
                if (new_size < 0)
                {
                    new_size = 0;
                }
                if (!_info_once && align_addr != addr)
                {
                    INFO("due to the alignment, usable extern sram from the address:%#x with size:%#x for align in %#x bytes.", align_addr, new_size, alignment);
                }
                list.push_back(std::make_pair((uint32_t)align_addr, (uint32_t)new_size));
            }
            else
            {
                WARN() << "WRONG format of external SRAM: format size@address, size and address must be int number, but we received: " << s;
            }
        }
        _info_once = true;
    }
    return list;
}

void Scheduler::print_slots_size(const std::vector<MemoryObjectPtr> &slots)
{
    std::ostringstream oss;
    std::ostringstream oss_fp;
    size_t total_size = 0;
    oss << std::hex << "[";
    oss_fp << std::hex << "[";
    for (unsigned int i = 0; i < slots.size(); ++i)
    {
        if (i != 0)
        {
            oss << ", ";
            oss_fp << ", ";
        }
        // oss << "0x" << slots[i]->size;
        oss << std::dec << (slots[i]->size >> 10);
        oss_fp << utils::get_human_readable_size(slots[i]->foot_print);
        total_size += slots[i]->size;
    }
    oss << "]";
    oss_fp << "]";
    // DEBUG() << "Total memory for feature map: 0x" << std::hex << total_size
    //         << "Bytes, slots: " << oss.str()
    //         << ", slots foot_print: " << oss_fp.str();
    DEBUG() << "Total memory for feature map: " << std::dec << (total_size >> 10)
            << " KB, slots: " << oss.str() << " KB";
    DEBUG() << "Slots footprints:" << oss_fp.str();
}

void Scheduler::graph_memory(std::map<MemoryObjectPtr, size_t> &memory_size,
                             const std::vector<NodePtr> &nodes)
{
    for (auto n : nodes)
    {
        if (SubGraphNode::is_subgraph_node(n) && !n->attrs().count(Constants::plugin_attrs_name))
        {
            auto sgn = std::static_pointer_cast<SubGraphNode>(n);
            graph_memory(memory_size, sgn->subgraph()->nodes());
        }
        else
        {
            for (auto &out : n->outputs())
            {
                auto ptr = out->memory_offset()->get_base_object();
                if (!memory_size.count(ptr))
                {
                    memory_size[ptr] = 0;
                }
                size_t out_size = std::max(memory_size[ptr], out->mem_size());
                auto aligned_size = Target::get_the_target()->isainfo().type == ISA_ZYv6 ? aligned(out_size, 64) : out_size;
                memory_size[ptr] = aligned_size;
                // INFO() << "out: " << out->name();
                // INFO() << "max mem: 0x" << std::hex << memory_size[ptr] << " out mem: 0x" << out->mem_size()<<" ptr=0x"<<ptr;
            }
        }
    }
}

void Scheduler::graph_consumer(const std::set<TensorPtr> &keeps_tensors,
                               std::set<MemoryObjectPtr> &keep,
                               std::map<MemoryObjectPtr, int> &consumer_cnt,
                               const std::vector<NodePtr> &nodes,
                               bool recurrently)
{
    for (auto n : nodes)
    {
        if (recurrently && SubGraphNode::is_subgraph_node(n) && !n->attrs().count(Constants::plugin_attrs_name))
        {
            auto sgn = std::static_pointer_cast<SubGraphNode>(n);
            graph_consumer(keeps_tensors, keep, consumer_cnt, sgn->subgraph()->nodes());
        }
        else
        {
            for (auto &inp : n->inputs())
            {
                auto ptr = inp->memory_offset()->get_base_object();
                if (!consumer_cnt.count(ptr))
                {
                    consumer_cnt[ptr] = 0;
                }
                consumer_cnt[ptr]++;
                if (keeps_tensors.count(inp))
                {
                    keep.insert(ptr);
                }
            }
            for (auto &out : n->outputs())
            {
                auto ptr = out->memory_offset()->get_base_object();
                if (keeps_tensors.count(out))
                {
                    keep.insert(ptr);
                    // end_nodes.push_back(n);
                }
            }
        }
    }
}

void Scheduler::graph_memory_allocate(MemoryAllocator *m_allocator,
                                      const std::vector<NodePtr> &nodes,
                                      std::set<MemoryObjectPtr> &keep,
                                      std::set<TensorPtr> &final_tensors,
                                      std::map<MemoryObjectPtr, int> &consumer_cnt,
                                      std::map<MemoryObjectPtr, size_t> &memory_size,
                                      std::set<MemoryObjectPtr> &allocated,
                                      const std::set<TensorPtr> &skip_tensors)
{
    std::set<MemoryObjectPtr> final_buffers;
    for (auto t : final_tensors)
    {
        final_buffers.insert(t->memory_offset()->get_base_object());
    }
    for (auto n : nodes)
    {
        if (SubGraphNode::is_subgraph_node(n) && !n->attrs().count(Constants::plugin_attrs_name))
        {
            auto sgn = std::static_pointer_cast<SubGraphNode>(n);
            graph_memory_allocate(m_allocator, sgn->subgraph()->nodes(), keep, final_tensors, consumer_cnt, memory_size, allocated, skip_tensors);
            continue;
        }
        if (n->type() == OpType_SubGraphInput || n->type() == OpType_InternalConstant)
        {
            continue;
        }

        auto mem_clock = m_allocator->get_clock();
        n->set_attr(Constants::mem_clock, mem_clock);
        for (auto t : n->outputs())
        {
            if (skip_tensors.count(t))
            {
                continue;
            }
            // INFO() << "allocate " << t->name();
            auto mmu = t->memory_offset()->get_mmu_object();
            auto ptr = t->memory_offset()->get_base_object();
            if (mmu)
            {
                if (mmu->base_offset->offset() >= 0xE0000000 && mmu->next_block->offset() >= 0xE0000000)
                {
                    continue;
                }
                else
                {
                    if (t && t->mem_size() > 0)
                    {
                        //allocate the second part for it
                        auto mmu_ddr = mmu->next_block;
                        if (!allocated.count(mmu_ddr->get_base_object())) // if the mmu_ddr is shared and allocated
                        {

                            mmu_ddr->align = MAX(mmu_ddr->get_max_aligned(), 16 * 1024);
                            mmu_ddr->relative_offset = 0;
                            mmu->next_block = m_allocator->allocate(aligned(mmu_ddr->size), nullptr);
                            // INFO() << "allocating " << t->name() << " size:" << mmu_ddr->size << " " << mmu_ddr;
                            mmu->next_block->relative_offset = 0;
                            mmu->next_block->align = mmu_ddr->align;
                            mmu->next_block->size = mmu_ddr->size;
                            allocated.insert(mmu->next_block->get_base_object());
                        }

                        continue;
                    }
                }
            }
            else
            {
                if (ptr->offset() > 0xE0000000)
                {
                    continue;
                }
            }
            if (final_tensors.count(t) || final_buffers.count(ptr))
            {
                m_allocator->enable_soc_sram(false); // the output tensor should located on ddr.
            }
            if (!allocated.count(ptr) && memory_size.count(ptr))
            {
                if (t->attrs().count("prior_in_gm"))
                {
                    ptr->set_mem_flag(MemoryObject::flag::prior_in_gm);
                    INFO() << "prior in gm " << t->name();
                }
                uint32_t real_size = memory_size[ptr];
                // INFO() << "allocating " << t->name() << " size:" << real_size << ", flag: " << ptr->mem_flag;
                m_allocator->allocate(aligned(real_size), ptr);
                allocated.insert(ptr->get_base_object());
            }
            t->memory_offset()->get_base_object()->align = t->memory_offset()->get_max_aligned();
            if (final_tensors.count(t) || final_buffers.count(ptr))
            {
                m_allocator->enable_soc_sram(true);
                final_tensors.erase(t);
            }
        }
        for (auto inp : n->inputs())
        {
            // INFO() << "free... " << inp->name();
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
                    // INFO() << "free " << inp->memory_offset()->get_base_object();
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
                // the input is not in this allocator
            }
        }
        // m_allocator->print();
        m_allocator->clock(); //for nodes in subgraph, each node is considered a clock
    }
}

static bool compare_tensor_name(const TensorPtr &t1, const TensorPtr &t2)
{
    return t1->name() < t2->name();
}

std::vector<MemoryObjectPtr> Scheduler::mem_allocate(const std::set<TensorPtr> &not_reuse_tensor, const std::set<TensorPtr> &inputs, const std::set<TensorPtr> &keeps_tensors, const std::set<TensorPtr> &skip_tensors)
{
    if (!m_graph)
    {
        return std::vector<MemoryObjectPtr>();
    }
    if (m_allocator)
    {
        delete m_allocator;
    }
    // MemFootPrintEstimation fp_estimation;
    // fp_estimation(m_graph);
    // auto res=fp_estimation.get_result();
    m_allocator = get_mem_allocator();
    MemoryObjectPtr ocm_buffer = nullptr;
    if (BuilderConfig::exist(Constants::ocm_size) && BuilderConfig::get<uint32_t>(Constants::ocm_size) > 0)
    {
        ocm_buffer = std::make_shared<MemoryObject>(0);
        ocm_buffer->size = BuilderConfig::get<uint32_t>(Constants::ocm_size);
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
    graph_consumer(keeps_tensors, keep, consumer_cnt, m_graph->nodes());
    //allocate sram
    bool disable_mmu = aipubt::BuilderConfig::get<bool>(Constants::disable_mmu);
    m_allocator->enable_soc_sram(false);
    auto soc_sram = get_external_sram();
    for (auto sram : soc_sram)
    {
        m_allocator->add_soc_sram(sram.first, sram.second, MemoryAllocator::SRAMLevel::L4);
    }
    auto target = Target::get_the_target();
    if (target->name() == "X1_1204" && BuilderConfig::get<bool>(Constants::enable_tcm))
    {
        //add 4M tcm for x1 soc
        m_allocator->add_soc_sram(0xd0000000, BuilderConfig::get<uint32_t>(Constants::tcm_size));
    }
    // TODO : each subgraph has their own sram address space,
    // and all subgraphs shared the same ddr address space.
    if (!disable_mmu)
    {
        auto reserve_orders = sram_optimize(m_graph, keeps_tensors);
        if (reserve_orders.size()) //TODO!!!! support mmu in X2
        {
            m_graph->topological_sort(reserve_orders);
        }
    }
    // allocate global tensors
    // auto target = Target::get_the_target();
    // for (auto gt : m_graph->global_tensors())
    // {
    //     if (gt.second)
    //     {
    //         if (gt.first == "mmu_reserve" || gt.second->mem_size() <= 0)
    //         {
    //             continue;
    //         }
    //         not_reuse.insert(gt.second);
    //     }
    // }
    // sort the pre_alloc set to make sure allocate is ordered
    std::vector<TensorPtr> not_reuse_sorted(not_reuse.begin(), not_reuse.end());
    std::sort(not_reuse_sorted.begin(), not_reuse_sorted.end(), compare_tensor_name);

    // allocate input tensors
    for (auto t : inputs)
    {
        auto ptr = t->memory_offset()->get_base_object();
        //check the input is in not reuse
        if (!not_reuse.count(t) && !allocated.count(ptr) && memory_size.count(ptr))
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
    graph_memory_allocate(m_allocator, m_graph->nodes(), keep, final_tensors, consumer_cnt, memory_size, allocated, skip_tensors);
    auto slots = m_allocator->slots();
    print_slots_size(slots);

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
        for (size_t i = 0; i < slots.size(); ++i)
        {
            if (ocm_buffer->is_shared_by(slots[i]))
            {
                slots.erase(slots.begin() + i);
                break;
            }
        }
        slots.insert(slots.begin(), ocm_buffer);
    }
    return slots;
}

std::vector<MemoryObjectPtr> Scheduler::global_memory_opt(std::vector<MemoryObjectPtr> &slots, const std::vector<uint32_t> &gm_region_sizes)
{
    std::vector<MemoryObjectPtr> ret;
    if (slots.empty() || gm_region_sizes.empty())
    {
        return ret;
    }
    std::sort(slots.begin(), slots.end(), [](MemoryObjectPtr a, MemoryObjectPtr b) {
        return (((float)(a->foot_print) / (float)(a->size)) > ((float)(b->foot_print) / (float)(b->size)));
    });
    std::set<MemoryObjectPtr> standalone_buffer; // all global tensors are standalone buffer
    if (m_graph)
    {
        for (auto kv : m_graph->global_tensors())
        {
            standalone_buffer.insert(kv.second->memory_offset()->get_base_object());
        }
    }

    std::vector<MemoryObjectPtr> not_merge;
    std::vector<std::vector<MemoryObjectPtr>> to_merge(gm_region_sizes.size());
    std::vector<uint32_t> current_size(gm_region_sizes.size(), 0);
    std::vector<uint32_t> alignment(gm_region_sizes.size(), 0);
    uint64_t footprint = 0;
    for (auto slot : slots)
    {
        if (standalone_buffer.count(slot)) // global tensor will has no "footprint" on record
        {
            not_merge.push_back(slot);
            continue;
        }
        bool can_merge = false;
        int32_t max_partial = 0;
        uint32_t max_partial_idx = 0;
        // try to fully cover
        for (uint32_t i = 0; i < gm_region_sizes.size(); ++i)
        {
            if (current_size[i] + slot->size < gm_region_sizes[i])
            {
                if (slot->align > alignment[i])
                {
                    alignment[i] = slot->align;
                }
                to_merge[i].push_back(slot);
                current_size[i] += slot->size + slot->align;
                footprint += slot->foot_print;
                can_merge = true;
                break;
            }
            if ((int64_t)gm_region_sizes[i] - current_size[i] > max_partial)
            {
                max_partial = gm_region_sizes[i] - current_size[i];
                max_partial_idx = i;
            }
        }
        if (!can_merge)
        {
            if (max_partial > 0 && slot->foot_print > 0) //when partial use, only foot_print>0 alow, to avoid some global var use gm
            {
                if (slot->align > alignment[max_partial_idx])
                {
                    alignment[max_partial_idx] = slot->align;
                }
                to_merge[max_partial_idx].push_back(slot);
                current_size[max_partial_idx] += slot->size + slot->align;
                footprint += slot->foot_print * (gm_region_sizes[max_partial_idx] - current_size[max_partial_idx]) / slot->size;
            }
            else
            {
                not_merge.push_back(slot);
                /* code */
            }
        }
    }
    for (uint32_t i = 0; i < to_merge.size(); ++i)
    {
        if (to_merge[i].empty())
        {
            continue;
        }
        auto region = MemoryObject::merge(to_merge[i]);
        ret.push_back(region);
        not_merge.push_back(region);
    }
    slots.clear();
    slots = std::move(not_merge);
    INFO() << "Global memory will reduce memory footprint(estimated): " << utils::get_human_readable_size(footprint);
    return ret;
}

bool Scheduler::gm_enhance()
{
    return true;
};

void Scheduler::print_gm_avg_usage(){

};
const std::vector<std::vector<NodePtr>> &Scheduler::get_levels()
{
    return m_levels;
}

void Scheduler::set_levels(const std::vector<std::vector<NodePtr>> &levels)
{
    m_levels = levels;
}

Scheduler::~Scheduler()
{
    if (m_allocator)
    {
        delete m_allocator;
        m_allocator = nullptr;
    }
}
}; // namespace aipubt
