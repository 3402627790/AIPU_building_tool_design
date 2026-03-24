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
 * @Date: 2020-01-29 20:01:25 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-04-13 11:01:12
 */
#include "builder/target.h"
#include "core/config.h"
#include "core/core.h"
#include "mmu_allocator.h"
#include "scheduler.h"
#include "sram_allocator.h"
#include "text_generator.h"
#include "utils/utils.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace aipubt
{
inline size_t aligned(size_t s, size_t align = 256)
{
    return (s + align - 1) & ~(align - 1);
}
inline bool is_virtual_mem(NodePtr node)
{
    return (node->attrs().count(Constants::virtual_mem) && node->get_attr<bool>(Constants::virtual_mem));
}
inline bool is_node_support_mmu(NodePtr node)
{
    if (node->attrs().count(Constants::sram_free))
    {
        if (node->get_attr<bool>(Constants::sram_free))
        {
            return true;
        }
    }
    if (node->attrs().count(Constants::mmu_support))
    {
        return node->get_attr<bool>(Constants::mmu_support);
    }
    if (is_virtual_mem(node))
    {
        for (auto p : node->parents())
        {
            if (!is_node_support_mmu(p))
            {
                return false;
            }
        }
        return true;
    }
    return false;
}

inline bool is_support_sram(NodePtr node, size_t sram_size = 0)
{
    if (is_node_support_mmu(node) && node->children().size())
    {
        if (sram_size != 0 &&
            node->outputs().size() == 1 &&
            node->outputs()[0]->mem_size() > sram_size)
        // if the output size> sram size, we not checking the child
        // due the output will never put in sram
        {
            return true;
        }

        for (auto child : node->children())
        {
            if (!is_node_support_mmu(child))
            {
                return false;
            }
        }
        return true;
    }
    return false;
}
template <typename T>
inline bool is_parents_in_chain(NodePtr node, T &chain)
{
    for (auto p : node->parents())
    {
        if (!contains(chain, p))
        {
            return false;
        }
        if (is_virtual_mem(p))
        {
            if (!is_parents_in_chain(p, chain))
            {
                return false;
            }
        }
    }
    return true;
}

inline bool is_all_children_in_chain(NodePtr node, std::vector<NodePtr> &chain)
{
    for (auto c : node->children())
    {
        if (!contains(chain, c))
        {
            return false;
        }
    }
    return true;
}

inline bool is_all_children_not_vm(NodePtr node)
{
    if (node->children().empty())
    {
        return false;
    }
    for (auto c : node->children())
    {
        if (is_virtual_mem(c))
        {
            return false;
        }
    }
    return true;
}
static std::vector<std::vector<NodePtr>> find_sram_subgraph(GraphPtr g, size_t max_sram_size = 0)
{
    std::vector<NodePtr> sram_chain;
    std::vector<std::vector<NodePtr>> chains;
    for (auto node : *g)
    {
        if (is_support_sram(node, max_sram_size))
        {
            if (!is_parents_in_chain(node, sram_chain))
            {
                if (sram_chain.size() > 0)
                {
                    chains.push_back(sram_chain);
                }
                sram_chain.clear();
            }
            // if (is_virtual_mem(node))
            // {
            //     INFO() << node->name();
            // }
            if (is_virtual_mem(node) && (sram_chain.empty() || !is_all_children_not_vm(node))) // virtual mem must follow its parents
            {
            }
            else
            {
                sram_chain.push_back(node);
            }
        }
    }
    if (sram_chain.size() > 0)
    {
        chains.push_back(sram_chain);
    }
    // prune the chain for the vm node
    for (unsigned int i = 0; i < chains.size(); ++i)
    {
        sram_chain.clear();
        for (auto node : chains[i])
        {
            if (is_virtual_mem(node) && !is_all_children_in_chain(node, chains[i]))
            {
                continue;
            }
            sram_chain.push_back(node);
        }
        chains[i] = sram_chain;
    }
    //complete the chain
    for (unsigned int i = 0; i < chains.size(); ++i)
    {
        sram_chain.clear();
        for (auto node : chains[i])
        {
            sram_chain.push_back(node);
            for (auto child : node->children())
            {
                if (!contains(chains[i], child) &&
                    !contains(sram_chain, child) &&
                    !is_virtual_mem(child) && // we can not append a vm node!!!
                    is_parents_in_chain(child, sram_chain))
                {
                    sram_chain.push_back(child);
                }
            }
        }
        chains[i] = sram_chain;
    }
    //remove overlap
    std::set<NodePtr> viewed_nodes;
    for (unsigned int i = 0; i < chains.size(); ++i)
    {
        sram_chain.clear();
        for (auto node : chains[i])
        {
            if (!viewed_nodes.count(node))
            {
                sram_chain.push_back(node);
                viewed_nodes.insert(node);
            }
        }
        chains[i] = sram_chain;
    }
    return chains;
}

inline void print_slots_size(std::vector<MemoryObjectPtr> &slots)
{
    std::ostringstream oss;
    oss << std::hex << "[";
    for (auto s : slots)
    {
        oss << "0x" << s->size << ", ";
    }
    oss << "]";
    INFO() << oss.str();
}

inline bool is_in_keep(TensorPtr out, const std::set<MemoryObjectPtr> &keep)
{
    auto ptr = out->memory_offset()->get_base_object();
    if (contains(keep, ptr))
    {
        return true;
    }

    return false;
}

// return max tensor size
static size_t get_tensors_consumers_and_memory_size(std::vector<NodePtr> &chain,
                                                    std::map<MemoryObjectPtr, size_t> &consumer_cnt,
                                                    std::map<MemoryObjectPtr, size_t> &memory_size)
{
    size_t ret = 0;
    for (auto n : chain)
    {

        for (auto &inp : n->inputs())
        {
            auto ptr = inp->memory_offset()->get_base_object();
            if (!consumer_cnt.count(ptr))
            {
                consumer_cnt[ptr] = 0;
            }
            consumer_cnt[ptr]++;
        }
        for (auto &out : n->outputs())
        {

            auto ptr = out->memory_offset()->get_base_object();
            if (!memory_size.count(ptr))
            {
                memory_size[ptr] = 0;
            }
            size_t s = std::max(memory_size[ptr], out->mem_size());
            memory_size[ptr] = s;
            if (s > ret)
            {
                ret = s;
            }
        }
    }
    return ret;
}

// return the sram footprint for the chain
static size_t sram_subgraph_allocation(std::vector<NodePtr> &chain,
                                       std::map<MemoryObjectPtr, size_t> &consumer_cnt,
                                       std::map<MemoryObjectPtr, size_t> &memory_size,
                                       MMUAllocator *allocator,
                                       const std::set<MemoryObjectPtr> &keeping_mem)
{
    size_t ret = 0;
    std::set<NodePtr> ran_nodes;
    std::set<MemoryObjectPtr> allocated;
    std::map<MemoryObjectPtr, size_t> mmu_ddr_msize;
    auto target = Target::get_the_target();
    std::set<MemoryObjectPtr> failed_to_allocate;
    // std::set<NodePtr> ddr_only;

    auto is_in_failed_to_allocate = [&failed_to_allocate](NodePtr n) -> bool {
        bool ret = false;
        for (auto t : n->outputs())
        {
            auto ptr = t->memory_offset()->get_base_object();
            if (failed_to_allocate.count(ptr))
            {
                ret = true;
            }
        }
        return ret;
    };

    for (auto n : chain)
    {
        if (!is_all_children_in_chain(n, chain) || is_in_failed_to_allocate(n) || n->children().empty() || ran_nodes.count(n))
        {
            for (auto t : n->outputs())
            {
                auto ptr = t->memory_offset()->get_base_object();
                if (!allocated.count(ptr) && !failed_to_allocate.count(ptr))
                {
                    failed_to_allocate.insert(ptr);
                }
            }
            ran_nodes.insert(n);
        }
        else
        {
            std::set<MemoryObjectPtr> slow_objects;
            auto disable_spsram_optimization = aipubt::BuilderConfig::get<bool>(Constants::disable_spsram_optimization);
            if (!disable_spsram_optimization)
            {
                for (auto t : n->inputs())
                {
                    // record the original Memory Object, instead of the base_object
                    auto ptr = t->memory_offset();
                    slow_objects.insert(ptr);
                }
            }
            // allocate

            for (auto t : n->outputs())
            {
                if (is_in_keep(t, keeping_mem))
                {
                    continue;
                }

                auto ptr = t->memory_offset()->get_base_object();
                if (!allocated.count(ptr) && memory_size.count(ptr) && !failed_to_allocate.count(ptr))
                {
                    //TODO check if the node need to  dump/output
                    if (allocator->allocate(memory_size[ptr], ptr, slow_objects))
                    {
                        // use mmu
                        allocated.insert(ptr);
                    }
                    else
                    {
                        failed_to_allocate.insert(ptr);
                    }
                }
            }

            for (auto t : n->outputs())
            {
                auto ptr = t->memory_offset()->get_base_object();
                auto mmu = t->memory_offset()->get_mmu_object();

                if (allocated.count(ptr) || allocated.count(mmu))
                {
                    n->set_tag(n->tag() | BaseNode::SegMMU);
                    if (mmu && mmu->next_block && mmu->next_block->offset() < 0xe0000000) //make sure this is ddr
                    {
                        if (!mmu_ddr_msize.count(mmu->next_block))
                        {
                            mmu_ddr_msize[mmu->next_block] = 0;
                        }
                        mmu_ddr_msize[mmu->next_block] = MAX(mmu_ddr_msize[mmu->next_block], t->mem_size());
                        ret += mmu->base_offset->size;
                        // INFO()<<t->name()<<"  size:"<< t->mem_size()<<" "<<mmu->next_block<<" "<<mmu_ddr_msize[mmu->next_block];
                    }
                    else
                    {
                        ret += t->mem_size();
                    }
                }
            }
            ran_nodes.insert(n);
        }
        for (auto inp : n->inputs())
        {
            MemoryObjectPtr ptr = inp->memory_offset();

            while (!consumer_cnt.count(ptr) && ptr)
            {
                ptr = ptr->base_offset;
            }
            if (consumer_cnt.count(ptr))
            {
                consumer_cnt[ptr]--;
                if (consumer_cnt[ptr] <= 0)
                {

                    allocator->free(ptr);
                }
            }
        }
    }

    // free all chains memory
    for (auto n : chain)
    {
        for (auto t : n->outputs())
        {
            allocator->free(t->memory_offset());
        }
    }

    if ((target->isainfo().type == ISA_ZYv2) || (target->isainfo().type == ISA_ZYv3))
    {
        //make each ddr is a free ddr
        for (auto n : chain)
        {
            for (auto t : n->outputs())
            {
                auto ptr = t->memory_offset()->get_base_object();
                auto mmu = t->memory_offset()->get_mmu_object();

                if (allocated.count(ptr) || allocated.count(mmu))
                {
                    n->set_tag(n->tag() | BaseNode::SegMMU);
                    if (mmu && mmu->next_block && mmu->next_block->offset() < 0xe0000000) //make sure this is ddr
                    {
                        auto ddr = mmu->next_block;
                        auto new_ddr = std::make_shared<MemoryObject>(0, 0);
                        *new_ddr = *ddr;
                        new_ddr->base_offset = 0;
                        new_ddr->next_block = 0;
                        new_ddr->size = aligned(t->mem_size() - mmu->base_offset->size + mmu->relative_offset + 64, 16 * 1024);
                        mmu->next_block = new_ddr;
                    }
                }
            }
        }
    }
    return ret;
}

static void filter_unsupported_nodes(const std::vector<NodePtr> &nodes, std::set<MemoryObjectPtr> &allocated)
{
    std::set<NodePtr> ran_nodes;
    std::set<NodePtr> sram_free;
    auto is_all_child_ran = [&ran_nodes](NodePtr n) {
        for (auto c : n->children())
        {
            if (!ran_nodes.count(c))
            {
                return false;
            }
        }
        return true;
    };
    for (auto node : nodes)
    {
        // move unsupported nodes into allocated
        ran_nodes.insert(node);
        if (SubGraphNode::is_subgraph_node(node) && !node->attrs().count(Constants::plugin_attrs_name))
        {
            // move all nodes in loop/if subgraph into allocated
            auto sg = SubGraphNode::from_node(node)->subgraph();
            sg->walk_on_node([&allocated](NodePtr n) {
                for (auto out : n->outputs())
                {
                    allocated.insert(out->memory_offset()->get_base_object());
                }
            });
        }
        if (!is_support_sram(node))
        {
            if (sram_free.size())
            {
                for (auto n : sram_free)
                {
                    if (!is_all_child_ran(n))
                    {
                        for (auto out : n->outputs())
                        {
                            allocated.insert(out->memory_offset()->get_base_object());
                        }
                    }
                }
                sram_free.clear();
            }
            for (auto out : node->outputs())
            {
                allocated.insert(out->memory_offset()->get_base_object());
            }
        }
        else
        {
            sram_free.insert(node);
        }
    }
}

static void sram_opt_x2(GraphPtr g, const std::set<TensorPtr> &keeping_tensors, const std::set<TensorPtr> &skip_tensors)
{
    auto target = get_target();
    size_t base = 0xf8000000, size = 0;
    if (target->isainfo().type >= ISA_ZYv5)
    {
        size = target->gsram_size();
    }
    SRAMAllocator allocator(base, size);
    std::map<MemoryObjectPtr, int> consumer_cnt;
    std::map<MemoryObjectPtr, size_t> memory_size;
    std::set<MemoryObjectPtr> allocated, keep;
    std::set<TensorPtr> final_tensors = keeping_tensors;
    Scheduler::graph_memory(memory_size, g->nodes());
    Scheduler::graph_consumer(keeping_tensors, keep, consumer_cnt, g->nodes());

    filter_unsupported_nodes(g->nodes(), allocated);

    Scheduler::graph_memory_allocate(&allocator, g->nodes(), keep, final_tensors, consumer_cnt, memory_size, allocated, skip_tensors);
    allocator.slots();
}

std::vector<NodePtrList> sram_optimize(GraphPtr g, const std::set<TensorPtr> &keeping_tensors, const std::set<TensorPtr> &skip_tensors)
{
    auto target = Target::get_the_target();
    std::vector<NodePtrList> ret;
    if (BuilderConfig::get<bool>(Constants::disable_mmu))
    {
        return ret;
    }
    size_t max_sram_size = 0;
    if (target->isainfo().type >= ISA_ZYv5)
    {
        sram_opt_x2(g, keeping_tensors, skip_tensors);
        return ret;
    }
    auto chains = find_sram_subgraph(g, max_sram_size);
    for (auto chain : chains)
    {
        TensorPtr mmu_ddr = g->global_tensors()[Constants::mmu_reserve];
        if (!mmu_ddr)
        {
            mmu_ddr = std::make_shared<Tensor>(g->name() + "/mmurd/" + utils::get_uuid());
            g->global_tensors()[Constants::mmu_reserve] = mmu_ddr;
        }
        std::map<MemoryObjectPtr, size_t> consumer_cnt;
        std::map<MemoryObjectPtr, size_t> memory_size;
        auto ddr_size = get_tensors_consumers_and_memory_size(chain, consumer_cnt, memory_size);
        size_t max_ddrsize = 1024 * 1024;
        if ((target->isainfo().type == ISA_ZYv2) || (target->isainfo().type == ISA_ZYv3))
        {
            max_ddrsize = 64 * 1024 * 1024;
        }
        if (target->isainfo().type == ISA_ZY)
        {
            ddr_size = max_ddrsize;
        }
        else
        {
            ddr_size = ddr_size > max_ddrsize ? max_ddrsize : ddr_size;
        }

        MMUAllocator allocator(mmu_ddr, ddr_size);
        std::set<MemoryObjectPtr> keep_mems;
        for (auto t : keeping_tensors)
        {
            auto ptr = t->memory_offset()->get_base_object();
            keep_mems.insert(ptr);
        }
        auto opt_size = sram_subgraph_allocation(chain, consumer_cnt, memory_size, &allocator, keep_mems);
        if (opt_size > 0)
        {
            ret.push_back(chain);
        }
        // virtual_memory_remapping(chain);
    }
    return ret;
}
} // namespace aipubt
