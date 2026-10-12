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
 * @Author: Neo Wang 
 * @Date: 2020-01-29 14:27:29 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-04-12 15:58:20
 */
#ifndef BUILDER_SCHEDULER_H
#define BUILDER_SCHEDULER_H

#include "builder/mem_allocator.h"
#include "builder/optimizer.h"
#include "core/core.h"
#include <set>
namespace aipubt
{

/**
 * @brief The base/default class for scheduling node execution.
 * 
 */
class Scheduler
{
public:
    //for schdule the memory
    Scheduler(GraphPtr g = nullptr) : m_graph(g) {}
    virtual ~Scheduler();
    GraphPtr graph() const
    {
        return m_graph;
    }
    void set_graph(GraphPtr g)
    {
        m_graph = g;
    }
    virtual std::vector<MemoryObjectPtr> mem_allocate(const std::set<TensorPtr> &pre_alloc_tensors = std::set<TensorPtr>(),
                                                      const std::set<TensorPtr> &inputs = std::set<TensorPtr>(),
                                                      const std::set<TensorPtr> &keeps_tensors = std::set<TensorPtr>(),
                                                      const std::set<TensorPtr> &skip_tensors = {});
    const std::vector<MemoryObjectPtr> &get_gm_regions()
    {
        return m_gm_region_slots;
    }
    void set_gm_size(const std::vector<uint32_t> &s)
    {
        m_gm_size = s;
    }
    const std::vector<uint32_t> &gm_size()
    {
        return m_gm_size;
    }

    void print_slots_size(const std::vector<MemoryObjectPtr> &slots);
    static void graph_memory(std::map<MemoryObjectPtr, size_t> &memory_size,
                             const std::vector<NodePtr> &nodes);
    static void graph_memory_allocate(MemoryAllocator *m_allocator,
                                      const std::vector<NodePtr> &nodes,
                                      std::set<MemoryObjectPtr> &keep,
                                      std::set<TensorPtr> &final_tensors,
                                      std::map<MemoryObjectPtr, int> &consumer_cnt,
                                      std::map<MemoryObjectPtr, size_t> &memory_size,
                                      std::set<MemoryObjectPtr> &allocated,
                                      const std::set<TensorPtr> &skip_tensors = {});
    static void graph_consumer(const std::set<TensorPtr> &keeps_tensors,
                               std::set<MemoryObjectPtr> &keep,
                               std::map<MemoryObjectPtr, int> &consumer_cnt,
                               const std::vector<NodePtr> &nodes,
                               bool recurrently = true);
    virtual bool gm_enhance();
    virtual void print_gm_avg_usage();
    void set_levels(const std::vector<std::vector<NodePtr>> &levels);
    const std::vector<std::vector<NodePtr>> &get_levels();

protected:
    /**
     * @brief The gm optimization will try to merge slots to maximum the footprint on GM
     * 
     * @param slots the allocated slots of buffer, it may be changed after optimization
     * @param gm_region_sizes  the global memory each regions' size.
     * @return std::vector<MemoryObjectPtr>  the buffer which will use a relative gm region
     */
    std::vector<MemoryObjectPtr> global_memory_opt(std::vector<MemoryObjectPtr> &slots, const std::vector<uint32_t> &gm_region_sizes);
    std::vector<MemoryObjectPtr> m_gm_region_slots;
    std::vector<uint32_t> m_gm_size;
    MemoryAllocator *m_allocator = nullptr;
    std::vector<std::vector<NodePtr>> m_levels;
    GraphPtr m_graph;
};
std::vector<std::pair<uint32_t, uint32_t>> get_external_sram();
};     // namespace aipubt
#endif //!BUILDER_SCHEDULER_H
