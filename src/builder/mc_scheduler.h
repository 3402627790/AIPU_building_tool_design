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
 * @Date: 2020-01-29 14:27:29 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-02-21 11:16:43
 */
#ifndef BUILDER_MC_SCHEDULER_H
#define BUILDER_MC_SCHEDULER_H

#include "core/core.h"
#include "mem_allocator.h"
#include "optimizer.h"
#include "scheduler.h"
#include <set>
namespace aipubt
{

/**
 * @brief The class for scheduling node execution on a multi-core target.
 * 
 */
class MCScheduler : public Scheduler
{
public:
    using Scheduler::Scheduler;
    //for schedule the memory
    virtual std::vector<MemoryObjectPtr> mem_allocate(const std::set<TensorPtr> &not_reuse_tensor = std::set<TensorPtr>(),
                                                      const std::set<TensorPtr> &inputs = std::set<TensorPtr>(),
                                                      const std::set<TensorPtr> &keeps_tensors = std::set<TensorPtr>(),
                                                      const std::set<TensorPtr> &skip_tensors = {}) override;
    virtual bool gm_enhance() override;
    virtual void print_gm_avg_usage() override;
    // std::map<MemoryObjectPtr, std::pair<uint32_t, uint32_t>> m_liveness;
    std::map<GraphPtr, std::map<MemoryObjectPtr, std::pair<uint32_t, uint32_t>>> m_liveness_map;

protected:
    void graph_memory_allocate_mc(MemoryAllocator *m_allocator,
                                  std::set<MemoryObjectPtr> &keep,
                                  std::set<TensorPtr> &final_tensors,
                                  std::map<MemoryObjectPtr, int> &consumer_cnt,
                                  std::map<MemoryObjectPtr, size_t> &memory_size,
                                  std::set<MemoryObjectPtr> &allocated);
};
};     // namespace aipubt
#endif //!BUILDER_MC_SCHEDULER_H
