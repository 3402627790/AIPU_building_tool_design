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
 * @Date: 2022-05-13 14:12:11 
 * @Last Modified by: Tom Gao
 * @Last Modified time: 2022-08-04 01:16:04
 */
#include "mem_footprint_estimation.h"
#include "core/core.h"

namespace aipubt
{

bool MemFootPrintEstimation::operator()(GraphPtr g) const
{
    auto get_big_size = [](uint64_t cur_size, uint64_t big_size) -> uint64_t {
        uint64_t size = cur_size > big_size ? cur_size : 0;
        return size;
    };
    if (g)
    {
        m_res.clear();
        auto big_buffer_thres = aipubt::get_big_buffer_thresholds();
        auto numbers = big_buffer_thres.size();
        MemFPInfos total(numbers, {0, 0});
        for (auto n : g->nodes())
        {
            MemFPInfos node_rws;
            for (size_t i = 0; i < numbers; ++i)
            {
                auto big_buffer_size = big_buffer_thres[i];
                MemFPInfo node_rw = {0, 0};
                for (auto t : n->inputs())
                {
                    node_rw.first += get_big_size(t->mem_size(), big_buffer_size);
                }
                for (auto t : n->outputs())
                {
                    node_rw.second += get_big_size(t->mem_size(), big_buffer_size);
                }
                for (auto kv : n->constants())
                {
                    // constants should be on DDR, and could not be allocated in DTCM/SRAM
                    node_rw.first += kv.second->mem_size();
                }
                total[i].first += node_rw.first;
                total[i].second += node_rw.second;
                node_rws.push_back(node_rw);
            }
            m_res[n->name()] = node_rws;
        }
        m_res["_total"] = total;
    }

    return false;
}
}; // namespace aipubt