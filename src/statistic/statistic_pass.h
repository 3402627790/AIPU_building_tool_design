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
 * @Last Modified time: 2022-08-04 01:12:06
 */
#ifndef STATISTIC_STATISTIC_PASS_H
#define STATISTIC_STATISTIC_PASS_H
#include "core/basepass.h"
#include "core/core.h"
#include <algorithm>
#include <set>

/**
 * @brief This file is all passes apply for graph analysis, which would not change the graph at all!
 *        all data is store to graph's attribute, with key: `_analysis`
 *
 */
namespace aipubt
{

struct DDRFootprint
{
    uint64_t read;
    uint64_t write;
};
using DDRFootprints = std::vector<DDRFootprint>;

struct PerfDataItem
{
    double count_cycles;
    double read_bytes;
    double write_bytes;
    double aiff_cycles;
    double dma_cycles;
    double aiff_utilization;
    double dma_utilization;

    void reset()
    {
        count_cycles = 0;
        read_bytes = 0;
        write_bytes = 0;
        aiff_cycles = 0;
        dma_cycles = 0;
        aiff_utilization = 0;
        dma_utilization = 0;
    }
};
using PerfDataItems = std::map<BaseNodePtr, PerfDataItem>;

const std::vector<uint64_t> get_big_buffer_thresholds();

template <typename PassT, typename Result>
class StatisticPass : public PassBase<PassT>
{
public:
    const Result &get_result()
    {
        // std::cout<<" pass name: "<<PassBase<PassT>::name()<<"\n";
        return m_res;
    }
    virtual std::string summary()
    {
        return "";
    }

protected:
    mutable Result m_res;
};

}; //namespace aipubt

#endif //!STATISTIC_STATISTIC_PASS_H