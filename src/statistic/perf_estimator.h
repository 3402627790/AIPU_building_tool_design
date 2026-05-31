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
 * @Author: Lang Hu
 * @Date: 2023-03-03 15:30:23
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2023-03-03 15:30:23
 */
#ifndef STATISTIC_PERFORMANCE_ESTIMATION_H
#define STATISTIC_PERFORMANCE_ESTIMATION_H
#include "core/core.h"
#include "statistic_pass.h"

namespace aipubt
{

class PerfEstimator : public StatisticPass<PerfEstimator, PerfDataItems>
{
public:
    bool operator()(GraphPtr g) const override;

    std::string summary() override;

    std::string graph_path() const
    {
        return m_graph_path;
    }
    void graph_path(std::string t)
    {
        m_graph_path = t;
    }
    std::string weight_path() const
    {
        return m_weight_path;
    }
    void weight_path(std::string t)
    {
        m_weight_path = t;
    }

protected:
    std::string m_graph_path{""};
    std::string m_weight_path{""};
};

} // namespace aipubt

#endif //!STATISTIC_PERFORMANCE_ESTIMATION_H
