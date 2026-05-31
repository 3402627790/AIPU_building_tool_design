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
 * @Date: 2020-04-09 11:56:45
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-11-14 11:28:40
 */
#include "utils/utils.h"
#include "core/logger.h"
#include <iomanip>
#include <random>
#include <sstream>
#include <sys/time.h>

namespace utils
{
uint64_t get_time_in_us()
{
    struct timeval now;
    gettimeofday(&now, NULL);
    return now.tv_sec * 1000000 + now.tv_usec;
}
bool is_pow_of2(long long n)
{
    return (n > 0) && ((n & (n - 1)) == 0);
}
std::string get_uuid()
{
    static std::random_device rd;
    static std::mt19937 mt(rd());
    std::ostringstream oss;
    oss << std::hex;
    for (int i = 0; i < 6; ++i)
    {
        oss << mt();
    }
    oss << get_time_in_us();
    return oss.str();
}
uint32_t get_increase_id()
{
    static uint32_t i = 0;
    return i++;
}
uint64_t get_increase_id64()
{
    static uint64_t i = 0;
    return i++;
}

DataUnit get_best_data_unit(uint64_t s)
{
    DataUnit du;
    if (s >= (1ull << 40))
    {
        du = DU_TB;
    }
    else if (s >= (1ull << 30))
    {
        du = DU_GB;
    }
    else if (s >= (1ull << 20))
    {
        du = DU_MB;
    }
    else if (s >= (1ull << 10))
    {
        du = DU_KB;
    }
    else
    {
        du = DU_B;
    }

    return du;
}
std::string get_human_readable_size(uint64_t s, DataUnit unit, uint32_t precision)
{
    std::ostringstream oss;
    oss << std::fixed;
    oss << std::setprecision(precision);
    if (unit == DU_TB || (unit == DU_Undefined && s >= (1ull << 40)))
    {
        oss << s / ((double)(1ull << 40)) << "TB";
    }
    else if (unit == DU_GB || (unit == DU_Undefined && s >= (1ull << 30)))
    {
        oss << s / ((double)(1 << 30)) << "GB";
    }
    else if (unit == DU_MB || (unit == DU_Undefined && s >= (1ull << 20)))
    {
        oss << s / ((double)(1 << 20)) << "MB";
    }
    else if (unit == DU_KB || (unit == DU_Undefined && s >= (1ull << 10)))
    {
        oss << s / ((double)(1 << 10)) << "KB";
    }
    else
    {
        oss << s << "B";
    }
    return oss.str();
}

TimeItCounter::~TimeItCounter()
{
    auto t1 = get_time_in_us();
    INFO() << "[TIMEIT]" << m_msg << " costs " << ((double)(t1 - m_t0)) / 1000. << "ms";
}

}; //namespace utils