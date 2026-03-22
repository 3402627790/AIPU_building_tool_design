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
#ifndef UTILS_UTILS_H
#define UTILS_UTILS_H
#include "core/logger.h"
#include "file_utils.h"
#include "json.h"
#include "os_utils.h"
#include "path_utils.h"
#include "sha256.h"
#include "utils/string_utils.h"
#include <stdint.h>
#include <string>

namespace utils
{

#define TIMEIT(a)                                                                                                     \
    {                                                                                                                 \
        auto _timeit_ = utils::TimeItCounter(#a " at " + std::string(__FILENAME__) + ":" + std::to_string(__LINE__)); \
        a;                                                                                                            \
    }
enum DataUnit
{
    DU_Undefined = 0,
    DU_B = 1,
    DU_KB = 2,
    DU_MB = 3,
    DU_GB = 4,
    DU_TB = 5,
};

uint64_t get_time_in_us();
std::string get_uuid();
uint32_t get_increase_id();
uint64_t get_increase_id64();
// convert a byte size to a human readbale size string
std::string get_human_readable_size(uint64_t s, DataUnit unit = DU_Undefined, uint32_t precision = 3);
DataUnit get_best_data_unit(uint64_t s);
bool is_pow_of2(long long n);

class TimeItCounter
{

    uint64_t m_t0;
    std::string m_msg;

public:
    TimeItCounter(const std::string &msg) : m_msg(msg)
    {
        m_t0 = get_time_in_us();
    }
    ~TimeItCounter();
};

}; //namespace utils

#endif //!UTILS_UTILS_H
