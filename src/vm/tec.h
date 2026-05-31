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
 * @Date: 2024-12-13 10:05:59 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-12-16 14:04:18
 */
#pragma once
#include "memory.h"
#include <algorithm>
#include <array>
#include <core/fp16.h>
#include <memory>

namespace aipubt
{

union V256 {
    uint8_t u8[32];
    int8_t s8[32];
    uint16_t u16[16];
    int16_t s16[16];
    uint32_t u32[8];
    int32_t s32[8];
    float_t f32[8];
    float_t f32r;
    // half f16[16];
    // half f16r;
    // bfp16 bf16[16];
    // bfp16 bf16r;
};
struct ASX
{
    uint64_t as0 = 0;
    uint64_t as1 = 0;
    uint64_t as2 = 0;
    uint64_t as3 = 0;
};

constexpr int32_t RegNum = 32;
constexpr int32_t PregNum = 8;
constexpr int32_t TregNum = 32;
constexpr int32_t LMSIZE = 32 * 1024;

/**
 * @brief TecStatus a structure contains all tec private resource
 * 
 */
// TReg need 256bit alignment
class VM;
class TransCode;
class Core;
struct alignas(32) TecStatus
{
    V256 TReg[32] = {0};
    V256 PRegMask[8] = {0};
    uint32_t RReg[RegNum] = {0xfffffff0};
    uint32_t PReg[8] = {0};
    int loop_count = 0;
    uint8_t lm[LMSIZE] = {0};
    uint32_t pc = 0;
    ASX asx = {0};
    TransCode *tec = 0;
    Core *core = 0;
    std::shared_ptr<Memory> mem = 0;
    VM *host = 0;

    void gen_pmask()
    {
        PReg[7] = 0xffffffff;
        for (auto i = 0; i < 8; ++i)
        {
            for (auto j = 0; j < 32; ++j)
            {
                PRegMask[i].u8[j] = (PReg[i] & (1 << j)) ? 0xff : 0;
            }
        }
    }
    void set_mem(std::shared_ptr<Memory> m) { mem = m; }
    void run(uint32_t offset);
};
#define TOFFSET offsetof(TecStatus, TReg)
#define PMskOFFSET offsetof(TecStatus, PRegMask)
#define ROFFSET offsetof(TecStatus, RReg)
#define POFFSET offsetof(TecStatus, PReg)
#define LOOP_COUNT_OFFSET offsetof(TecStatus, loop_count)
#define ASX_OFFSET offsetof(TecStatus, asx)
} // namespace aipubt
