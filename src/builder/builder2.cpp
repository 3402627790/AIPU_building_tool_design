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
 * @Date: 2020-01-29 14:26:28
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-07-24 10:42:31
 */
#include "builder/builder2.h"
#include "builder/aipu_elf.h"
#include "builder/ar/ar_reader.h"
#include "builder/linker/elfreader.h"
#include "builder/mc_scheduler.h"
#include "builder/ops/ops.h"
#include "builder/profiler.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/core.h"
#include "core/ifnode.h"
#include "core/loopnode.h"
#include "ds_code_generator.h"
#include "ro_generator.h"
#include "scheduler.h"
#include "text_generator.h"
#include "tongue/infershape/infershape.h"
#include "utils/utils.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <istream>
#include <stdlib.h>
#include <streambuf>
#include <time.h>
#include <vector>

// #define DUMP_DATA_BASE_ADDRESS 0x20000000
#define DUMP_RO_BASE_ADDRESS 0x10000000
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) > (b) ? (b) : (a))

inline size_t aligned(size_t s, size_t align = 256)
{
    return (s + align - 1) & ~(align - 1);
}

// #define DUMP_DATA_BASE_ADDRESS 0x0
#define RESERVE_RO_NUMBER 0

namespace aipubt
{

static const std::string INTERRUPT_HANDLER = "._INTERRUPT_HANDLER_";
static const std::string INTERRUPT_FP_HANDLER = "._INTERRUPT_FP_HANDLER_";
static const std::string TEXT_BEGIN_LABEL = "._TEXT_BEGIN_LABEL";
static const std::string COREDUMP_HANDLER = "._COREDUMP_HANDLER_";
static void gen_clear_fault_code(std::stringstream &m_stream, int reg = 3)
{
    m_stream << "{mfctrl0 r" << reg << ",0x90;}\n"
             << "{movh r" << reg << ",0x17D}\n"     // 0x17D0000 clear 24/22/21/20/19/18/16 bit
             << "{mtctrl0 r" << reg << ",0x90;}\n"; //clear fault
}

static void gen_flush_cache_code(std::stringstream &m_stream, int reg = 2)
{
    m_stream << "{mfctrl0 r" << reg << ",0x82;}\n"
             << "{orl r" << reg << ",r" << reg << ",4}\n" // flush cache
             << "{mtctrl0 r" << reg << ",0x82;}\n";
}

static void gen_check_cache_flush_done_code(std::stringstream &m_stream, int reg = 2)
{
    std::string flush_cache = "._FLUSH_CACHE_" + std::to_string(utils::get_increase_id64());
    m_stream << flush_cache << ":\n"
             << "{mfctrl0 r" << reg << ",0x82;}\n"
             << "{andl r" << reg << ",r" << reg << ",4;}\n"       //check
             << "{cbnz r" << reg << "," << flush_cache << ";}\n"; //check
}

inline int uint2int(uint32_t v, uint32_t bit = 16)
{
    v = (v & ((1 << bit) - 1));
    if (v & (1 << (bit - 1)))
    {
        return v | (((1 << (32 - bit)) - 1) << bit);
    }
    return v;
}

/**
 * @brief prefetch  instruction into icache
 * 
 * @param m_stream  the stream of asm generator
 * @param offset   the relative offset in byte to prefetch 
 * @param length    the length in byte to prefetch
 * @return * 
 */

// void gen_prefetch_icache(std::stringstream &m_stream, int32_t offset = 0, uint32_t length = 0)
// {
//     // only valide for x3+
//     if (Target::get_the_target()->isainfo().type < ISA_ZYv6)
//     {
//         return;
//     }
//     if (length == 0)
//     {
//         length = utils::split_string(m_stream.str(), "}\n").size() * 16;
//     }
//     // convert byte into cache line
//     const int line = 256; // for X3, one line is 256bytes
//     offset = (offset + line - 1) / line;
//     length = (length + line - 1) / line;
//     if (length > 255)
//     {
//         length = 255;
//     }
//     auto label_name = utils::get_uuid();
//     m_stream << "{mfctrl0 r0,0x14;}\n" // get TCB pointer
//              << "{ld r1,r0,80;}\n"     //get tec id
//              << "{lsr r1,r1,16;}\n"
//              << "{cbnz r1, .L_ONLY_TEC0_" << label_name << ";}\n";
//     if (offset > 127 || offset < -128)
//     {
//         m_stream << "{mov r3, " << (offset & 0xffff) << ";}\n";
//         if (offset >= 0 && offset < 0xffff)
//         {
//             m_stream << "{movh r3, " << (offset >> 16) << ";}\n";
//         }
//         m_stream << "{cachepm r3, 0, " << length << ";}\n";
//     }
//     else
//     {
//         m_stream << "{cachepm " << offset << ", 0, " << length << ";}\n";
//     }
//     m_stream << ".L_ONLY_TEC0_" << label_name << ":\n";
// }

static void gen_prefetch_icache_and_desc(std::stringstream &m_stream, const std::string &symbol, uint32_t length = 0, const std::vector<uint32_t> prefetch_desc_len = std::vector<uint32_t>())
{
    // only valide for x3+
    if (Target::get_the_target()->isainfo().type < ISA_ZYv6)
    {
        return;
    }
    if (length == 0)
    {
        length = utils::split_string(m_stream.str(), "}\n").size() * 16;
    }
    // convert byte into cache line
    const int line = 256; // for X3, one line is 256bytes
    length = (length + line - 1) / line;
    if (length > 255)
    {
        length = 255;
    }
    auto label_name = utils::get_uuid();
    m_stream << "{mfctrl0 r0,0x14;}\n" // get TCB pointer
             << "{ld r1,r0,80;}\n"     //get tec id
             << "{lsr r1,r1,16;}\n"
             << "{cbnz r1, .L_ONLY_TEC0_" << label_name << ";}\n";

    if (BuilderConfig::get<bool>(Constants::prefetch_desc) && prefetch_desc_len.size() == 3)
    {
        auto ctrl_len = prefetch_desc_len[0];
        // if (ctrl_len > 8192)
        // {
        //     ctrl_len = 4;
        // }
        // else
        if (ctrl_len > 4096)
        {
            ctrl_len = 3;
        }
        else if (ctrl_len > 2048)
        {
            ctrl_len = 2;
        }
        else if (ctrl_len > 1024)
        {
            ctrl_len = 1;
        }
        else
        {
            ctrl_len = 1;
        }
        m_stream
            << "{ld r2,r0,56;}\n"
            << "{add r2, r2, 256;}\n" //get pp
            << "{ld r3,r2, 55 ;}\n"   //get descriptor
            // << "{ld r4,r2,  56; ld r5,r2, 57 ;}\n"              //get descriptor
            << "{cacheop r3, 8, " << ctrl_len << ";}\n";
        // << "{cacheop r4, 8, 1;}\n" // for param and act only prefetch 1K
        // << "{cacheop r5, 8, 1;}\n";
    }
    m_stream << "{mov r3, %pc_rel_lo(" << symbol << ");}\n"
             << "{movh r3, %pc_rel_hi(" << symbol << ");}\n"
             << "{add r3, r3, 256;}\n"
             << "{lsr r3,r3,8;}\n"; //  line= 256, offset = (offset + line - 1) / line;
    // for  X3 ICA oustanding is 128: 128*512/8 = 8192, 8192/256=  32 line
    if (length <= 32)
    {
        m_stream << "{cachepm r3, 0, " << length << ";}\n";
    }
    else
    {
        while (length > 32)
        {
            m_stream << "{cachepm r3, 0, 32;}\n"
                     << "{add r3, r3, 32;}\n";
            length -= 32;
        }
        m_stream << "{cachepm r3, 0, " << length << ";}\n";
    }

    m_stream << ".L_ONLY_TEC0_" << label_name << ":\n";
}

static void generate_wait_IRQ_empty(std::stringstream &m_stream)
{
    std::string wai_irq = "._WAIT_IRQ_LBL_" + std::to_string(utils::get_increase_id64());
    m_stream << wai_irq << ":\n"
             << "{mfctrl0 r24,0x32;}\n"           // get IRQ signal
             << "{bfe r24,r24,16,0}\n"            // get IRQ_empty
             << "{cbz r24," << wai_irq << ";}\n"; // check if IRQ_empty is zero(IRQ is cleared)
}
static void set_timeout_reg(std::stringstream &m_stream)
{
    static char *timeout = getenv("AIPU_AXI_TIMEOUT");
    if (timeout)
    {
        uint32_t timeout_value = std::atoi(timeout);
        if (timeout_value > pow(2, 8) - 1)
        {
            WARN() << "timeout value config failed due to timeout value larger than " << pow(2, 8) - 1;
        }
        else
        {
            timeout_value = (timeout_value << 8) | 1;

            m_stream << "{mov r24, 0;}\n"
                     << "{movh r24, -4608;}\n" // 0xee00
                     << "{mov r23," << timeout_value << ";}\n"
                     << "{st r23, r24, 0x0;}\n"; // MIF Timeout Control Register
        }
    }
}

static void add_delay(std::stringstream &m_stream, bool enable)
{
    if (enable)
    {
        static std::string dead_loop = "._DEAD_LOOP_" + std::to_string(utils::get_increase_id64());
        static std::string delay_exit = "._DELAY_EXIT_" + std::to_string(utils::get_increase_id64());
        {
            m_stream << "{mfctrl0 r0,0x14;}\n"             // get TCB pointe
                     << "{ld r2,r0,124;}\n"                //get delay count
                     << "{cbz r2," << delay_exit << ";}\n" // let the program loop a bit
                     << dead_loop << ":\n"
                     << "{sub r2,r2,1;}\n"
                     << "{nop;}{nop;}{nop;}{nop;}{nop}\n"
                     << "{cbnz r2," << dead_loop << ";}\n" // let the program loop a bit
                     << delay_exit << ":\n";
        }
    }
}

#if 0
// define in Compass_NPU_driver/Linux/driver/umd/src/coredump/coredump.h
struct CoredumpBufferTec /* first address: 0x29A(NPU ELF machine code, 12bit alignment, pa|0x29A) */
                         /* 10bit actually for uimm10 of instruction. */
{
    uint32_t buf_valid;
    uint32_t buf_shared_address; /* [UW].CoredumpBufferShare address: 0x29A(NPU ELF machine code, 12bit alignment, pa|0x29A) */
    uint32_t gpr[32]; /* general purpose: r0-r31, r24:tcbp */
    uint32_t t_reg[32][8];
    uint32_t p_reg[8];
    struct CpxReg
    {
        uint32_t addr;
        uint32_t value;
    } cp0_reg[67]; /* 45 + 10 + 12 */
    struct MemBuffer
    {
        uint32_t addr;
        uint32_t size; /* [UW].tec local memory size */
    } lsram_info;
    uint32_t lsram_data[0]; /* tec local memory size */
};
struct CoredumpBufferShare
{
    uint32_t memory_count; /* [UW].all sm + all gm counter*/
    struct DdrInfo
    {
        uint32_t addr; /* [UW].only write gm address */
        uint32_t size; /* [UW].sm/gm memory size */
    } ddr_info[0]; /* all sm + all gm counter */
    uint32_t ddr_data[0]; /* all cores sm + gm size */
};
#endif

static void generate_coredump_handler(std::stringstream &m_stream)
{
    static std::string label_gpr = "._CD_LBL_" + std::to_string(utils::get_increase_id64());
    static std::string label_treg = "._CD_LBL_" + std::to_string(utils::get_increase_id64());
    static std::string label_preg = "._CD_LBL_" + std::to_string(utils::get_increase_id64());
    static std::string label_cp = "._CD_LBL_" + std::to_string(utils::get_increase_id64());
    static std::string label_lm = "._CD_LBL_" + std::to_string(utils::get_increase_id64());
    static std::string label_buffer_valid = "._CD_LBL_" + std::to_string(utils::get_increase_id64());
    static std::string label_mem = "._CD_LBL_" + std::to_string(utils::get_increase_id64());
    static std::string label_one_mem = "._CD_LBL_" + std::to_string(utils::get_increase_id64());
    static std::string label_mem_end = "._CD_LBL_" + std::to_string(utils::get_increase_id64());
    static std::string label_sm = "._CD_LBL_" + std::to_string(utils::get_increase_id64());
    static std::string label_gm = "._CD_LBL_" + std::to_string(utils::get_increase_id64());
    static std::string label_skip_sm = "._CD_LBL_" + std::to_string(utils::get_increase_id64());
    static std::string label_next_mem = "._CD_LBL_" + std::to_string(utils::get_increase_id64());
    static std::string label_cache_check = "._CD_LBL_" + std::to_string(utils::get_increase_id64());
    static std::string label_tec_cache_check = "._CD_LBL_" + std::to_string(utils::get_increase_id64());
    static std::string label_send_irq = "._CD_LBL_" + std::to_string(utils::get_increase_id64());
    static std::string label_end = "._CD_LBL_" + std::to_string(utils::get_increase_id64());
    uint32_t offset = Target::get_the_target()->isainfo().type == ISA_ZYv5 ? 120 : 88;
    m_stream << COREDUMP_HANDLER << ":\n"
             << "{mfctrl0 r24, 0x14;}\n"
             << "{ld r24, [r24 +" << offset << "];}\n"
             << "{sub r24, r24, 666;}\n" // 666(0x29a) : NPU ELF machine code
             << label_gpr << ":\n"
             << "{st r0, [r24 + 0x8];}\n"
             << "{st r1, [r24 + 0xc];}\n"
             << "{st r2, [r24 + 0x10];}\n"
             << "{st r3, [r24 + 0x14];}\n"
             << "{st r4, [r24 + 0x18];}\n"
             << "{st r5, [r24 + 0x1c];}\n"
             << "{st r6, [r24 + 0x20];}\n"
             << "{st r7, [r24 + 0x24];}\n"
             << "{st r8, [r24 + 0x28];}\n"
             << "{st r9, [r24 + 0x2c];}\n"
             << "{st r10, [r24 + 0x30];}\n"
             << "{st r11, [r24 + 0x34];}\n"
             << "{st r12, [r24 + 0x38];}\n"
             << "{st r13, [r24 + 0x3c];}\n"
             << "{st r14, [r24 + 0x40];}\n"
             << "{st r15, [r24 + 0x44];}\n"
             << "{st r16, [r24 + 0x48];}\n"
             << "{st r17, [r24 + 0x4c];}\n"
             << "{st r18, [r24 + 0x50];}\n"
             << "{st r19, [r24 + 0x54];}\n"
             << "{st r20, [r24 + 0x58];}\n"
             << "{st r21, [r24 + 0x5c];}\n"
             << "{st r22, [r24 + 0x60];}\n"
             << "{mfctrl0 r0, 0x14; nop; st r23, [r24 + 0x64]; nop;}\n"
             << "{st r0, [r24 + 0x68];}\n"
             << "{st r25, [r24 + 0x6c];}\n"
             << "{st r26, [r24 + 0x70];}\n"
             << "{st r27, [r24 + 0x74];}\n"
             << "{st r28, [r24 + 0x78];}\n"
             << "{st r29, [r24 + 0x7c];}\n"
             << "{st r30, [r24 + 0x80];}\n"
             << "{add r0, r24, 0x88; nop; st r31, [r24 + 0x84]; nop;}\n"
             << label_treg << ":\n"
             << "{st t0, [r0 + 0x00];}\n"
             << "{st t1, [r0 + 0x20];}\n"
             << "{st t2, [r0 + 0x40];}\n"
             << "{st t3, [r0 + 0x60];}\n"
             << "{st t4, [r0 + 0x80];}\n"
             << "{st t5, [r0 + 0xa0];}\n"
             << "{st t6, [r0 + 0xc0];}\n"
             << "{add r0, r0, 0x100; nop; st t7, [r0 + 0xe0]; nop}\n"
             << "{st t8, [r0 + 0x00];}\n"
             << "{st t9, [r0 + 0x20];}\n"
             << "{st t10, [r0 + 0x40];}\n"
             << "{st t11, [r0 + 0x60];}\n"
             << "{st t12, [r0 + 0x80];}\n"
             << "{st t13, [r0 + 0xa0];}\n"
             << "{st t14, [r0 + 0xc0];}\n"
             << "{add r0, r0, 0x100; nop; st t15, [r0 + 0xe0]; nop}\n"
             << "{st t16, [r0 + 0x00];}\n"
             << "{st t17, [r0 + 0x20];}\n"
             << "{st t18, [r0 + 0x40];}\n"
             << "{st t19, [r0 + 0x60];}\n"
             << "{st t20, [r0 + 0x80];}\n"
             << "{st t21, [r0 + 0xa0];}\n"
             << "{st t22, [r0 + 0xc0];}\n"
             << "{add r0, r0, 0x100; nop; st t23, [r0 + 0xe0]; nop}\n"
             << "{st t24, [r0 + 0x00];}\n"
             << "{st t25, [r0 + 0x20];}\n"
             << "{st t26, [r0 + 0x40];}\n"
             << "{st t27, [r0 + 0x60];}\n"
             << "{st t28, [r0 + 0x80];}\n"
             << "{st t29, [r0 + 0xa0];}\n"
             << "{st t30, [r0 + 0xc0];}\n"
             << "{add r0, r0, 0x100; nop; st t31, [r0 + 0xe0]; nop}\n"
             << label_preg << ":\n"
             << "{cpyp r1, p0.b;}\n"
             << "{nop; nop; st r1, [r0 + 0x0]; cpyp r1, p1.b;}\n"
             << "{nop; nop; st r1, [r0 + 0x4]; cpyp r1, p2.b;}\n"
             << "{nop; nop; st r1, [r0 + 0x8]; cpyp r1, p3.b;}\n"
             << "{nop; nop; st r1, [r0 + 0xc]; cpyp r1, p4.b;}\n"
             << "{nop; nop; st r1, [r0 + 0x10]; cpyp r1, p5.b;}\n"
             << "{nop; nop; st r1, [r0 + 0x14]; cpyp r1, p6.b;}\n"
             << "{nop; nop; st r1, [r0 + 0x18]; cpyp r1, p7.b;}\n"
             << "{add r0, r0, 0x20; nop; st r1, [r0 + 0x1c]; nop;}\n"
             << label_cp << ":\n"
             << "{mfctrl0 r2, 0x0; add r1, r31, 0x0; nop; nop;}\n"
             << "{add r3, r31, 0x001;  nop; st r1, [r0+0x00]; nop}\n"
             << "{mfctrl0 r4, 0x001;   nop; st r2, [r0+0x04]; nop}\n"
             << "{add r1, r31, 0x010;  nop; st r3, [r0+0x08]; nop}\n"
             << "{mfctrl0 r2, 0x010;   nop; st r4, [r0+0x0c]; nop}\n"
             << "{add r3, r31, 0x014;  nop; st r1, [r0+0x10]; nop}\n"
             << "{mfctrl0 r4, 0x014;   nop; st r2, [r0+0x14]; nop}\n"
             << "{add r1, r31, 0x030;  nop; st r3, [r0+0x18]; nop}\n"
             << "{mfctrl0 r2, 0x030;   nop; st r4, [r0+0x1c]; nop}\n"
             << "{add r3, r31, 0x031;  nop; st r1, [r0+0x20]; nop}\n"
             << "{mfctrl0 r4, 0x031;   nop; st r2, [r0+0x24]; nop}\n"
             << "{add r1, r31, 0x032;  nop; st r3, [r0+0x28]; nop}\n"
             << "{mfctrl0 r2, 0x032;   nop; st r4, [r0+0x2c]; nop}\n"
             << "{add r3, r31, 0x033;  nop; st r1, [r0+0x30]; nop}\n"
             << "{mfctrl0 r4, 0x033;   nop; st r2, [r0+0x34]; nop}\n"
             << "{add r1, r31, 0x040;  nop; st r3, [r0+0x38]; nop}\n"
             << "{mfctrl0 r2, 0x040;   nop; st r4, [r0+0x3c]; nop}\n"
             << "{add r3, r31, 0x041;  nop; st r1, [r0+0x40]; nop}\n"
             << "{mfctrl0 r4, 0x041;   nop; st r2, [r0+0x44]; nop}\n"
             << "{add r1, r31, 0x044;  nop; st r3, [r0+0x48]; nop}\n"
             << "{mfctrl0 r2, 0x044;   nop; st r4, [r0+0x4c]; nop}\n"
             << "{add r3, r31, 0x045;  nop; st r1, [r0+0x50]; nop}\n"
             << "{mfctrl0 r4, 0x045;   nop; st r2, [r0+0x54]; nop}\n"
             << "{add r1, r31, 0x080;  nop; st r3, [r0+0x58]; nop}\n"
             << "{mfctrl0 r2, 0x080;   nop; st r4, [r0+0x5c]; nop}\n"
             << "{add r3, r31, 0x081;  nop; st r1, [r0+0x60]; nop}\n"
             << "{mfctrl0 r4, 0x081;   nop; st r2, [r0+0x64]; nop}\n"
             << "{add r1, r31, 0x082;  nop; st r3, [r0+0x68]; nop}\n"
             << "{mfctrl0 r2, 0x082;   nop; st r4, [r0+0x6c]; nop}\n"
             << "{add r3, r31, 0x090;  nop; st r1, [r0+0x70]; nop}\n"
             << "{mfctrl0 r4, 0x090;   nop; st r2, [r0+0x74]; nop}\n"
             << "{add r1, r31, 0x091;  nop; st r3, [r0+0x78]; nop}\n"
             << "{mfctrl0 r2, 0x091;   nop; st r4, [r0+0x7c]; nop}\n"
             << "{add r3, r31, 0x092;  nop; st r1, [r0+0x80]; nop}\n"
             << "{mfctrl0 r4, 0x092;   nop; st r2, [r0+0x84]; nop}\n"
             << "{add r1, r31, 0x093;  nop; st r3, [r0+0x88]; nop}\n"
             << "{mfctrl0 r2, 0x093;   nop; st r4, [r0+0x8c]; nop}\n"
             << "{add r3, r31, 0x0d0;  nop; st r1, [r0+0x90]; nop}\n"
             << "{mfctrl0 r4, 0x0d0;   nop; st r2, [r0+0x94]; nop}\n"
             << "{add r1, r31, 0x0d1;  nop; st r3, [r0+0x98]; nop}\n"
             << "{mfctrl0 r2, 0x0d1;   nop; st r4, [r0+0x9c]; nop}\n"
             << "{add r3, r31, 0x0e0;  nop; st r1, [r0+0xa0]; nop}\n"
             << "{mfctrl0 r4, 0x0e0;   nop; st r2, [r0+0xa4]; nop}\n"
             << "{add r1, r31, 0x0e1;  nop; st r3, [r0+0xa8]; nop}\n"
             << "{mfctrl0 r2, 0x0e1;   nop; st r4, [r0+0xac]; nop}\n"
             << "{add r3, r31, 0x0e2;  nop; st r1, [r0+0xb0]; nop}\n"
             << "{mfctrl0 r4, 0x0e2;   nop; st r2, [r0+0xb4]; nop}\n"
             << "{add r1, r31, 0x0e3;  nop; st r3, [r0+0xb8]; nop}\n"
             << "{mfctrl0 r2, 0x0e3;   nop; st r4, [r0+0xbc]; nop}\n"
             << "{add r3, r31, 0x0e8;  nop; st r1, [r0+0xc0]; nop}\n"
             << "{mfctrl0 r4, 0x0e8;   nop; st r2, [r0+0xc4]; nop}\n"
             << "{add r1, r31, 0x0e9;  nop; st r3, [r0+0xc8]; nop}\n"
             << "{mfctrl0 r2, 0x0e9;   nop; st r4, [r0+0xcc]; nop}\n"
             << "{add r3, r31, 0x0ea;  nop; st r1, [r0+0xd0]; nop}\n"
             << "{mfctrl0 r4, 0x0ea;   nop; st r2, [r0+0xd4]; nop}\n"
             << "{add r1, r31, 0x0eb;  nop; st r3, [r0+0xd8]; nop}\n"
             << "{mfctrl0 r2, 0x0eb;   nop; st r4, [r0+0xdc]; nop}\n"
             << "{add r3, r31, 0x0f0;  nop; st r1, [r0+0xe0]; nop}\n"
             << "{mfctrl0 r4, 0x0f0;   nop; st r2, [r0+0xe4]; nop}\n"
             << "{add r1, r31, 0x1e0;  nop; st r3, [r0+0xe8]; nop}\n"
             << "{mfctrl0 r2, 0x1e0;   nop; st r4, [r0+0xec]; nop}\n"
             << "{add r3, r31, 0x1e4;  nop; st r1, [r0+0xf0]; nop}\n"
             << "{mfctrl0 r4, 0x1e4;   nop; st r2, [r0+0xf4]; nop}\n"
             << "{add r1, r31, 0x1e5;  nop; st r3, [r0+0xf8]; nop}\n"
             << "{mfctrl0 r2, 0x1e5;   nop; st r4, [r0+0xfc]; nop}\n"
             << "{add r0, r0, 0x100}\n"
             << "{add r3, r31, 0x1e6;  nop; st r1, [r0+0x00]; nop}\n"
             << "{mfctrl0 r4, 0x1e6;   nop; st r2, [r0+0x04]; nop}\n"
             << "{add r1, r31, 0x1e1;  nop; st r3, [r0+0x08]; nop}\n"
             << "{mfctrl0 r2, 0x1e1;   nop; st r4, [r0+0x0c]; nop}\n"
             << "{add r3, r31, 0x1e2;  nop; st r1, [r0+0x10]; nop}\n"
             << "{mfctrl0 r4, 0x1e2;   nop; st r2, [r0+0x14]; nop}\n"
             << "{add r1, r31, 0x1e3;  nop; st r3, [r0+0x18]; nop}\n"
             << "{mfctrl0 r2, 0x1e3;   nop; st r4, [r0+0x1c]; nop}\n"
             << "{add r3, r31, 0x1e8;  nop; st r1, [r0+0x20]; nop}\n"
             << "{mfctrl0 r4, 0x1e8;   nop; st r2, [r0+0x24]; nop}\n"
             << "{add r1, r31, 0x1e9;  nop; st r3, [r0+0x28]; nop}\n"
             << "{mfctrl0 r2, 0x1e9;   nop; st r4, [r0+0x2c]; nop}\n"
             << "{add r3, r31, 0x1eb;  nop; st r1, [r0+0x30]; nop}\n"
             << "{mfctrl0 r4, 0x1eb;   nop; st r2, [r0+0x34]; nop}\n"
             << "{add r1, r31, 0x1ec;  nop; st r3, [r0+0x38]; nop}\n"
             << "{mfctrl0 r2, 0x1ec;   nop; st r4, [r0+0x3c]; nop}\n"
             << "{add r3, r31, 0x1ed;  nop; st r1, [r0+0x40]; nop}\n"
             << "{mfctrl0 r4, 0x1ed;   nop; st r2, [r0+0x44]; nop}\n"
             << "{add r1, r31, 0x1ee;  nop; st r3, [r0+0x48]; nop}\n"
             << "{mfctrl0 r2, 0x1ee;   nop; st r4, [r0+0x4c]; nop}\n"
             << "{add r3, r31, 0x1ef;  nop; st r1, [r0+0x50]; nop}\n"
             << "{mfctrl0 r4, 0x1ef;   nop; st r2, [r0+0x54]; nop}\n"
             << "{add r1, r31, 0x1ea;  nop; st r3, [r0+0x58]; nop}\n"
             << "{mfctrl0 r2, 0x1ea;   nop; st r4, [r0+0x5c]; nop}\n"
             << "{add r3, r31, 0x000;  nop; st r1, [r0+0x60]; nop}\n"
             << "{mfctrl1 r4, 0x000;   nop; st r2, [r0+0x64]; nop}\n"
             << "{add r1, r31, 0x001;  nop; st r3, [r0+0x68]; nop}\n"
             << "{mfctrl1 r2, 0x001;   nop; st r4, [r0+0x6c]; nop}\n"
             << "{add r3, r31, 0x002;  nop; st r1, [r0+0x70]; nop}\n"
             << "{mfctrl1 r4, 0x002;   nop; st r2, [r0+0x74]; nop}\n"
             << "{add r1, r31, 0x003;  nop; st r3, [r0+0x78]; nop}\n"
             << "{mfctrl1 r2, 0x003;   nop; st r4, [r0+0x7c]; nop}\n"
             << "{add r3, r31, 0x010;  nop; st r1, [r0+0x80]; nop}\n"
             << "{mfctrl1 r4, 0x010;   nop; st r2, [r0+0x84]; nop}\n"
             << "{add r1, r31, 0x011;  nop; st r3, [r0+0x88]; nop}\n"
             << "{mfctrl1 r2, 0x011;   nop; st r4, [r0+0x8c]; nop}\n"
             << "{add r3, r31, 0x080;  nop; st r1, [r0+0x90]; nop}\n"
             << "{mfctrl1 r4, 0x080;   nop; st r2, [r0+0x94]; nop}\n"
             << "{add r1, r31, 0x081;  nop; st r3, [r0+0x98]; nop}\n"
             << "{mfctrl1 r2, 0x081;   nop; st r4, [r0+0x9c]; nop}\n"
             << "{add r3, r31, 0x082;  nop; st r1, [r0+0xa0]; nop}\n"
             << "{mfctrl1 r4, 0x082;   nop; st r2, [r0+0xa4]; nop}\n"
             << "{add r1, r31, 0x083;  nop; st r3, [r0+0xa8]; nop}\n"
             << "{mfctrl1 r2, 0x083;   nop; st r4, [r0+0xac]; nop}\n"
             << "{add r3, r31, 0x000;  nop; st r1, [r0+0xb0]; nop}\n"
             << "{mfctrl2 r4, 0x000;   nop; st r2, [r0+0xb4]; nop}\n"
             << "{add r1, r31, 0x002;  nop; st r3, [r0+0xb8]; nop}\n"
             << "{mfctrl2 r2, 0x002;   nop; st r4, [r0+0xbc]; nop}\n"
             << "{add r3, r31, 0x003;  nop; st r1, [r0+0xc0]; nop}\n"
             << "{mfctrl2 r4, 0x003;   nop; st r2, [r0+0xc4]; nop}\n"
             << "{add r1, r31, 0x004;  nop; st r3, [r0+0xc8]; nop}\n"
             << "{mfctrl2 r2, 0x004;   nop; st r4, [r0+0xcc]; nop}\n"
             << "{add r3, r31, 0x030;  nop; st r1, [r0+0xd0]; nop}\n"
             << "{mfctrl2 r4, 0x030;   nop; st r2, [r0+0xd4]; nop}\n"
             << "{add r1, r31, 0x031;  nop; st r3, [r0+0xd8]; nop}\n"
             << "{mfctrl2 r2, 0x031;   nop; st r4, [r0+0xdc]; nop}\n"
             << "{add r3, r31, 0x040;  nop; st r1, [r0+0xe0]; nop}\n"
             << "{mfctrl2 r4, 0x040;   nop; st r2, [r0+0xe4]; nop}\n"
             << "{add r1, r31, 0x042;  nop; st r3, [r0+0xe8]; nop}\n"
             << "{mfctrl2 r2, 0x042;   nop; st r4, [r0+0xec]; nop}\n"
             << "{add r3, r31, 0x043;  nop; st r1, [r0+0xf0]; nop}\n"
             << "{mfctrl2 r4, 0x043;   nop; st r2, [r0+0xf4]; nop}\n"
             << "{add r1, r31, 0x044;  nop; st r3, [r0+0xf8]; nop}\n"
             << "{mfctrl2 r2, 0x044;   nop; st r4, [r0+0xfc]; nop}\n"
             << "{add r0, r0, 0x100}\n"
             << "{add r3, r31, 0x070;  nop; st r1, [r0+0x00]; nop}\n"
             << "{mfctrl2 r4, 0x070;   nop; st r2, [r0+0x04]; nop}\n"
             << "{add r1, r31, 0x071;  nop; st r3, [r0+0x08]; nop}\n"
             << "{mfctrl2 r2, 0x071;   nop; st r4, [r0+0x0c]; nop}\n"
             << "{st r1, [r0+0x10]}\n"
             << "{st r2, [r0+0x14]}\n"
             << label_lm << ":\n"
             << "{add r2, r31, 0x0; nop; ld r1, [r0+0x1c]; nop}\n"             // r1 count=size/0x20
             << "{nop; lsr r1, r1, 5; movh r2, -1536; nop}\n"                  // r2 lm addr
             << "{cbz r1," << label_buffer_valid << ";}\n"                     // if size = 0
             << "{add r0, r0, 0x20; sub r1, r1, 0x1; st r2, [r0+0x18]; nop}\n" // r0 buffer address
             << "{loop r1}\n"
             << "{add r2, r2, 32; nop; nop; ld t0, [r2+0];}\n"
             << "{add r0, r0, 32; nop; st t0, [r0+0]; nop;}\n"
             << "{loopend}\n"
             << label_buffer_valid << ":\n"
             << "{add r2, r31, 1;}\n"
             << "{st r2, [r24+0];}\n" // valid size for CoredumpBufferTec
             << "{mfctrl0 r1, 0x082}\n"
             << "{orl r1, r1, 2}\n" // flush D-cache: write-back
             << "{mtctrl0 r1, 0x082;}\n"
             << label_tec_cache_check << ":\n"
             << "{mfctrl0 r1,0x82;}\n"
             << "{andl r1, r1, 2;}\n"                          //check
             << "{cbnz r1," << label_tec_cache_check << ";}\n" //check
             << label_mem << ":\n"
             << "{nop; nop; nop; ld r0, [r24+4];}\n"                  // load shared buffer
             << "{sub r0, r0, 666;}\n"                                // 666(0x29a) : NPU ELF machine code
             << "{bfe r1.u, r0.u, 0, 11;}\n"                          // check bit[0:11]
             << "{cbnz r1," << label_mem_end << ";}\n"                // don't coredump if buffer is invalid
             << "{mfctrl0 r3, 0x000; nop; nop; ld r1, [r0+0];}\n"     // r1: memory count, r3: tec id
             << "{add r0, r0, 0x4; lsl r2, r1, 3; nop; nop}\n"        // r2 = size of struct mem info
             << "{add r2, r2, r0; bfe r3.u,r3.u, 20, 3; nop; nop;}\n" // r0: mem info; r2 = data_addr; r3: core id
             << "{cmp.lt r6,r3.u,r1.u;}\n"                            // if (r3>r1) then r6=1; else then r6=0(skip sm)
             << label_one_mem << ":\n"
             << "{cbz r1," << label_mem_end << ";}\n"       // if mem count == 0 then goto the end
             << "{nop; nop; ld r4, [r0+0];ld r5, [r0+4]}\n" // r4:addr;r5:size
             << "{cbnz r4," << label_gm << ";}\n"           // r4:addr != 0 then mem is GM
             << "{cbz r6," << label_skip_sm << ";}\n"       // r6 == 0: core id > mem count
             << "{cbz r3," << label_sm << ";}\n"            // r3=0, find SM idx
             << "{sub r3, r3, 1;}\n"                        // next SM idx
             << label_skip_sm << ":\n"
             << "{add r2, r2, r5;}\n"             // next data buffer: pre-address + size
             << "{b " << label_next_mem << ";}\n" // next loop
             << label_sm << ":\n"
             << "{add r6, r31, 0x0; nop; movh r4, -2048; nop;}\n" // only dump one sm, r4=0xf8000000
             << label_gm << ":\n"
             << "{lsr r5, r5, 5}\n" // r4 = size / 0x20
             << "{sub r5, r5, 1}\n" // r4 = size / 0x20
             << "{loop r5}\n"
             << "{add r4, r4, 32; nop; nop; ld t0, [r4+0];}\n"
             << "{add r2, r2, 32; nop; st t0, [r2+0]; nop;}\n"
             << "{loopend}\n"
             << label_next_mem << ":\n"
             << "{add r0, r0, 8; sub r1, r1, 1; nop; nop;}\n" // next mem info
             << "{b " << label_one_mem << ";}\n"              // next loop
             << label_mem_end << ":\n"
             << "{mfctrl0 r1, 0x082}\n"
             << "{orl r1, r1, 2}\n" // flush D-cache: write-back
             << "{mtctrl0 r1, 0x082;}\n"
             << label_cache_check << ":\n"
             << "{mfctrl0 r1,0x82;}\n"
             << "{andl r1, r1, 2;}\n"                      //check
             << "{cbnz r1," << label_cache_check << ";}\n" //check
             << label_send_irq << ":\n"
             << "{mfctrl0 r1, 0x032}\n"
             << "{bfe r0.u, r1.u, 16, 0;}\n"           // check irq empty(=1)
             << "{cbz r0," << label_send_irq << ";}\n" // wait for irq empty == 0
             << "{bfs r0,r31, 27, 0;}\n"               // ignore the origin value of CP0:0x33
             << "{bfs r1,r1, 31, 0;}\n"
             << "{mtctrl0 r0, 0x033}\n"
             << "{mtctrl0 r1, 0x032}\n"
             << label_end << ":\n"
             << "{wfe r0, 0;}\n"
             << "{exit;}\n"
             << "{irtn;}\n";
}
static void generate_interrupt_handler(std::stringstream &m_stream)
{
    // IRQ flag bit
    // hardware exception 31
    // software exception 30
    // real-time print 29
    // profiler buffer overflow 28
    // coredump debug 27
    // real-time dump tensor 26
    // reserve 25:16
    // content 15:0

    static std::string coredump_lbl = "._SKIP_LBL_" + std::to_string(utils::get_increase_id64());
    static std::string skip_coredump = "._SKIP_LBL_" + std::to_string(utils::get_increase_id64());
    static std::string skip = "._SKIP_LBL_" + std::to_string(utils::get_increase_id64());
    static std::string exit_lbl = "._SKIP_LBL_" + std::to_string(utils::get_increase_id64());
    static std::string check_vfpu_lbl = "._CK_VFPU_LBL_" + std::to_string(utils::get_increase_id64());
    static std::string enable_FP = "._EN_FP_LBL_" + std::to_string(utils::get_increase_id64());
    static std::string dead_loop = "._DEAD_LOOP_" + std::to_string(utils::get_increase_id64());
    uint32_t offset = Target::get_the_target()->isainfo().type == ISA_ZYv5 ? 120 : 88;
    m_stream << ".global _start\n"
             << "_start:\n"
             << "{cbz r31," << skip << ";}\n"
             // interrupt handler implementation~
             << INTERRUPT_HANDLER << ":\n"
             << "{nop;}{nop;}{nop;}{nop;}{nop}\n" // wait a little bit
             << coredump_lbl << ":\n"
             << "{mfctrl0 r24,0x14;}\n"                    // get TCBP
             << "{ld r24, [r24 +" << offset << "];}\n"     // get coredump buffer
             << "{bfe r24.u, r24.u, 0, 11;}\n"             // check buffer address
             << "{sub r24, r24, 666;}\n"                   // 666(0x29a) : NPU ELF machine code
             << "{cbnz r24," << skip_coredump << ";}\n"    // don't coredump if bit[0:10] is not machine code.
             << "{mfctrl0 r24, 0x90;}\n"                   // get CP0 TEC status
             << "{bfe r24.u, r24.u, 16, 15;}\n"            // get TEC fault
             << "{cbz r24," << COREDUMP_HANDLER << ";}\n"  // coredump if not TEC fault is empty
             << "{bfe r24.u, r24.u, 5, 1;}\n"              // get HTS/LSU fault
             << "{cbnz r24," << COREDUMP_HANDLER << ";}\n" // coredump if hts/lsu fault
             << skip_coredump << ":\n"
             << "{mfctrl0 r24,0x32;}\n"                    // get IRQ signal
             << "{bfe r24,r24,16,0}\n"                     // get IRQ_empty
             << "{cbz r24," << INTERRUPT_HANDLER << ";}\n" // check if IRQ_empty is zero(IRQ is cleared)
             << "{mfctrl0 r24,0x90;}\n"                    // get TEC status
             << "{bfe r24,r24,16,15}\n"                    // hight 16 bit is the fault reason
             << "{movh r24, -32767;}\n"                    // bit31 is for hw exception
             << "{mtctrl0 r24,0x33;}\n"                    // IRQ flag
             << "{movh r24, -32767;}\n"                    // IRQ signal
             << "{mtctrl0 r24,0x32;}\n"                    // IRQ
             << "{mov r24, 0;}\n"
             << "{movh r24, 0;}\n"
             << dead_loop << ":\n"
             << "{cbz r24," << dead_loop << ";}\n" // let the program hung forever
             << "{wfe r0, 0;}\n"
             << "{exit;}\n"
             << "{irtn;}\n"

             //FP exception is here
             << INTERRUPT_FP_HANDLER << ":\n"
             // use r24 as temp reg
             << "{mfctrl1 r24,0x11;}\n"                 // get SPU FP status
             << "{bfe r24, r24, 16,0;}\n"               //get the 16bit
             << "{cbz r24," << check_vfpu_lbl << ";}\n" //if not a decode error, exit
             << "{mfctrl0 r24,0x1e3;}\n"                //get FPU feature
             << "{bfe r24, r24, 0,31;}\n"               //get the bit if support FP TODO!!
             << "{cbz r24," << exit_lbl << ";}\n"       //if not support, exit
             << "{cbz r31," << enable_FP << ";}\n"      //if  support, enable

             << check_vfpu_lbl << ":\n"
             << "{mfctrl1 r24,0x3;}\n"            // get VPU FP status
             << "{bfe r24, r24, 16,0;}\n"         //get the 16bit
             << "{cbz r24," << exit_lbl << ";}\n" //if not a decode error, exit
             << "{mfctrl0 r24,0x1e3;}\n"          //get FPU feature
             << "{bfe r24, r24, 0,31;}\n"         //get the bit if support FP TODO!!
             << "{cbz r24," << exit_lbl << ";}\n" //if not support, exit

             << enable_FP << ":\n"
             << "{mfctrl1 r24, 0x2;}\n"
             << "{bfs r24, r24,31,0;}\n"
             << "{mtctrl1 r24, 0x2;}\n"
             << "{mfctrl1 r24, 0x10;}\n"
             << "{bfs r24,r24, 31, 0;}\n"
             << "{mtctrl1 r24, 0x10;}\n"
             << "{mfctrl0 r24, 0x14;}\n" // recover r24
             << "{irtn;}\n"

             << exit_lbl << ":\n"
             << "{mfctrl0 r24, 0x14;}\n"                  // get TCBP
             << "{ld r24, [r24+" << offset << "];}\n"     // get coredump buffer
             << "{bfe r24.u, r24.u, 0, 11;}\n"            // check buffer address
             << "{sub r24, r24, 666;}\n"                  // 666(0x29a) : NPU ELF machine code
             << "{cbz r24," << COREDUMP_HANDLER << ";}\n" // coredump if buffer has machine code
             << "{wfe r0, 0;}\n"
             << "{exit;}\n"
             << "{irtn;}\n";

    generate_coredump_handler(m_stream);

    m_stream << skip << ":\n"
             << "{wfe r0, 0;}\n"
             << "{exit;}\n";
}

static void generate_hw_debug_info(uint32_t gid, std::stringstream &m_stream, uint32_t op_id = 0)
{
    // the debug info is a uint32_t number to record current gid and op id
    // the start offset is  0xc1000000, for multicore case:
    // core 0: 0xc100 0000
    // core 1: 0xc100 0004
    // core 2: 0xc100 0008
    // core n: 0xc100 0000+n*4

    if (BuilderConfig::get<bool>("DEBUG"))
    {
        uint16_t l = op_id & 0xffff;
        int16_t op_id_low = (int16_t)l;
        l = gid & 0xffff;
        int16_t gid_low = (int16_t)l;
        // we suppose core id has updated to TCB
        auto label_name = utils::get_uuid();
        m_stream << "{mfctrl0 r0,0x14;}\n" // get TCB pointer
                 << "{ld r1,r0,80;}\n"     //get tec id
                 << "{lsr r1,r1,16;}\n"
                 << "{cbnz r1, .L_ONLY_TEC0_" << label_name << ";}\n"
                 << "{ld r2,r0,52;}\n"                //get tcb stack pointer
                 << "{mov r0," << op_id_low << ";}\n" //
                 << "{movh r0," << gid_low << ";}\n"
                 << "{st r0,r2, 8;}\n" // store to sp+8
                 << ".L_ONLY_TEC0_" << label_name << ":\n";
    }
}

static void generate_prologue(std::stringstream &m_stream)
{
    m_stream << "{st ra,r29,4;}\n";
}

static void generate_dump_rt_tensor(std::stringstream &m_stream)
{
    m_stream << "{movh r24, 1024;}\n"    // bit26 is for dump realtime tensor
             << "{mtctrl0 r24,0x33;}\n"  // IRQ flag
             << "{movh r24, -32767;}\n"  // IRQ signal
             << "{mtctrl0 r24,0x32;}\n"; // IRQ
}

void generate_pp_updater(std::stringstream &m_stream, bool prefetch)
{
    int layer_ro_size = 256; // bytes
    m_stream
        << "{mfctrl0 r0,0x14;}\n" // get TCB pointer
        << "{ld r1,r0,56;}\n"     // get pp_base
        << "{add r1,r1," << layer_ro_size << "}\n";
    if (prefetch)
    {
        m_stream << "{cacheop r1,8,1;}\n"; // prefetch 1K=4 layers
    }
    m_stream << "{st r1, r0, 56;}\n"; //update TCB PP

    //TODO disbale FPU here

    //if debug
    if (BuilderConfig::get<bool>("DEBUG"))
    {
        auto label_name = utils::get_uuid();
        m_stream
            // << "{mfctrl0 r0,0x14   ;  }\n" // get TCB pointer
            << "{ld r1,r0,80;}\n" //get tec id
            << "{lsr r1,r1,16;}\n"
            << "{cbnz r1, .L_ONLY_TEC0_" << label_name << ";}\n"
            << "{ld r1,r0,52;}\n" //get sp
            << "{mov r2, 0;}\n"
            << "{movh r2, -16128;}\n"
            << "{ld r0,r1, 8;}\n"  // ld op id
            << "{add r0,r0, 1;}\n" // update op id
            << "{st r0,r1, 8;}\n"  // store back to stack
            << "{st r0,r2, 0;}\n"  //  store to 0xc1000 0000 to trigger emulator
            << ".L_ONLY_TEC0_" << label_name << ":\n";
    }
}

static void generate_epilogue(std::stringstream &m_stream)
{
    m_stream << "{ld ra,r29,4;}\n"
             << "{br ra;}\n";
}

static bool is_control_flow_code(CodeTreePtr code)
{
    static const std::set<OpType> ctrl_types = {
        OpType_Loop,
        OpType_If,
    };
    if (!code->node.expired() && ctrl_types.count(code->node.lock()->type()))
    {
        return true;
    }
    return false;
}

static inline bool is_dummy_code(CodeTreePtr code)
{
    return code->value == "_aipubt_dummy_kernel";
}

static inline std::string get_code_value(CodeTreePtr code)
{
    return is_control_flow_code(code) ? ".aipubt_func" + code->value : code->value;
}

static void generate_call(std::stringstream &m_stream, const std::string &func, size_t reg = 21)
{
    m_stream << "{mov r" << reg << ", %pc_rel_lo(" << func << ");}\n"
             << "{movh r" << reg << ", %pc_rel_hi(" << func << ");}\n"
             << "{brrl r" << reg << ";}\n";
}

static void generate_tec_sync(std::stringstream &m_stream, size_t reg = 21)
{
    // m_stream << "{mov r" << reg << ",0;}\n"; // wait all internal events
    m_stream << "{mov r" << reg << ",-33;}\n"; // 0xffffffdf: only wait lsu idle

    m_stream << "{wfe r" << reg << ",0;}\n"
             << "{wfe r" << reg << ",2;}\n";
}

void GraphELFBuilder::prefetch_cache(std::stringstream &m_stream, const std::string &symbol, const std::vector<uint32_t> prefetch_desc_len)
{
    if (Target::get_the_target()->isainfo().type < ISA_ZYv6)
    {
        return;
    }
    gen_prefetch_icache_and_desc(m_stream, symbol, get_func_size(symbol), prefetch_desc_len);
}

void GraphELFBuilder::gen_loop_kernel_asm(std::stringstream &m_stream,
                                          const std::string &kernel,
                                          const DSCodePkg &codes,
                                          bool prefetch)
{
    if (codes.ds[0].size() || codes.ds[1].size())
    {
        generate_tec_sync(m_stream);
        if (codes.ds[0].size())
        {
            // m_stream << "{bl " << codes.ds[0] << ";}\n"; // backup ro and desc
            generate_call(m_stream, codes.ds[0]);
        }
        if (codes.ds[1].size())
        {
            // m_stream << "{bl " << codes.ds[1] << ";}\n"; // modify ro and desc
            generate_call(m_stream, codes.ds[1]);
        }
        generate_tec_sync(m_stream);
    }
    // m_stream << "{bl " << kernel << ";}\n";
    generate_call(m_stream, kernel);

    if (codes.mb[0].size()) // loop max-batch
    {
        auto label_id = std::to_string(utils::get_increase_id64());
        auto loop_label = "._loop_lbl_" + label_id;
        auto loop_if_label = "._loop_if_lbl_" + label_id;
        auto loop_else_label = "._loop_else_lbl_" + label_id;
        auto exit_label = "._loop_exit_lbl_" + label_id;
        m_stream << codes.mb[0] // compute loop count
                 << "{st r0,r29,12;}\n"
                 << loop_label << ":\n"
                 << "{ld r0,r29,12;}\n"
                 << "{cmp.eq r1,r0,0; sub r0,r0,1;}\n"
                 << "{cbnz r1," << exit_label << ";}\n"
                 << "{cmp.eq r1,r0,0; st r0,r29,12;}\n"
                 << "{cbnz r1," << loop_if_label << ";}\n";
        // m_stream << "{bl " << codes.mb[1] << ";}\n"; // modify addrs
        generate_call(m_stream, codes.mb[1]);
        m_stream << "{b " << loop_else_label << ";}\n"
                 << loop_if_label << ":\n";
        // m_stream << "{bl " << codes.mb[2] << ";}\n"; // modify last batch addrs
        generate_call(m_stream, codes.mb[2]);
        m_stream << loop_else_label << ":\n";
        generate_tec_sync(m_stream);
        // m_stream << "{bl " << kernel << ";}\n";
        generate_call(m_stream, kernel);
        m_stream << "{b " << loop_label << ";}\n"
                 << exit_label << ":\n";
    }

    if (codes.ds[2].size()) // backup output shape if necessary
    {
        m_stream << codes.ds[2];
    }

    bool dump_rt_tensor = BuilderConfig::get<bool>(Constants::dump_rt_output_tensors) || BuilderConfig::get<bool>(Constants::dump_rt_input_tensors) || (BuilderConfig::get<int>(Constants::dump_rt_tensor_id) != -1);
    if (dump_rt_tensor)
    {
        auto label_name = utils::get_uuid();
        m_stream
            << "{mfctrl0 r0,0x14;}\n" // get TCB pointer
            << "{ld r1,r0,80;}\n"     //get tec id
            << "{lsr r1,r1,16;}\n"
            << "{cbnz r1, .L_ONLY_TEC0_" << label_name << ";}\n";
        generate_dump_rt_tensor(m_stream);
        generate_wait_IRQ_empty(m_stream);
        m_stream << ".L_ONLY_TEC0_" << label_name << ":\n";
    }

    if (codes.ds[0].size()) // restore and update TCB pp_base
    {
        std::string func_name = DSCodeGenerator::restore_update_pp_fn;
        if (utils::startswith(codes.ds[0], "_backup_param_da"))
        {
            func_name = DSCodeGenerator::restore_update_pp_da_fn;
        }
        // m_stream << "{bl " << func_name << ";}\n";
        generate_call(m_stream, func_name);
    }
    else
    {
        generate_pp_updater(m_stream, prefetch);
    }
}

static void gen_func_ender_asm(std::stringstream &m_stream, const std::string &func_name)
{
    auto label_name = ".Lfunc_ender_" + std::to_string(utils::get_increase_id64()) + func_name;
    m_stream << label_name << ":\n";
    m_stream << ".size	" << func_name << ", " << label_name << "-" << func_name << "\n";
}

void GraphELFBuilder::gen_func_asm(std::stringstream &m_stream,
                                   CodeTreePtr code,
                                   std::map<int, DSCodePkg> &code_map,
                                   int ro_index)
{
    static const std::set<std::string> internal_kernels = {
        "_aipubt_cpy_weight_to_sm",
        "_aipubt_cpy_weight_to_sm_wait",
        "_aipubt_cpy_weight_to_gm",
        "_aipubt_cpy_weight_to_gm_wait",
        "_aipubt_cpy_data_to_ddr",
        "_aipubt_cpy_data_to_ddr_wait",
        "_aipubt_tec_sync",
    };

    if (code->is_leaf())
    {
        return;
    }

    std::function<uint32_t(CodeTreePtr, std::map<CodeTreePtr, uint32_t> &)> dfs;
    dfs = [&dfs](CodeTreePtr code, std::map<CodeTreePtr, uint32_t> &res) {
        if (res.count(code))
        {
            return res[code];
        }
        uint32_t ro_size = 0;
        if (code->is_leaf())
        {
            if (code->value != Constants::profiler_kernel &&
                !internal_kernels.count(code->value) &&
                !is_dummy_code(code))
            {
                ro_size = DEFAULT_RO_NUMBER_PER_LAYER;
            }
        }
        else
        {
            for (uint32_t i = 0; i < code->codes.size(); ++i)
            {
                ro_size += dfs(code->codes[i], res);
            }
        }
        if (is_control_flow_code(code))
        {
            ro_size += 2 * DEFAULT_RO_NUMBER_PER_LAYER;
        }
        res[code] = ro_size;
        return ro_size;
    };
    std::map<CodeTreePtr, uint32_t> ro_size_map;
    dfs(code, ro_size_map);

    std::string func_name = ".aipubt_func" + code->value;
    m_stream << ".type " << func_name << ", function\n";
    m_stream << func_name << ":\n";

    auto label_id = std::to_string(utils::get_increase_id64());
    auto loop_entry_label = "._loop_entry_lbl_" + label_id;
    auto loop_exit_label = "._loop_exit_lbl_" + label_id;
    auto else_entry_label = "._else_entry_lbl_" + label_id;
    auto if_exit_label = "._if_exit_lbl_" + label_id;

    auto is_branch_separator = [](CodeTreePtr code, CodeTreePtr c) -> bool {
        if (!code->node.expired() && !c->node.expired())
        {
            auto n = code->node.lock();
            if (n->type() == OpType_If)
            {
                auto sgn = std::dynamic_pointer_cast<IfNode>(n);
                if (sgn->sep() == c->node.lock())
                {
                    return true;
                }
            }
        }
        return false;
    };
    auto split_imm = [](uint32_t offset, int16_t &low, int16_t &high) {
        uint16_t half = offset & 0xffff;
        low = *(int16_t *)(&half);
        half = (offset >> 16) & 0xffff;
        high = *(int16_t *)(&half);
    };

    if (is_control_flow_code(code))
    {
        auto optype = code->node.lock()->type();
        int16_t low = 0, high = 0;
        if (optype == OpType_Loop)
        {
            uint32_t offset = ro_size_map[code] - DEFAULT_RO_NUMBER_PER_LAYER;
            split_imm(4 * offset, low, high);
            m_stream << "{add r0,ra,0; mov r1," << low << ";}\n";
            if (low < 0 || high != 0)
            {
                m_stream << "{movh r1," << high << ";}\n";
            }
            // m_stream << "{bl _forrange_start_fn;}\n";
            generate_call(m_stream, "_forrange_start_fn");
            m_stream << "{cbnz r0," << loop_exit_label << ";}\n"
                     << loop_entry_label << ":\n";
        }
        else if (optype == OpType_If)
        {
            uint32_t offset = DEFAULT_RO_NUMBER_PER_LAYER;
            for (auto c : code->codes)
            {
                offset += ro_size_map[c];
                if (is_branch_separator(code, c))
                {
                    break;
                }
            }
            split_imm(4 * offset, low, high);
            m_stream << "{add r0,ra,0; mov r1," << low << ";}\n";
            if (low < 0 || high != 0)
            {
                m_stream << "{movh r1," << high << ";}\n";
            }
            // m_stream << "{bl _if_start_fn;}\n";
            generate_call(m_stream, "_if_start_fn");
            m_stream << "{cbz r0," << else_entry_label << ";}\n";
        }
        ro_index += DEFAULT_RO_NUMBER_PER_LAYER;
    }
    else
    {
        generate_prologue(m_stream);
    }

    auto index = ro_index;
    auto interface = AIPUBuilder::get_profiler()->interface();
    for (size_t i = 0; i < code->codes.size(); ++i)
    {
        auto c = code->codes[i];
        if (c->value != Constants::profiler_kernel && i < code->codes.size() - 1)
        {
            auto get_descriptor_size = [](CodeTreePtr code) {
                auto p = code->node.lock();
                std::vector<uint32_t> ret;
                if (p && p->attrs().count(Constants::real_desc_size))
                {
                    auto s = p->get_attr<uint32_t>(Constants::real_desc_size);
                    if (s > 100)
                    {
                        ret.push_back(s);
                        ret.push_back(1);
                        ret.push_back(1);
                    }
                }
                return ret;
            };
            auto nc = code->codes[i + 1];
            if (nc->value == Constants::profiler_kernel)
            {
                if (i < code->codes.size() - 2)
                {
                    nc = code->codes[i + 2];
                }
            }
            if (nc->value != Constants::profiler_kernel && !is_dummy_code(nc))
            {
                prefetch_cache(m_stream, get_code_value(nc), get_descriptor_size(nc));
            }
        }
        if (is_dummy_code(c))
        {
            if (is_branch_separator(code, c))
            {
                uint32_t offset = 0;
                for (size_t j = i + 1; j < code->codes.size(); ++j)
                {
                    offset += ro_size_map[code->codes[j]];
                }
                int16_t low = 0, high = 0;
                split_imm(4 * offset, low, high);
                m_stream << "{mfctrl0 r0,0x14;}\n"                    // get TCB pointer
                         << "{ld r1,r0,56; mov r2," << low << ";}\n"; // r1: pp_base
                if (low < 0 || high != 0)
                {
                    m_stream << "{movh r2," << high << ";}\n";
                }
                m_stream << "{add r1,r1,r2;}\n"
                         << "{st r1,r0,56;}\n" // update TCB PP
                         << "{cbz r31," << if_exit_label << ";}\n"
                         << else_entry_label << ":\n";
            }
            continue;
        }
        if (c->value == Constants::profiler_kernel)
        {
            m_stream << cast_any<std::string>(interface->get_function("reset_counter_text")(0));
        }
        else if (!internal_kernels.count(c->value))
        {
            DSCodePkg codes;
            if (code_map.count(index))
            {
                codes = code_map[index];
            }
            // if (!c->node.expired())
            // {
            //     c->node.lock()->set_attr(Constants::kernel_func, c->value);
            // }
            bool prefetch = index * 4u + 1024u < m_aipubin->ro.bin_size;
            prefetch = false;
            gen_loop_kernel_asm(m_stream, get_code_value(c), codes, prefetch);

            index += ro_size_map[c];
        }
        else
        {
            // m_stream << "{bl " << c->value << ";}\n";
            generate_call(m_stream, c->value);
        }
    }

    if (is_control_flow_code(code))
    {
        auto node = code->node.lock();
        if (node->type() == OpType_Loop)
        {
            m_stream << loop_exit_label << ":\n";
        }
        else if (node->type() == OpType_If)
        {
            m_stream << if_exit_label << ":\n";
        }
        DSCodePkg codes;
        if (code_map.count(index))
        {
            codes = code_map[index];
        }

        if (codes.ds[0].size() || codes.ds[1].size())
        {
            generate_tec_sync(m_stream);
            if (codes.ds[0].size())
            {
                // m_stream << "{bl " << codes.ds[0] << ";}\n"; // backup ro and desc
                generate_call(m_stream, codes.ds[0]);
            }
            if (codes.ds[1].size())
            {
                // m_stream << "{bl " << codes.ds[1] << ";}\n"; // modify ro and desc
                generate_call(m_stream, codes.ds[1]);
            }
            generate_tec_sync(m_stream);
        }

        uint32_t restore_pp = 0, check_flag = 0, ts_offset = 0;
        if (codes.ds[0].size())
        {
            restore_pp = 1;
            bool data_aware = utils::startswith(codes.ds[0], "_backup_param_da");
            if (!data_aware && BuilderConfig::exist(Constants::default_input_shape))
            {
                check_flag = 1;
            }
        }
        NodePtr tail = nullptr;
        if (node->type() == OpType_Loop)
        {
            tail = std::dynamic_pointer_cast<LoopNode>(node)->tail();
        }
        else if (node->type() == OpType_If)
        {
            tail = std::dynamic_pointer_cast<IfNode>(node)->tail();
        }
        if (codes.ds[2].size() || tail->attrs().count(Constants::output_shape_offset))
        {
            ts_offset = tail->get_attr<uint32_t>(Constants::output_shape_offset);
        }
        m_stream << "{add r0,r31," << restore_pp << ";"
                 << " add r1,r31," << check_flag << ";}\n";
        int16_t low = 0, high = 0;
        split_imm(ts_offset, low, high);
        m_stream << "{mov r2," << low << ";}\n";
        if (low < 0 || high != 0)
        {
            m_stream << "{movh r2," << high << ";}\n";
        }
        if (node->type() == OpType_Loop)
        {
            // m_stream << "{bl _forrange_end_fn;}\n";
            generate_call(m_stream, "_forrange_end_fn");
            m_stream << "{cbz r0," << loop_entry_label << ";}\n";
        }
        else if (node->type() == OpType_If)
        {
            // m_stream << "{bl _if_end_fn;}\n";
            generate_call(m_stream, "_if_end_fn");
        }
        m_stream << "{add ra,r0,0;}\n"
                 << "{br ra;}\n";
    }
    else
    {
        generate_epilogue(m_stream);
    }

    gen_func_ender_asm(m_stream, func_name);

    index = ro_index;
    for (auto c : code->codes)
    {
        gen_func_asm(m_stream, c, code_map, index);
        index += ro_size_map[c];
    }
}

static ModelGlobalParam build_global_param(GraphPtr graph)
{
    ModelGlobalParam param;
    param.input_shape_offset = GP_INPUT_SHAPE_OFFSET;
    param.num_params = GLOBAL_PARAM_COUNT;
    param.params.resize(GLOBAL_PARAM_COUNT);

    std::vector<std::vector<int>> shapes;
    for (auto inp : graph->get_input_tensors())
    {
        shapes.push_back(get_base_shape(inp->shape()));
    }
    param.update(shapes);

    if (BuilderConfig::get<bool>(Constants::enable_ds) &&
        BuilderConfig::exist(Constants::default_input_shape))
    {
        uint32_t index = 0;
        auto default_shape = parse_input_shape(graph, Constants::default_input_shape);
        for (auto &s : default_shape)
        {
            for (auto d : s)
            {
                param.params[GP_RESERVE0_COUNT + GP_INPUT_SHAPE_COUNT + index++] = d;
            }
        }
    }
    return param;
}

static bool infer_shape_backward(GraphPtr graph, std::map<std::string, SymbolShape> &tensors)
{
    bool ret = true;
    std::vector<NodePtr> nodes(graph->nodes().rbegin(), graph->nodes().rend());
    for (auto &&n : nodes)
    {
        if (SubGraphNode::is_subgraph_node(n))
        {
            ret &= infer_shape_backward(SubGraphNode::from_node(n)->subgraph(), tensors);
        }
        else
        {
            std::vector<SymbolShape> out_shapes;
            for (auto &&out : n->outputs())
            {
                if (tensors.count(out->name()))
                {
                    out_shapes.push_back(tensors[out->name()]);
                }
                else
                {
                    ret = false;
                }
            }
            if (ret)
            {
                auto in_shapes = tongue::infer_shape(n, out_shapes, true);
                if (in_shapes.size() == n->inputs().size())
                {
                    for (uint32_t i = 0; i < in_shapes.size(); ++i)
                    {
                        auto inp = n->inputs()[i];
                        if (!tensors.count(inp->name()))
                        {
                            tensors[inp->name()] = in_shapes[i];
                        }
                    }
                }
                else if (n->type() != OpType_Input && n->type() != OpType_SubGraphInput)
                {
                    ret = false;
                }
            }
        }
        if (!ret)
        {
            break;
        }
    }
    return ret;
}

static std::vector<AIPUTensorShape> compute_minimum_input_shape(GraphPtr graph)
{
    std::vector<AIPUTensorShape> min_shapes;
    std::map<std::string, SymbolShape> tensors;
    for (auto &&out : graph->get_output_tensors())
    {
        tensors[out->name()] = std::vector<Expr>(out->shape().dim(), 1);
    }
    auto in_tensors = graph->get_input_tensors();
    if (infer_shape_backward(graph, tensors))
    {
        for (size_t i = 0; i < in_tensors.size(); ++i)
        {
            std::vector<int> s;
            for (auto &&d : tensors[in_tensors[i]->name()].operator std::vector<Expr>())
            {
                s.push_back(static_cast<int>(d->value()));
            }
            auto shape = get_base_shape(TensorShape(s, in_tensors[i]->shape().layout()));
            min_shapes.push_back(shape.operator std::vector<uint32_t>());
        }
    }
    else
    {
        for (auto &&inp : in_tensors)
        {
            auto shape = get_base_shape(inp->shape());
            min_shapes.push_back(std::vector<uint32_t>(shape.dim(), 0));
        }
    }
    return min_shapes;
}

static InputShapeConstraint build_input_shape_constraint(GraphPtr graph)
{
    InputShapeConstraint constraint;
    auto enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);
    std::vector<AIPUTensorShape> min_shapes;
    if (enable_ds)
    {
        min_shapes = compute_minimum_input_shape(graph);
    }
    auto in_tensors = graph->get_input_tensors();
    for (uint32_t i = 0; i < in_tensors.size(); ++i)
    {
        auto shape = get_base_shape(in_tensors[i]->shape());
        if (enable_ds)
        {
            for (int j = 0; j < shape.dim(); ++j)
            {
                if (shape.symbol()[j]->is_constant())
                {
                    min_shapes[i].shapes[j] = shape[j];
                }
            }
        }
        else
        {
            min_shapes.push_back(AIPUTensorShape(shape));
        }
        constraint.constrains.push_back(min_shapes[i]);

        AIPUTensorShape max_shape(shape);
        constraint.constrains.push_back(max_shape);
    }
    DEBUG() << "graph input shape constraint:";
    std::ostringstream oss;
    oss << "min-shape: [";
    for (uint32_t i = 0; i < constraint.constrains.size(); ++i)
    {
        if (i % 2 == 0)
        {
            oss << constraint.constrains[i].to_string();
            if (i != constraint.constrains.size() - 2)
            {
                oss << ",";
            }
        }
    }
    oss << "]";
    DEBUG() << oss.str();
    oss.str("");
    oss << "max-shape: [";
    for (uint32_t i = 0; i < constraint.constrains.size(); ++i)
    {
        if (i % 2 != 0)
        {
            oss << constraint.constrains[i].to_string();
            if (i != constraint.constrains.size() - 1)
            {
                oss << ",";
            }
        }
    }
    oss << "]";
    DEBUG() << oss.str();
    return constraint;
}

static OutputTensorShape build_output_tensor_shape(GraphPtr g)
{
    OutputTensorShape ret;
    auto tensors = g->get_output_tensors();
    for (auto t : tensors)
    {
        AIPUTensorShape s;
        s.shapes = t->shape();
        ret.shapes.push_back(s);
    }
    return ret;
}

static ConstantHashTable build_constant_hash_table(const std::vector<std::vector<TensorPtr>> &tensors,
                                                   const std::vector<std::vector<uint64_t>> &offsets)
{
    ConstantHashTable table;
    for (size_t i = 0; i < tensors.size(); ++i)
    {
        ConstantHashItems items;
        for (size_t j = 0; j < tensors[i].size(); ++j)
        {
            ConstantHashItem item;
            auto t = tensors[i][j];
            item.bin_offset = offsets[i][j];
            item.bin_size = t->mem_size();
            item.hash_code = utils::sha256_bytes(t->data<void>(), t->mem_size());
            items.items.push_back(item);
        }
        table.items.push_back(items);
    }
    return table;
}

GraphELFBuilder::GraphELFBuilder(GraphPtr graph) : GraphBuilder(graph)
{
    m_elf = std::make_shared<AIPUELF>();
}

GraphELFBuilder::GraphELFBuilder(std::string file) : GraphBuilder(file)
{
    m_elf = std::make_shared<AIPUELF>();
    m_elf->load(file);
}

GraphELFBuilder::GraphELFBuilder(AIPUELFPtr elf) : GraphBuilder("")
{
    m_elf = elf;
}

void GraphELFBuilder::codes_to_kernel_object()
{
    m_objects.clear();
    std::set<std::string> funcs;
    for (auto &sg : m_codes)
    {
        for (auto &k : sg)
        {
            if (k == Constants::profiler_code)
            {
                continue;
            }
            if (utils::startswith(k, "."))
            {
                auto f = utils::split_string(k, " ");
                if (f.size() == 2)
                {
                    funcs.insert(f[1]);
                }
                continue;
            }
            if (funcs.count(k))
            {
                continue;
            }
            auto obj = get_lib_full_filename(k);
            if (obj.size())
            {
                // if(utils::endswith(obj,".a"))
                // {
                //     m_objects.push_back("-l"+obj);
                // }
                // else
                {
                    m_objects.push_back(obj);
                }
            }
        }
    }
}

uint32_t GraphELFBuilder::get_func_size(const std::string &func_name) const
{
    if (m_func_size.count(func_name))
    {
        return m_func_size.at(func_name);
    }
    m_func_size[func_name] = 0;
    std::set<std::string> unique_objects(m_objects.begin(), m_objects.end());
    for (auto &obj : unique_objects)
    {
        if (ELFReader::is_elf_file(obj))
        {
            AIPUELF elf;
            elf.load(obj);
            auto s = elf.get_func_size(func_name);
            if (s)
            {
                m_func_size[func_name] = s;
                return s;
            }
        }
        else
        {
            auto ar = ARFileReader(obj);
            if (!ar.is_ar_file())
            {
                WARN() << "Unsupport link with file " << obj << ", only support elf file or .a file.";
                continue;
            }
            auto c = ar.get_file_content_by_symbol(func_name);
            if (c.second)
            {
                AIPUELF elf;
                elf.load(c.first, c.second);
                auto s = elf.get_func_size(func_name);
                if (s)
                {
                    delete[] c.first;
                    c.first = 0;
                    m_func_size[func_name] = s;
                    return s;
                }
            }
        }
    }
    return 0;
}

/**
 * @brief build the text
 *  for elf file, the text_bin is the whole elf. temp for Z5
 *
 * @param text_bin
 * @param header
 * @param codes
 */
void GraphELFBuilder::build_text()
{
    std::stringstream oss2;
    std::stringstream oss3;
    codes_to_kernel_object();
    generate_asm(oss2);
    auto masm = oss2.str();
    // INFO() << masm;
    m_linker.set_code(masm);
    m_linker.set_objects(m_objects);
    m_linker.link(oss3);
    m_mainobj = oss3.str();
    if (m_aipubin->text.bin)
    {
        delete m_aipubin->text.bin;
        m_aipubin->text.bin = nullptr;
        m_aipubin->text.bin_size = 0;
    }
}

GMConfig GraphELFBuilder::build_gmconfig()
{
    auto regions = m_gm_regions;
    size_t gms = 0;
    for (auto s : m_gm_size)
    {
        gms += s;
    }
    GMConfig gm;

    if (regions.size())
    {
        gm.set_remap_enable(regions.size());
        gm.set_gm_size(gms);
        uint32_t region_size = ::pow(2, (int32_t)((gm.GM_control >> 16) & 0xf) - 1) * 1024 * 1024 / regions.size();
        for (uint32_t i = 0; i < regions.size(); ++i)
        {
            gm.set_region_buffer_idx(i, regions[i]->buffer_id);
            if ((uint32_t)regions[i]->buffer_id < m_aipubin->bss.buffer_descriptor.size())
            {
                auto &sec = m_aipubin->bss.buffer_descriptor[regions[i]->buffer_id];
                bool has_input = false;
                uint32_t start = UINT32_MAX;
                uint32_t end = 0;
                for (auto sub : sec.sub_sections)
                {
                    if (sub.type == 0)
                    {
                        has_input = true;
                        start = MIN(start, sub.relative_offset);
                        end = MAX(end, sub.relative_offset + sub.size);
                    }
                }
                if ((end > start) && (start < region_size))
                {
                    GMConfig::GMSyncDir dir = GMConfig::NoSync;
                    if (end > region_size)
                    {
                        INFO() << "sync size larger than region size";
                        end = region_size;
                    }
                    uint32_t size = end - start;
                    if (has_input)
                    {
                        dir = GMConfig::DDR2GM;
                    }
                    gm.set_region_sync(i, start, size, dir);
                }
            }
        }
    }
    return gm;
}

BinSection GraphELFBuilder::build_segmmu()
{
    BinSection sec;

    SegMMUList smmu_list;
    std::vector<SegMMUConfig> smmu_vec;
    size_t num_mmu = 0;
    if (aipubt::BuilderConfig::get<bool>("enable_custom_smmu"))
    {
        num_mmu = 1;
        SegMMUConfig smmu;
        smmu.seg0.control0 = (0x1800 << 14) | (0 & 0x7f); //remap addr=0x6000_0000, size = no limit; this is for test only
        smmu.SegMMU_remap = 0;
        smmu.SegMMU_ctl = 0x1 << 5 | 0x1 << 4 | 0x1; //share_en = remap_en = ctrl_en = 1
        smmu_vec.push_back(smmu);
        BssSubSectionDescriptor sub;
        sub.type = 255;      // segmmu_buffer
        sub.id = 0xffff0000; // segment 0 control 0 is available for all cores;
        sub.addr_mask = 0xffffffff;
        m_aipubin->bss.buffer_descriptor[1].add_subsection(sub); //slot[1] is used to remap smmu for the test case
    }

    smmu_list.num_mmu = num_mmu;
    smmu_list.smmu = smmu_vec.data();

    sec.bin = new char[sizeof(uint32_t) + sizeof(SegMMUConfig) * smmu_vec.size()];
    sec.bin_size = sizeof(uint32_t) + sizeof(SegMMUConfig) * smmu_vec.size();

    memcpy(sec.bin, &smmu_list.num_mmu, sizeof(uint32_t));
    memcpy(sec.bin + sizeof(uint32_t), smmu_list.smmu, sizeof(SegMMUConfig) * smmu_vec.size());

    return sec;
}

bool GraphELFBuilder::build(const std::string &filename,
                            const std::vector<TensorPtr> &inputs_,
                            const std::vector<TensorPtr> &outputs,
                            const std::vector<int32_t> &dump_ids)
{
    auto ret = build_aipu(inputs_, outputs, dump_ids);
    m_elf->write(filename);
    return ret;
}
void GraphELFBuilder::write(std::ostream &ofs)
{
    m_elf->write(ofs);
}
void GraphELFBuilder::write(const std::string &filename)
{
    m_elf->write(filename);
}

bool GraphELFBuilder::build_aipu(const std::vector<TensorPtr> &inputs_,
                                 const std::vector<TensorPtr> &outputs_,
                                 const std::vector<int32_t> &dump_ids)
{
    auto ret = GraphBuilder::build_aipu(inputs_, outputs_, dump_ids);
    aipubin_to_elf();
    show_memory_info();
    return ret;
}

void GraphELFBuilder::partition_graph()
{
    // TODO implement parallel algorithm to partition graph here
    GraphBuilder::partition_graph();
}
void GraphELFBuilder::build_debug_info()
{
    if (!BuilderConfig::get<bool>(Constants::enable_debug_info))
        return;
    std::ostringstream txt, dumpir, wb;
    m_origin_graph->serialize(txt, wb);
    m_elf->set_ir(txt.str());
    txt.clear();
    wb.clear();
    auto loc = BuilderConfig::get<bool>(Constants::show_location);
    BuilderConfig::set(Constants::show_location, true);
    m_graph->serialize(dumpir, wb);
    wb.clear();
    m_elf->set_dump_ir(dumpir.str());
    BuilderConfig::set(Constants::show_location, loc);

    std::map<MemoryObjectPtr, uint32_t> buffers;
    for (uint32_t i = 0; i < m_ro.size(); ++i)
    {
        auto &ro = m_ro[i];
        if (ro.tensor)
        {
            buffers[ro.tensor->memory_offset()] = i * 4;
        }
    }
    for (uint32_t i = 0; i < m_descriptor.size(); ++i)
    {
        auto &ro = m_descriptor[i];
        if (ro.tensor)
        {
            buffers[ro.tensor->memory_offset()] = (i + m_ro.size()) * 4;
        }
    }
    std::vector<LayerDebugInfo> infos;
    std::map<NodePtr, int> subgid;
    for (auto n : m_graph->nodes())
    {
        if (n->attrs().count(Constants::subgraphid))
        {
            auto id = n->get_attr<uint32_t>(Constants::subgraphid);
            subgid[n] = id;
            if (SubGraphNode::is_subgraph_node(n))
            {
                auto sg = SubGraphNode::from_node(n);
                sg->subgraph()->walk_on_node([&](NodePtr sn) {
                    subgid[sn] = id;
                });
            }
        }
    }

    for (auto level : m_levels)
    {
        for (uint32_t i = 0; i < level.size(); ++i)
        {
            auto node = level[i];
            auto sg_node = SubGraphNode::from_node(node);

            if (sg_node)
            {
                for (auto n : sg_node->subgraph()->nodes())
                {
                    if (((SubGraphNode::is_subgraph_node(n) &&
                          n->attrs().count(Constants::plugin_attrs_name)) ||
                         (n->type() == OpType_LayoutConvertor &&
                          !is_a_remapping_convertor(n))) &&
                        n->attrs().count(Constants::kernel_func))
                    {
                        LayerDebugInfo info;
                        auto sgn = SubGraphNode::from_node(n);
                        if (sgn)
                        {
                            auto sg = sgn->subgraph();
                            info.layer_id = sg->nodes()[0]->get_attr_deep<uint32_t>("layer_id");
                            for (auto &node : sg->nodes())
                            {
                                info.ids.push_back(node->get_attr_deep<uint32_t>("layer_id"));
                            }
                        }
                        else
                        {
                            info.layer_id = n->get_attr_deep<uint32_t>("layer_id");
                        }

                        if (subgid.count(n))
                        {
                            info.subgraph_id = subgid[n];
                        }
                        auto ro_base = 0; //tcb_ro_base_map[n];
                        // if (ro_base <= 0)
                        // {
                        //     ro_base = 0; //TODO fixme
                        // }
                        for (auto inp : n->inputs())
                        {
                            if (buffers.count(inp->memory_offset()))
                            {
                                info.inputs.push_back(buffers.at(inp->memory_offset()) - ro_base);

                                continue;
                            }
                            bool not_found = true;
                            for (auto kv : buffers)
                            {
                                if (kv.first->offset() == inp->memory_offset()->offset())
                                {
                                    info.inputs.push_back(kv.second - ro_base);
                                    not_found = false;
                                    break;
                                }
                            }
                            if (not_found)
                            {
                                WARN() << "Cannot find buffer in ro, use 0 for debug info";
                                info.inputs.push_back(0);
                            }
                        }
                        for (auto out : n->outputs())
                        {
                            if (buffers.count(out->memory_offset()))
                            {
                                info.outputs.push_back(buffers.at(out->memory_offset()) - ro_base);
                                continue;
                            }
                            bool not_found = true;
                            for (auto kv : buffers)
                            {
                                if (kv.first->is_shared_by(out->memory_offset()) && kv.first->offset() == out->memory_offset()->offset())
                                {
                                    info.outputs.push_back(buffers.at(out->memory_offset()) - ro_base);
                                    not_found = false;
                                    break;
                                }
                            }
                            if (not_found)
                            {
                                WARN() << "Cannot find buffer in ro, use 0 for debug info";
                                info.outputs.push_back(0);
                            }
                        }
                        // auto kernel = utils::split_string(n->get_attr<std::string>(Constants::kernel_func), '\n')[0];
                        auto kernels = utils::split_string(n->get_attr<std::string>(Constants::kernel_func), '\n');
                        for (auto kernel : kernels)
                        {
                            if (kernel == "" || kernel == "profiler_kernel")
                            {
                                continue;
                            }
                            info.address = m_elf->get_function_offset(kernel);
                            if (info.address == 0)
                            {
                                WARN() << "Can not get function of node:" << n->name() << " kernel  is :" << kernel;
                            }
                            // INFO() << "kernel: " << kernel << " addr " << info.address;
                            infos.push_back(info);
                        }
                    }
                }
            }
        }
    };
    m_elf->set_layer_debug_info({infos});
} // namespace aipubt

void GraphELFBuilder::show_memory_info()
{
    uint64_t total = 0;
    INFO() << "memory statistics for this graph"
           << " (" << m_graph->name() << ")";
    show_section_size("Text            section:\t", get_text_size());
    total += get_text_size();
    show_section_size("Weight          section:\t", m_elf->get_weight().bin_size);
    total += m_elf->get_weight().bin_size;
    auto extra_weight = m_elf->get_extra_weight();
    for (auto wt : extra_weight)
    {
        show_section_size("ExtraWeight     section:\t", wt.bin_size);
        total += wt.bin_size;
    }
    show_section_size("Ro              section:\t", m_elf->get_rodata().bin_size);
    total += m_elf->get_rodata().bin_size;
    show_section_size("Desc            section:\t", m_elf->get_descriptor().bin_size);
    total += m_elf->get_descriptor().bin_size;

    auto bss_vec = m_elf->get_fm_list().fmdesc;
    show_section_size("Stack           section:\t", bss_vec[0].stack_size * Target::get_the_target()->core_num());
    total += bss_vec[0].stack_size * Target::get_the_target()->core_num();

    auto sgs = m_elf->get_subgraphs().sgs;
    uint32_t ws_offset = 0;
    uint32_t max_ws_size = 0;
    for (auto sg : sgs)
    {
        if (sg.precursor_cnt == -1)
        {
            ws_offset = 0;
        }
        if (sg.private_buffer_cnt == 0)
        {
            continue;
        }

        auto sg_buf = sg.private_buffer[0]; //for now the only private buffer is workspace
        ws_offset = aligned(ws_offset, sg_buf.align_bytes) + sg_buf.size;
        max_ws_size = std::max(max_ws_size, ws_offset);
    }
    show_section_size("Workspace       section:\t", max_ws_size);
    total += max_ws_size;

    uint64_t next_addr = 0;
    for (auto bss : bss_vec)
    {
        for (uint32_t i = 0; i < bss.num_buffer_descriptor; ++i)
        {
            auto address = aligned(next_addr, bss.buffer_descriptor[i].align_bytes);
            next_addr = address + bss.buffer_descriptor[i].size;
        }
    }
    total += next_addr;
    show_section_size("Activations     section:\t", next_addr);

    show_section_size("Total:\t", total);
}

void GraphELFBuilder::aipubin_to_elf()
{
    m_elf = std::make_shared<AIPUELF>();
    m_elf->compiler_msg().arch = m_aipubin->header.arch;
    m_elf->compiler_msg().flag = m_aipubin->header.flag;
    uint32_t disable_out = aipubt::BuilderConfig::get<bool>(Constants::disable_output_buffer_reuse) ? 1 : 0;
    uint32_t disable_in = aipubt::BuilderConfig::get<bool>(Constants::disable_input_buffer_reuse) ? 1 : 0;
    uint32_t option_msg = ((disable_out & 0x1) << 1) | (disable_in & 0x1);
    m_elf->compiler_msg().reserve0[0] = option_msg;
    m_elf->set_elf_file(m_mainobj);
    m_elf->set_comment(aipubt::BuilderConfig::get<std::string>(Constants::arg_map));
    m_elf->set_rodata(m_aipubin->ro);
    m_elf->set_descriptor(m_aipubin->desc);
    m_elf->set_weight(m_aipubin->data);
    m_elf->set_extra_weight(m_aipubin->get_extra_data());

    auto text_size = m_elf->get_text_size();
    auto segmmu = build_segmmu();
    m_elf->set_aipu_note(segmmu, "segmmu");
    FeatureMapList fm;
    for (auto bss : m_bss_vec)
    {
        fm.add_bss(bss);
    }
    m_elf->set_fm_list(fm);
    m_elf->set_remap_table(m_aipubin->remap);

    auto param = build_global_param(m_graph);
    m_elf->set_globalparam(param);

    auto constraint = build_input_shape_constraint(m_graph);
    m_elf->set_input_shape_constraint(constraint);

    auto shapes = build_output_tensor_shape(m_graph);
    m_elf->set_output_tensor_shape(shapes);

    auto table = build_constant_hash_table(m_const_tensors, m_const_offsets);
    m_elf->set_constant_hash_table(table);

    SGList sgs;

    uint32_t num_sg = 0;
    // auto nodes = temp_test_confirm_dependencies();
    auto enable_profile = AIPUBuilder::get_profiler()->enable_profile();
    auto interface = AIPUBuilder::get_profiler()->interface();
    bool level_start = false;
    bool grid_start = true;

    // deprecated, instead: layer_id <-> perf_node_id
    std::vector<std::string> subgraph_order;
    std::vector<uint32_t> subgraph_levels;

    uint32_t sg_lv = 0;
    for (auto level : m_levels)
    {
        level_start = true;
        for (uint32_t i = 0; i < level.size(); ++i)
        {
            if (m_task_map.count(level[i]) == 0 || m_task_map[level[i]].empty())
            {
                continue;
            }
            SubGraphDesc sg = {0};
            sg.id = num_sg;
            level[i]->set_attr(Constants::subgraphid, num_sg);
            sg.type = 0;
            sg.elf_text_offset = m_elf->get_function_offset(m_task_map[level[i]]);
            //TODO ro and desc  offset
            sg.fm_desc_offset = level[i]->get_attr<size_t>(Constants::bss_id);
            sg.rodata_offset = level[i]->get_attr<size_t>(Constants::ro_offset);
            sg.rodata_size = level[i]->get_attr<size_t>(Constants::ro_size);
            sg.descriptor_offset = level[i]->get_attr<size_t>(Constants::desc_offset);
            sg.descriptor_size = level[i]->get_attr<size_t>(Constants::desc_size);
            sg.printfifo_size = 1024;
            sg.profiler_buf_size = 0;
            sg.private_data_size = 1024;
            sg.warmup_len = 9; //9*256bytes->2304
            if (text_size > 1536 && text_size < 65536)
            {
                sg.warmup_len = std::min(((text_size + 255) >> 8) + 1, (uint32_t)255);
            }
            if (BuilderConfig::exist(Constants::warmup_len))
            {
                sg.warmup_len = BuilderConfig::get<int>(Constants::warmup_len);
            }
            if (enable_profile)
            {
                auto fn = interface->get_function("subgraph_record_size");
                uint32_t subgraph_record_size = cast_any<uint32_t>(fn(0));
                sg.profiler_buf_size = subgraph_record_size * num_sg;
                subgraph_order.push_back(level[i]->name());
                subgraph_levels.push_back(sg_lv);
            }
            sg.precursor_cnt = 0;
            if (grid_start)
            {
                sg.precursor_cnt = 0;
                grid_start = false;
                level_start = false;
            }
            else if (level_start)
            {
                sg.precursor_cnt = -1;
                level_start = false;
            }
            else if (Target::get_the_target()->isainfo().type >= ISA_ZYv6)
            {
                std::vector<NodePtr> dped = level[i]->get_attr("dped_nodes");
                for (auto n : dped)
                {
                    auto p_id = n->get_attr<uint32_t>(Constants::subgraphid);
                    sg.add_precursor(p_id);
                    // INFO() << "sg " << sg.id << " depends on " << id;
                }
                sg.precursor_cnt = sg.precursors.size();
                const int max_precursor_cnt = 4;
                if (sg.precursor_cnt > max_precursor_cnt)
                {
                    WARN() << "maximum " << max_precursor_cnt;
                }
            }
            // TODO currently, the core-private buffer is only workspace
            // we need a general way for core-private buffer
            if (level[i]->attrs().count(Constants::private_buffers))
            {
                std::vector<MemoryObjectPtr> private_buffers = level[i]->get_attr(Constants::private_buffers);
                for (auto buf : private_buffers)
                {
                    size_t buf_size = buf->size;
                    BssBufferDescriptor desc;
                    desc.align_bytes = buf->align;
                    for (unsigned int j = 0; j < m_buffer_map[buf->get_base_object()].size(); ++j)
                    {
                        BssSubSectionDescriptor subsec;
                        auto tensor = m_buffer_map[buf->get_base_object()][j];
                        subsec.relative_offset = tensor->memory_offset()->offset() - tensor->memory_offset()->get_base_object()->offset();
                        subsec.size = tensor->mem_size();
                        buf_size = MAX(buf_size, subsec.size);
                        buf_size = Target::get_the_target()->isainfo().type == ISA_ZYv6 ? aligned(buf_size, 64) : buf_size;
                        subsec.type = 3;
                        subsec.id = 0;
                        subsec.addr_mask = ~tensor->memory_offset()->mask;

                        subsec.dtype = tensor->type();
                        subsec.scale = tensor->quantization().scale();
                        subsec.zp = tensor->quantization().offset();
                        for (unsigned int k = 0; k < m_tensor_map[tensor].size(); ++k)
                        {
                            subsec.add_ro_offset(m_tensor_map[tensor][k] * 4);
                        }
                        desc.add_subsection(subsec);
                    }
                    desc.size = buf_size;
                    if (buf_size > 0)
                    {
                        if (desc.sub_sections.size() == 0)
                            DEBUG() << level[i]->name() << " has no subsections for private buffer";
                        sg.private_buffer.push_back(desc);
                    }
                }
            }
            sg.private_buffer_cnt = sg.private_buffer.size();

            sgs.add_sg(sg);
            num_sg++;
        }
        sg_lv++;
    }

    if (!subgraph_order.empty())
    {
        auto fn = interface->get_function("subgraph_order");
        fn(subgraph_order);
    }
    if (!subgraph_levels.empty())
    {
        auto fn = interface->get_function("subgraph_levels");
        fn(subgraph_levels);
    }
    //TODO support multi-subgraphs
    sgs.num_subgraph = num_sg;
    m_elf->set_subgraphs(sgs);
    m_elf->set_GMconfig(build_gmconfig());
    build_debug_info();

    // add profiler info
    if (AIPUBuilder::get_profiler()->enable_profile())
    {
        // for field: graph.json.tensors.memory = 'gm' or not
        if (!m_gm_regions.empty())
            AIPUBuilder::get_profiler()->set_mem_info(m_gm_regions[0]);
        m_elf->set_graph_json(AIPUBuilder::get_profiler()->json_from_graph(m_graph));
    }
}

GraphELFBuilder::~GraphELFBuilder()
{
}

void GraphELFBuilder::gen_init_ctx(std::stringstream &m_stream)
{
    /**
     * The stack layout for global var:
     * 
     * global ------------
     * var    ------------
     * region ------------   
     * -------------------   the verification value == TCB's stack pointer
     * -------------------   <- TCB's stack pointer
     * stack  ------------
     * region ------------
     *        ------------
     *        ------------   <- R29, the stack pointer for kernel function
     *        ------------
     * 
     * =============================
     * 
     * 
     * ======================
     * |       128 words    |
     * |--------------------|
     * |                    |
     * |--------------------|
     * |                    |
     * |--------------------|
     * |                    |
     * |--------------------|
     * |                    |
     * |--------------------|
     * |                    |
     * |--------------------|
     * |      reserve       |
     * |--------------------|
     * |  Debug: gid,opid   |
     * |--------------------|
     * |    R30 backup      |
     * ======================
     */
    uint32_t stack_size = BuilderConfig::get<uint32_t>("stack_size");
    uint64_t label_num = utils::get_increase_id64();
    std::string entry_label = "._lbl_" + std::to_string(label_num);
    std::string intr_handler_label = "._INTR_HANDLER_LABEL_" + std::to_string(label_num);
    std::string intr_fp_handler_label = "._INTR_FP_HANDLER_LABEL_" + std::to_string(label_num);
    // the main function will use 512 bytes stack
    stack_size -= 512;
    m_stream << "{b " << entry_label << ";}\n"           // SPC
             << "{b " << intr_handler_label << ";}\n"    // SPC + 16*1 Fetch Exception
             << "{b " << intr_handler_label << ";}\n"    // SPC + 16*2 reserved
             << "{b " << intr_fp_handler_label << ";}\n" // SPC + 16*3 FP Exception
             << "{exit;}\n"                              // SPC + 16*4 reserved
             << "{exit;}\n"                              // SPC + 16*5 reserved
             << "{exit;}\n"                              // SPC + 16*6 reserved
             << "{exit;}\n"                              // SPC + 16*7 reserved
             << "{exit;}\n"                              // SPC + 16*8 reserved
             << "{exit;}\n"                              // SPC + 16*9 reserved
             << "{exit;}\n"                              // SPC + 16*10 reserved
             << "{exit;}\n"                              // SPC + 16*11 reserved
             << "{exit;}\n"                              // SPC + 16*12 reserved
             << "{exit;}\n"                              // SPC + 16*13 reserved
             << "{exit;}\n"                              // SPC + 16*14 reserved
             << "{exit;}\n"                              // SPC + 16*15 reserved
             << "{b " << intr_handler_label << ";}\n"    // SPC + 16*16 CP0 fault
             << "{b " << intr_handler_label << ";}\n"    // SPC + 16*17 CP1 fault
             << "{b " << intr_handler_label << ";}\n"    // SPC + 16*18 CP2 fault
             << "{exit;}\n"                              // SPC + 16*19 reserved
             << "{exit;}\n"                              // SPC + 16*20 reserved
             << "{exit;}\n"                              // SPC + 16*21 reserved
             << "{exit;}\n"                              // SPC + 16*22 reserved
             << "{exit;}\n"                              // SPC + 16*23 reserved
             << "{exit;}\n"                              // SPC + 16*24 reserved
             << "{exit;}\n"                              // SPC + 16*25 reserved
             << "{exit;}\n"                              // SPC + 16*26 reserved
             << "{exit;}\n"                              // SPC + 16*27 reserved
             << "{exit;}\n"                              // SPC + 16*28 reserved
             << "{exit;}\n"                              // SPC + 16*29 reserved
             << "{exit;}\n"                              // SPC + 16*30 reserved
             << "{exit;}\n"                              // SPC + 16*31 reserved
             << intr_handler_label << ":\n"
             << "{b " << INTERRUPT_HANDLER << ";}\n"
             << "{irtn;}\n"

             << intr_fp_handler_label << ":\n"
             << "{b " << INTERRUPT_FP_HANDLER << ";}\n"
             << "{irtn;}\n"

             << entry_label << ":\n";
    // _start entry
    if (BuilderConfig::get<bool>("disable_aipu_cache"))
    {
        m_stream << "{wfe r0, 0;}\n";
        m_stream << "{mtctrl0 r31,0x82;}\n";
        m_stream << "{mtctrl0 r31,0x81;}\n";
        m_stream << "{wfe r0, 0;}\n";
    }
    else if (BuilderConfig::get<bool>(Constants::disable_icache))
    {
        m_stream << "{wfe r0, 0;}\n";
        m_stream << "{mtctrl0 r31,0x81;}\n";
        m_stream << "{wfe r0, 0;}\n";
    }

    // intentionally cause instruction decode error
    // m_stream << "{mov r0,0;}"
    //          << "{movh r0,1536;}"
    //          << "{br r0;}\n";

    m_stream << "{mfctrl0 r0,0x14;}\n"                           // get TCB pointer
             << "{ld r29,r0,52; ld r23,r0,64; add r24,r0,0;}\n"; // r29: sp_base, r23: cp_base
    uint16_t l = stack_size & 0xffff;
    int16_t stack_low = *(int16_t *)(&l);
    l = (stack_size >> 16) & 0xffff;
    int16_t stack_hi = *(int16_t *)(&l);
    m_stream << "{mov r1, " << stack_low << ";}\n";
    if (stack_hi)
    {
        m_stream << "{movh r1, " << stack_hi << ";}\n";
    }
    m_stream
        << "{add r29, r29, r1;}\n" //initial R29
        << "{mov  r2,0;}\n"
        << "{mov  r3,0;}\n"
        << "{mov  r4,0;}\n"
        << "{mov  r5,0;}\n"
        << "{mov  r6,0;}\n"
        << "{mov  r7,0;}\n"
        << "{mov  r8,0;}\n"
        << "{mov  r9,0;}\n"
        << "{mov r10,0;}\n"
        << "{mov r11,0;}\n"
        << "{mov r12,0;}\n"
        << "{mov r13,0;}\n"
        << "{mov r14,0;}\n"
        << "{mov r15,0;}\n"
        << "{mov r16,0;}\n"
        << "{mov r17,0;}\n"
        << "{mov r18,0;}\n"
        << "{mov r19,0;}\n"
        << "{mov r20,0;}\n"
        << "{mov r21,0;}\n"
        << "{mov r22,0;}\n"
        << "{mov r25,0;}\n"
        << "{mov r26,0;}\n"
        << "{mov r27,0;}\n"
        << "{mov r28,0;}\n"
        << "{mov r30,0;};\n";
    //for temp
    m_stream << "{st r29,r0,52;}\n"; //reset the r29 to tcb
                                     //  << "{st r29,r29,0;}\n"; //store r29 to the head of tcb for verification

    // get tec/core/cluster physical id
    // tec id[19:16], coreid[23:20], cluster [31:24]
    m_stream << "{mfctrl0 r1,0;}\n"   //r1: clusterid coreid tecid
             << "{lsr r1,r1,16;}\n"   //r1: physical clusterid/coreid/tecid
             << "{andl r2,r1,0xf;}\n" //r2: tecid
             << "{lsl r2,r2,16;}\n"
             << "{st r2,r0,80; lsr r1,r1,4;}\n" //r1: clusterid coreid
             << "{andl r2,r1,0xf;}\n"           //r2: coreid
             << "{lsr r1,r1,4;}\n"
             << "{andl r1,r1,0xf;}\n" //r1: clusterid
             << "{lsl r1,r1,16;}\n"
             << "{or r2,r1,r2;}\n"
             << "{st r2,r0,76;}\n";
    // prefetch PP, one layer's PP is 256Bytes
    m_stream << "{ld r3,r0,56;}\n"
             << "{cacheop r3,8,0;}\n"; // prefetch 1K=4 layers

    // prefetch global param
    if (BuilderConfig::get<bool>(Constants::enable_ds))
    {
        m_stream << "{ld r2,r0,112;}\n"
                 << "{cacheop r2,8,1;}\n"; // prefetch 1KB
    }
}

void GraphELFBuilder::generate_asm(std::stringstream &m_stream)
{
    m_codetree.legalize_func_name();
    // generate subgraph -> task
    // assume the top level of code tree is task level
    uint32_t gid = 0;
    uint32_t first_exec_level = UINT32_MAX;
    uint32_t last_exec_level = 0;
    auto profiler = AIPUBuilder::get_profiler();
    bool enable_profile = profiler->enable_profile();
    bool enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);
    if (enable_ds)
    {
        for (auto c : m_codetree.codes)
        {
            if (c->codes.size() && c->node.lock())
            {
                uint32_t level = c->node.lock()->get_attr_deep<int>(Constants::schedule_level);
                last_exec_level = std::max(level, last_exec_level);
            }
        }
    }
    for (auto c : m_codetree.codes)
    {
        if (c->codes.size() == 0)
        {
            gid++;
            continue;
        }
        std::string func_name = ".aipubt_main" + c->value;
        m_task_map[c->node.lock()] = func_name;
        m_stream << ".p2align 6\n";
        m_stream << ".type " << func_name << ", function\n";
        m_stream << ".globl " << func_name << "\n";
        m_stream << func_name << ":\n";
        gen_init_ctx(m_stream);

        prefetch_cache(m_stream, ".aipubt_func" + c->value);
        if (c->codes.size())
        {
            for (auto nc : c->codes)
            {
                if (!is_dummy_code(nc))
                {
                    prefetch_cache(m_stream, get_code_value(nc));
                    break;
                }
            }
        }
        set_timeout_reg(m_stream);

        // if profiler is enabled, init profiler. otherwise, disable DMA/AIFF perf counter.
        m_stream << cast_any<std::string>(profiler->interface()->get_function("init_profiler_text")(0));

        uint32_t op_id = 0;
        uint32_t ggid = gid;
        uint32_t level = 0;
        if (c->node.lock())
        {
            ggid = c->node.lock()->get_attr_deep<int>(Constants::schedule_gid);
            if (ggid == 0)
            {
                ggid = gid;
            }
            level = c->node.lock()->get_attr_deep<int>(Constants::schedule_level);
            if (first_exec_level == UINT32_MAX)
            {
                first_exec_level = level;
            }
            auto ids = c->node.lock()->get_attr_deep<std::vector<uint32_t>>(Constants::operator_id);
            if (ids.size())
            {
                op_id = ids[0];
                // INFO() << "gid=" << ggid << ", opid=" << op_id;
            }
        }
        generate_hw_debug_info(ggid, m_stream, op_id - 1);
        add_delay(m_stream, 0);
        if (enable_ds && level == first_exec_level)
        {
            // m_stream << "{bl " << DSCodeGenerator::set_global_param_fn << ";}\n";
            generate_call(m_stream, DSCodeGenerator::set_global_param_fn);
        }
        // m_stream << "{bl .aipubt_func" << c->value << ";}\n";
        generate_call(m_stream, ".aipubt_func" + c->value);
        if (level == last_exec_level)
        {
            if (enable_ds)
            {
                // m_stream << "{bl " << DSCodeGenerator::record_output_shape_fn << ";}\n";
                generate_call(m_stream, DSCodeGenerator::record_output_shape_fn);
            }
            m_stream << cast_any<std::string>(profiler->interface()->get_function("exit_profiler_text")(0));
        }
        if (enable_profile)
            m_stream << cast_any<std::string>(profiler->interface()->get_function("clean_profiler_text")(0));

        gen_flush_cache_code(m_stream);
        gen_clear_fault_code(m_stream);
        gen_check_cache_flush_done_code(m_stream);
        generate_wait_IRQ_empty(m_stream);

        if (BuilderConfig::get<bool>("disable_aipu_cache"))
        {
            m_stream << "{wfe r0, 0;}\n";
            m_stream << "{mov r1, 0x1}\n";
            m_stream << "{mtctrl0 r1,0x82;}\n"; //TEC LSU Control
            m_stream << "{mov r1, 0x1}\n";
            m_stream << "{mtctrl0 r1,0x81;}\n"; //TEC IFU Control
        }
        else if (BuilderConfig::get<bool>(Constants::disable_icache))
        {
            m_stream << "{wfe r0, 0;}\n";
            m_stream << "{mov r1, 0x1}\n";
            m_stream << "{mtctrl0 r1,0x81;}\n"; //TEC IFU Control
        }

        m_stream << "{wfe r0, 0;}\n"
                 << "{exit;}\n";
        gen_func_ender_asm(m_stream, func_name);
        gid++;
    }
    // generate
    std::map<int, DSCodePkg> code_map;
    DSCodeGenerator codegen(m_graph);
    codegen.generate(m_stream, m_ro, m_descriptor, code_map);
    for (auto c : m_codetree.codes)
    {
        int ro_index = c->node.lock()->get_attr<size_t>(Constants::ro_offset) / 4;
        gen_func_asm(m_stream, c, code_map, ro_index);
    }
    generate_interrupt_handler(m_stream);
}
uint32_t GraphELFBuilder::get_text_size()
{
    if (m_elf)
    {
        return m_elf->get_text_size();
    }
    return 0;
}

} // namespace aipubt
