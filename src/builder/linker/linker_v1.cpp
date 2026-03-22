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
// This entire notice must be reproduced on all copies of this file and copies
// of this file may only be made by a person if such person is permitted to do
// so under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Neo Wang
 * @Date: 2020-05-27 10:45:52
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-04-13 16:43:34
 */
#include "linker_v1.h"
#include "../text_generator.h"
#include "ZY.h"
#include "builder/aipu_binfile.h"
#include "builder/profiler.h"
#include "builder/target.h"
#include "core/config.h"
#include "utils/utils.h"
namespace aipubt
{
static uint32_t CACHE_CONTROL_REGISTER = 0x50;
using namespace AsmZY;

void aipu_memcpy_async(std::ostream &oss, uint32_t src, uint32_t dst, uint32_t size, uint32_t asid = 0, uint32_t chn = 7, bool save_ctx = true)
{
    // CHn Control Register:
    // - [11:10] ASID
    // - [7:6] DATA_UNIT: 0 (byte)
    // - [5:4] OP_MODE: 0 (direct data movement)
    // - [1] Direction:
    //       1 (external -> internal)
    //       0 (internal -> external)
    // - [0] START: 1 (enable DMA when in register control mode)
    uint32_t ext_addr = src;
    uint32_t int_addr = dst;
    uint32_t ctl = 0b11;
    if ((src >> 28) == 0xf)
    {
        aipu_assert((dst >> 28) != 0xf);
        int_addr = src;
        ext_addr = dst;
        ctl = 0b01;
    }
    else
    {
        aipu_assert((dst >> 28) == 0xf);
    }
    aipu_assert(asid <= 3);
    ctl = ctl | (asid << 10);
    CodeBlock codes;
    if (save_ctx)
    {
        codes.add(PUSH().add_reg(R27))
            .add(PUSH().add_reg(R26));
    }
    codes.add(MOV().add_reg(R26).add_imm(0xE0000440 + 0x40 * chn))
        .add(MOV().add_reg(R27).add_imm(0))
        .add(ST().add_reg(R27).add_reg(R26).add_imm(0x24))  //ext_gap
        .add(ST().add_reg(R27).add_reg(R26).add_imm(0x14))  //int_gap
        .add(ST().add_reg(R27).add_reg(R26).add_imm(0x2c)); // sync_ctl

    auto width = size;
    if (width >= (1 << 16))
    {
        width = 1 << 15;
    }
    codes.add(MOV().add_reg(R27).add_imm((1 << 24) + size))
        .add(ST().add_reg(R27).add_reg(R26).add_imm(0x1c)) //ext_size
        .add(ST().add_reg(R27).add_reg(R26).add_imm(0xc))  //int_size
        .add(MOV().add_reg(R27).add_imm((width << 16) + width))
        .add(ST().add_reg(R27).add_reg(R26).add_imm(0x20)) //ext_ws
        .add(ST().add_reg(R27).add_reg(R26).add_imm(0x10)) //int_ws
        .add(MOV().add_reg(R27).add_imm(ext_addr))
        .add(ST().add_reg(R27).add_reg(R26).add_imm(0x18)) //ext_addr
        .add(MOV().add_reg(R27).add_imm(int_addr))
        .add(ST().add_reg(R27).add_reg(R26).add_imm(0x8)) //int_addr
        .add(MOV().add_reg(R27).add_imm(ctl))
        .add(ST().add_reg(R27).add_reg(R26).add_imm(0x0)); //start
    if (save_ctx)
    {
        codes.add(POP().add_reg(R26))
            .add(POP().add_reg(R27));
    }
    codes.enc(oss);
}

void aipu_wait_dma(std::ostream &oss, uint32_t chn = 7, bool save_ctx = true)
{
    /**
    .local WAIT_DMA_FOR_W:
	LD R0, DMA_BASE, DMA_CMN_STATUS, R7 & NOP & NOP & NOP 
	AND R0, R0, #0x3, R7 & NOP & NOP & NOP
	CBNZ R0, WAIT_DMA_FOR_W & NOP & NOP & NOP
     * 
     */
    CodeBlock codes;
    if (save_ctx)
    {
        codes.add(PUSH().add_reg(R27))
            .add(PUSH().add_reg(R26));
    }

    codes.add(MOV().add_reg(R26).add_imm(0xE0000400))
        .add(LD().add_reg(R27).add_reg(R26).add_imm(0x10))
        .add(ANDi().add_reg(R27).add_reg(R27).add_imm(1 << chn))
        .add(CBNZ().add_reg(R27).add_imm(-2));

    if (save_ctx)
    {
        codes.add(POP().add_reg(R26))
            .add(POP().add_reg(R27));
    }
    codes.enc(oss);
}

void aipu_memcpy_sync(std::ostream &oss, uint32_t src, uint32_t dst, uint32_t size, uint32_t asid = 0, uint32_t chn = 7)
{
    CodeBlock codes, codes1;
    codes.add(PUSH().add_reg(R27))
        .add(PUSH().add_reg(R26))
        .add(MVI().add_reg(R26).add_imm(0xE0000400))
        .enc(oss);

    aipu_memcpy_async(oss, src, dst, size, asid, chn, false);
    aipu_wait_dma(oss, chn, false);

    codes1.add(POP().add_reg(R26))
        .add(POP().add_reg(R27))
        .enc(oss);
}

void aipu_memcpy_lsu_scalar2(std::ostream &oss, uint32_t src, uint32_t dst, uint32_t size)
{
    // for i in range: size
    //  dst[i]=src[i]
    CodeBlock codes;
    codes.add(PUSH().add_reg(R27))
        .add(PUSH().add_reg(R26))
        .add(PUSH().add_reg(R25))
        .add(PUSH().add_reg(R24))
        .add(MOV().add_reg(R25).add_imm(size))
        .add(MOV().add_reg(R26).add_imm(src))
        .add(MOV().add_reg(R27).add_imm(dst))
        .add(SLEsi().add_reg(R24).add_reg(R25).add_imm(0))
        .add(CBNZ().add_reg(R24).add_imm(7))
        .add(LD().add_reg(R24).add_reg(R26).add_imm(0))
        .add(ST().add_reg(R24).add_reg(R27).add_imm(0))
        .add(ADDU().add_reg(R26).add_reg(R26).add_imm(4))
        .add(ADDU().add_reg(R27).add_reg(R27).add_imm(4))
        .add(SUBsi().add_reg(R25).add_reg(R25).add_imm(4))
        .add(CBZ().add_reg(R31).add_imm(-7))
        .add(POP().add_reg(R24))
        .add(POP().add_reg(R25))
        .add(POP().add_reg(R26))
        .add(POP().add_reg(R27));
    codes.enc(oss);
}

void aipu_memcpy_lsu_scalar(std::ostream &oss, uint32_t src, uint32_t dst, uint32_t size)
{
    // for i in range: size
    //  dst[i]=src[i]
    CodeBlock codes;
    codes.add(PUSH().add_reg(R27))
        .add(PUSH().add_reg(R26))
        .add(PUSH().add_reg(R25))
        .add(PUSH().add_reg(R24))
        .add(MOV().add_reg(R25).add_imm(size))
        .add(MOV().add_reg(R26).add_imm(src))
        .add(MOV().add_reg(R27).add_imm(dst))
        .add(SLEsi().add_reg(R24).add_reg(R25).add_imm(0))
        .add(CBNZ().add_reg(R24).add_imm(37))
        .add(LD().add_reg(R24).add_reg(R26).add_imm(0))
        .add(ST().add_reg(R24).add_reg(R27).add_imm(0))
        .add(LD().add_reg(R24).add_reg(R26).add_imm(4))
        .add(ST().add_reg(R24).add_reg(R27).add_imm(4))
        .add(LD().add_reg(R24).add_reg(R26).add_imm(8))
        .add(ST().add_reg(R24).add_reg(R27).add_imm(8))
        .add(LD().add_reg(R24).add_reg(R26).add_imm(12))
        .add(ST().add_reg(R24).add_reg(R27).add_imm(12))
        .add(LD().add_reg(R24).add_reg(R26).add_imm(16))
        .add(ST().add_reg(R24).add_reg(R27).add_imm(16))
        .add(LD().add_reg(R24).add_reg(R26).add_imm(20))
        .add(ST().add_reg(R24).add_reg(R27).add_imm(20))
        .add(LD().add_reg(R24).add_reg(R26).add_imm(24))
        .add(ST().add_reg(R24).add_reg(R27).add_imm(24))
        .add(LD().add_reg(R24).add_reg(R26).add_imm(28))
        .add(ST().add_reg(R24).add_reg(R27).add_imm(28))
        .add(LD().add_reg(R24).add_reg(R26).add_imm(32))
        .add(ST().add_reg(R24).add_reg(R27).add_imm(32))
        .add(LD().add_reg(R24).add_reg(R26).add_imm(36))
        .add(ST().add_reg(R24).add_reg(R27).add_imm(36))
        .add(LD().add_reg(R24).add_reg(R26).add_imm(40))
        .add(ST().add_reg(R24).add_reg(R27).add_imm(40))
        .add(LD().add_reg(R24).add_reg(R26).add_imm(44))
        .add(ST().add_reg(R24).add_reg(R27).add_imm(44))
        .add(LD().add_reg(R24).add_reg(R26).add_imm(48))
        .add(ST().add_reg(R24).add_reg(R27).add_imm(48))
        .add(LD().add_reg(R24).add_reg(R26).add_imm(52))
        .add(ST().add_reg(R24).add_reg(R27).add_imm(52))
        .add(LD().add_reg(R24).add_reg(R26).add_imm(56))
        .add(ST().add_reg(R24).add_reg(R27).add_imm(56))
        .add(LD().add_reg(R24).add_reg(R26).add_imm(60))
        .add(ST().add_reg(R24).add_reg(R27).add_imm(60))
        .add(ADDU().add_reg(R26).add_reg(R26).add_imm(64))
        .add(ADDU().add_reg(R27).add_reg(R27).add_imm(64))
        .add(SUBsi().add_reg(R25).add_reg(R25).add_imm(64))
        .add(CBZ().add_reg(R31).add_imm(-37))
        .add(POP().add_reg(R24))
        .add(POP().add_reg(R25))
        .add(POP().add_reg(R26))
        .add(POP().add_reg(R27));
    codes.enc(oss);
}

// we move init_ctx to cpp side
static void generate_init_ctx(std::ostream &oss)
{
    static bool init = false;
    static CodeBlock codes;
    if (!init)
    {
        init = true;
        codes.add(CBZ().add_reg(R31))             //CBZ r31,_START & NOP &NOP &NOP
            .add(CBZ().add_reg(R31).add_imm(0x1)) //CBZ R31,_INTERRUPT_HANLDER &NOP &NOP &NOP
            .add(EXIT_ins())                      //EXIT &NOP &NOP &NOP
            .add(IRTN());                         //IRTN &NOP &NOP &NOP
        codes.packages[0].ins[0].add_imm(codes.size());

        if (BuilderConfig::get<bool>("disable_aipu_cache"))
        {
            // disable cache (both I-cache and D-cache)
            codes.add(MVI().add_reg(R1).add_imm(0).add_reg(R7));
            codes.add(MTCTRL0HR().add_reg(R1).add_imm(CACHE_CONTROL_REGISTER));
            // loop check
            auto codes_size = codes.size();
            codes.add(MFCTRL0HR().add_reg(R1).add_imm(CACHE_CONTROL_REGISTER));
            codes.add(CBNZ().add_reg(R1).add_imm(codes_size));
        }

        codes.add(MVI().add_reg(R0).add_imm(0xe0).add_reg(R7))           //MVI r0,0xE0,r7   & NOP & NOP &NOP  ; r0=0xE0
            .add(LSLi().add_reg(R1).add_reg(R0).add_imm(24).add_reg(R7)) //LSL r1,r0,#24,r7  & NOP & NOP &NOP ; r1=0xE0<<24==0xE000 0000
            .add(LD().add_reg(R29).add_reg(R1).add_imm(4).add_reg(R7))   //LD r29,r1,#4,r7   & NOP & NOP &NOP ; r29=para_mem1=mem[0xe000 0004]

            // initialize stack
            .add(MVI().add_reg(R2).add_imm(BuilderConfig::get<uint32_t>("stack_size")).add_reg(R7)) //MVI R2, #0x2000, R7 & NOP & NOP &NOP
            .add(ADD().add_reg(R29).add_reg(R29).add_reg(R2).add_reg(R7))                           //ADD R29, R29, R2, R7 & NOP & NOP &NOP

            // init error_code
            .add(LD().add_reg(R2).add_reg(R1).add_imm(0).add_reg(R7))
            .add(LD().add_reg(R2).add_reg(R2).add_imm(48).add_reg(R7))
            .add(MVI().add_reg(R3).add_imm(0).add_reg(R7))
            .add(ST().add_reg(R3).add_reg(R2).add_imm(0).add_reg(R7))

            // init R28
            .add(LD().add_reg(R28).add_reg(R1).add_imm(0).add_reg(R7))     //LD r28,r1, #0, r7 & NOP & NOP &NOP  ; r3=para_mem0
            .add(PUSH().add_reg(R28).add_reg(R7));                         //push r28,r7 & NOP & NOP &NOP
        codes.add(LD().add_reg(R4).add_reg(R1).add_imm(0).add_reg(R7))     // r4=ro_base=mem[0xe000 0000]
            .add(LD().add_reg(R5).add_reg(R4).add_imm(11 * 4).add_reg(R7)) // r5=ro[11]=print fifo'size
            .add(CBZ().add_reg(R5).add_imm(0x4))                           //CBZ R31,_NO_INITIAL_PRINT_FIFO
            .add(LD().add_reg(R6).add_reg(R4).add_imm(10 * 4).add_reg(R7)) // r6=r0[10]=print fifo's ptr
            .add(ST().add_reg(R31).add_reg(R6).add_imm(0).add_reg(R7))     // initial the print meta data
            .add(ST().add_reg(R31).add_reg(R6).add_imm(4).add_reg(R7));
        for (int i = 3; i < 28; ++i)
        {
            //MVI  rx,#0,r7 & NOP & NOP &NOP
            codes.add(MVI().add_reg(i).add_imm(0).add_reg(R7));
        }
        if (BuilderConfig::get<bool>("DEBUG"))
        {
            // // init 0xC100 0000
            // codes.add(MVI().add_reg(R9).add_imm(0xc1).add_reg(R7))           //MVI R9, #0xC1, R7 & NOP & NOP & NOP
            //     .add(LSLi().add_reg(R9).add_reg(R9).add_imm(24).add_reg(R7)) //LSL R9, R9, #24, R7 & NOP & NOP & NOP
            //     .add(ST().add_reg(R31).add_reg(R9).add_imm(0).add_reg(R7));  //ST R31, R9, #0, R7 & NOP & NOP & NOP

            // init OP_CNT
            codes.add(MVI().add_reg(R0).add_imm(0xE0).add_reg(R7))
                .add(LSLi().add_reg(R0).add_reg(R0).add_imm(24).add_reg(R7))
                .add(LD().add_reg(R1).add_reg(R0).add_imm(0).add_reg(R7))
                .add(LD().add_reg(R2).add_reg(R1).add_imm(13 * 4).add_reg(R7)) // the op_cnt is on ro[13]
                .add(MVI().add_reg(R3).add_imm(0).add_reg(R7))
                .add(ST().add_reg(R3).add_reg(R2).add_imm(0).add_reg(R7));
        }
    }
    codes.enc(oss);
}

static void generate_connector(std::ostream &oss)
{
    CodeBlock codes;
    codes.add(POP().add_reg(R28).add_reg(R7))
        .add(ADDU().add_reg(R28).add_reg(R28).add_imm(256).add_reg(R7))
        .add(PUSH().add_reg(R28).add_reg(R7));
    codes.enc(oss);
}

static size_t get_long_call_size()
{
    static size_t ret = 0;
    if (ret)
    {
        return ret;
    }
    AsmZY::Call dummy_call;
    // due the call is dynamic len, here we call a large offset
    dummy_call.add_imm(0x7fffffff);
    ret = dummy_call.enc().size() * 4;
    return ret;
}
static void fill_bubble(std::ostream &oss, size_t s)
{
    CodeBlock bubble;
    bubble.add(nop());
    s = (s + 15) / 16; // a vliw inst musth 16bytes aligned
    for (size_t i = 0; i < s; i++)
    {
        bubble.enc(oss);
    }
}
LinkerV1::LinkerV1()
{
}
void LinkerV1::set_codes(std::vector<std::vector<std::string>> &codes)
{
    m_codes = codes;
}

void LinkerV1::reset()
{

    m_simple_functions.clear();
    m_complex_functions.clear();
    // funcname -> offset
    m_func_map.clear();
    // all entries in the map need reloc: func, offset
    // implies that in the offset, need call the func
    // after link, the map should be empty
    m_reloc_map.clear();

    /*
        define the internal ASM format
        a kernel function is a basic function

        a line is a function all, syntax:
            conv2d -> call conv2d kernel
            
        a line with .function is a new function definition, is a complex function, syntax:
             .function <function name>
              function body
             .end_function <function name>
    */
    std::vector<std::string> func_stack;
    func_stack.push_back("_main_program");
    for (auto &code : m_codes)
    {
        for (auto &line : code)
        {
            line = utils::trim(line);
            if (utils::startswith(line, ".function"))
            {
                auto name = line.substr(sizeof(".function"));
                func_stack.push_back(utils::trim(name));
                m_complex_functions[name];
                continue;
            }
            else if (utils::startswith(line, ".end_function"))
            {
                // auto name=utils::trim(line.substr(sizeof(".end_function")));
                func_stack.pop_back();
                continue;
            }
            else
            {
                // a call line
                if (func_stack.size())
                {
                    m_complex_functions[func_stack[func_stack.size() - 1]].push_back(line);
                }
            }
        }
    }
    for (auto &complex : m_complex_functions)
    {
        for (auto &func : complex.second)
        {
            if (!m_complex_functions.count(func))
            {
                m_simple_functions.insert(func);
            }
        }
    }
}

uint32_t LinkerV1::link(std::ostream &oss)
{
    reset();
    std::stringstream m_stream;
    build_program(m_stream);
    oss << m_stream.rdbuf();
    // temp for debugger
    if (m_complex_functions["_main_program"].size())
    {
        return m_func_map[m_complex_functions["_main_program"][0]];
    }
    return m_main_entry;
}
void LinkerV1::build_program_head(std::stringstream &m_stream)
{
    m_stream.seekp(0);
    auto target = get_target();
    // auto connector = get_text_content_by_filename("layer_connector.bin");
    auto exit_with_st = get_text_content_by_filename("exit_with_st.bin");
    auto profiler = aipubt::AIPUBuilder::get_profiler();
    bool enable_profiler = profiler->enable_profile();
    // first block: init_ctx
    generate_init_ctx(m_stream);
    // do some global init, profiler / dumpdebug etc

    // reset / init profiler
    // Target Z1_0701 do not support profiler
    if (enable_profiler && (target->name() != "Z1_0701") && (target->name() != "Z1_0701_P"))
    {
        BuilderText init_profiler = cast_any<BuilderText>(profiler->interface()->get_function("init_profiler_text")(0));
        m_stream.write(init_profiler.first, init_profiler.second);
    }

    // add random reg text
    if (BuilderConfig::get<bool>("rand-reg"))
    {
        std::string random_init_name = "zy_random_init.bin";
        if ((target->isainfo().type == ISA_ZYv2) || (target->isainfo().type == ISA_ZYv3))
        {
            random_init_name = "z2_random_init.bin";
        }
        auto random_regs = get_text_content_by_filename(random_init_name);
        m_stream.write(random_regs.first, random_regs.second);
    }
    // skip global ro, and skip OP_CNT
    // m_stream.write(connector.first, connector.second);
    generate_connector(m_stream);

    //for testing, insert dma cmd here!!!
    // test DMA speed  DDR->SRAM->DDR
    // size_t total_bytes = 1 * 1024 * 1024;
    // size_t one_trans_bytes = 64 * 1024;
    // auto n = total_bytes / one_trans_bytes;
    // aipu_memcpy_sync(m_stream, 0x10000000, 0xf8000000, one_trans_bytes, 0, 6);
    // for (uint32_t i = 0; i < n; ++i)
    // {
    //     aipu_wait_dma(m_stream, 6);
    //     aipu_memcpy_async(m_stream, 0xf8000000, 0x50000000 + one_trans_bytes * i,  one_trans_bytes, 0, 7);
    //     aipu_wait_dma(m_stream, 7);
    //     if (i != n - 1)
    //     {
    //         aipu_memcpy_async(m_stream, 0x10000000 + one_trans_bytes * i, 0xf8000000, one_trans_bytes, 0, 6);
    //     }
    // }

    // aipu_memcpy_lsu_scalar(m_stream, 0xf8800000, 0xf8000000, one_trans_bytes);
    // for (uint32_t i = 0; i < n; ++i)
    // {
    //     aipu_memcpy_lsu_scalar(m_stream, 0xf8000000, 0xf8800000 , one_trans_bytes);
    //     if (i != n - 1)
    //     {
    //         aipu_memcpy_lsu_scalar(m_stream, 0xf8800000 , 0xf8000000, one_trans_bytes);
    //     }
    // }
    // auto tpc_ld_ub = get_text_content_by_filename("tpc_ld_ub.bin");

    //  m_stream.write(tpc_ld_ub.first, tpc_ld_ub.second);

    // seconde call the main func and exit
    m_main_entry = m_stream.tellp();
    // fill bubble for long call as placeholder
    fill_bubble(m_stream, get_long_call_size());

    m_stream.write(exit_with_st.first, exit_with_st.second);
    // TODO! align lib to 256bit/32bytes  for AIFF descriptor for complier~~~
    CodeBlock bubble;
    bubble.add(nop());

    size_t current_offset = m_stream.tellp();
    while (current_offset & (31))
    {
        bubble.enc(m_stream);
        current_offset = m_stream.tellp();
    }
    m_lib_base = m_stream.tellp();
}

void LinkerV1::build_complex_function(std::vector<std::string> &codes, std::stringstream &m_stream)
{
    auto target = get_target();
    auto connector = get_text_content_by_filename("layer_connector.bin");
    auto reset_dma = get_text_content_by_filename("reset_dma.bin");
    //prologue and epilogue
    // prolog is only push ra
    CodeBlock prolog, epilog;
    // because the layer connector will push r28 in the stack, and pop later
    prolog.add(POP().add_reg(R28).add_reg(R7))
        .add(PUSH().add_reg(RA).add_reg(R7))
        .add(PUSH().add_reg(R28).add_reg(R7));
    prolog.enc(m_stream);
    for (auto &func : codes)
    {
        if (utils::startswith(func, "inline "))
        {
            // inline function would not update R28
            auto f = utils::replace(func, "inline ", "");
            auto inline_func = get_text_content_by_filename(f);
            m_stream.write(inline_func.first, inline_func.second);
            continue;
        }
        if (m_func_map.count(func))
        {
            AsmZY::Call call_lib;
            int lib_pos = m_func_map[func];
            lib_pos = lib_pos - m_stream.tellp();
            call_lib.add_imm(lib_pos >> 4);
            call_lib.enc(m_stream);
        }
        else
        {
            m_reloc_map[func] = m_stream.tellp();
            // fill bubble for long call as placeholder
            fill_bubble(m_stream, get_long_call_size());
        }
        if (func != Constants::profiler_code)
        {
            // profiler no need to insert connector
            m_stream.write(connector.first, connector.second);
            if (BuilderConfig::get<bool>("reset_dma_per_layer") && target->isainfo().type == ISA_ZY)
            {
                m_stream.write(reset_dma.first, reset_dma.second);
            }
        }
    }
    //epilog is just pop ra
    epilog.add(POP().add_reg(R28).add_reg(R7))
        .add(POP().add_reg(RA).add_reg(R7))
        .add(PUSH().add_reg(R28).add_reg(R7))
        .add(BR().add_reg(RA));
    epilog.enc(m_stream);
}

void LinkerV1::build_functions(std::stringstream &m_stream)
{
    // the br r30 code
    const unsigned char BR_R30_CODE[] = {158, 1, 66, 4, 31, 0, 240, 5,
                                         31, 0, 240, 5, 31, 0, 240, 5};
    BuilderText BR_R30 = {(const char *)BR_R30_CODE, 0x10};
    // static std::set<std::string> not_add_br={
    //     "for_range_header.bin",
    //     // "for_range_ender.bin",
    // };
    CodeBlock bubble;
    bubble.add(nop());
    m_stream.seekp(m_lib_base);
    // build simple function
    for (auto func : m_simple_functions)
    {
        BuilderText t;
        if (func == "_exit")
        {
            t = get_text_content_by_filename("exit_with_st.bin");
        }
        else if (func == Constants::profiler_code)
        {
            auto profiler = aipubt::AIPUBuilder::get_profiler();
            t = cast_any<BuilderText>(profiler->interface()->get_function("reset_counter_text")(0));
        }
        else
        {
            t = get_text_content_by_filename(func);
        }

        if (utils::startswith(func, "inline "))
        {
            continue;
        }
        m_func_map[func] = m_stream.tellp();

        m_stream.write(t.first, t.second);

        m_stream.write(BR_R30.first, BR_R30.second);
        // TODO! align lib to 256bit/32bytes  for AIFF descriptor for complier~~~
        auto current_offset = m_stream.tellp();
        while (current_offset & (31))
        {
            bubble.enc(m_stream);
            current_offset = m_stream.tellp();
        }
    }

    // build complex function
    auto built_funcs = m_simple_functions;
    auto all_funcs = m_simple_functions;
    std::vector<std::string> to_build_funcs;
    for (auto &kv : m_complex_functions)
    {
        to_build_funcs.push_back(kv.first);
        all_funcs.insert(kv.first);
    }
    auto all_in_set = [](std::set<std::string> &m_set, std::vector<std::string> &m_vec, std::string &func) {
        for (auto &v : m_vec)
        {
            if (!m_set.count(v))
            {
                func = v;
                return false;
            }
        }
        return true;
    };
    std::string no_found;
    for (auto &kv : m_complex_functions)
    {
        if (!all_in_set(all_funcs, kv.second, no_found))
        {
            FATAL() << "Function: " << kv.first << " depend a function:" << no_found << " which no found.";
        }
    }
    // build complex funcs~~~
    for (auto &kv : m_complex_functions)
    {
        m_func_map[kv.first] = m_stream.tellp();
        build_complex_function(kv.second, m_stream);
    }
    // fix reloc
    for (auto &kv : m_reloc_map)
    {
        long long reloc_pos = kv.second;
        auto fun_offset = m_func_map[kv.first];
        m_stream.seekp(reloc_pos);
        AsmZY::Call call_lib;
        call_lib.add_imm(((fun_offset - reloc_pos) >> 4));
        call_lib.enc(m_stream);
    }
    m_stream.seekp(0, std::ios_base::end);
}
void LinkerV1::build_program(std::stringstream &m_stream)
{
    auto exit_with_st = get_text_content_by_filename("exit_with_st.bin");
    build_program_head(m_stream);
    build_functions(m_stream);
    m_stream.seekp(m_main_entry);
    AsmZY::Call call_main;
    if (m_func_map.count("_main_program"))
    {
        // avoid empty graph, in this case the _main_program does NOT exist.
        size_t main_func = m_func_map["_main_program"];
        call_main.add_imm(((main_func - m_main_entry) >> 4));
        call_main.enc(m_stream);
    }
    m_stream.write(exit_with_st.first, exit_with_st.second);

    m_stream.seekp(0, std::ios_base::end);
}

BinSection LinkerV1::merge_text(const BinSection &text1, const BinSection &text2, uint32_t entry1, uint32_t entry2, uint32_t &new_entry)
{
    BinSection out;
    CodeBlock codes; // move global ro and skip global ro

    codes.add(MOV().add_reg(R0).add_imm(0xe0000000))
        .add(LD().add_reg(R0).add_reg(R0).add_imm(0).add_reg(R7))
        .add(POP().add_reg(R28).add_reg(R7))
        .add(ADDU().add_reg(R28).add_reg(R28).add_imm(256).add_reg(R7))
        .add(LD().add_reg(R2).add_reg(R28).add_imm(0).add_reg(R7))
        .add(LD().add_reg(R3).add_reg(R28).add_imm(4).add_reg(R7))

        .add(LD().add_reg(R4).add_reg(R28).add_imm(8).add_reg(R7))
        .add(ST().add_reg(R2).add_reg(R0).add_imm(0).add_reg(R7))
        .add(LD().add_reg(R5).add_reg(R28).add_imm(12).add_reg(R7))
        .add(ST().add_reg(R3).add_reg(R0).add_imm(4).add_reg(R7))
        .add(LD().add_reg(R2).add_reg(R28).add_imm(16).add_reg(R7))
        .add(ST().add_reg(R4).add_reg(R0).add_imm(8).add_reg(R7))
        .add(LD().add_reg(R3).add_reg(R28).add_imm(20).add_reg(R7))
        .add(ST().add_reg(R5).add_reg(R0).add_imm(12).add_reg(R7))

        .add(LD().add_reg(R4).add_reg(R28).add_imm(24).add_reg(R7))
        .add(ST().add_reg(R2).add_reg(R0).add_imm(16).add_reg(R7))
        .add(LD().add_reg(R5).add_reg(R28).add_imm(28).add_reg(R7))
        .add(ST().add_reg(R3).add_reg(R0).add_imm(20).add_reg(R7))
        .add(LD().add_reg(R2).add_reg(R28).add_imm(32).add_reg(R7))
        .add(ST().add_reg(R4).add_reg(R0).add_imm(24).add_reg(R7))
        .add(LD().add_reg(R3).add_reg(R28).add_imm(36).add_reg(R7))
        .add(ST().add_reg(R5).add_reg(R0).add_imm(28).add_reg(R7))

        .add(LD().add_reg(R4).add_reg(R28).add_imm(40).add_reg(R7))
        .add(ST().add_reg(R2).add_reg(R0).add_imm(32).add_reg(R7))
        .add(LD().add_reg(R5).add_reg(R28).add_imm(44).add_reg(R7))
        .add(ST().add_reg(R3).add_reg(R0).add_imm(36).add_reg(R7))
        .add(LD().add_reg(R2).add_reg(R28).add_imm(48).add_reg(R7))
        .add(ST().add_reg(R4).add_reg(R0).add_imm(40).add_reg(R7))
        .add(LD().add_reg(R3).add_reg(R28).add_imm(52).add_reg(R7))
        .add(ST().add_reg(R5).add_reg(R0).add_imm(44).add_reg(R7))

        .add(LD().add_reg(R4).add_reg(R28).add_imm(56).add_reg(R7))
        .add(ST().add_reg(R2).add_reg(R0).add_imm(48).add_reg(R7))
        .add(LD().add_reg(R5).add_reg(R28).add_imm(60).add_reg(R7))
        .add(ST().add_reg(R3).add_reg(R0).add_imm(52).add_reg(R7))
        .add(LD().add_reg(R2).add_reg(R28).add_imm(64).add_reg(R7))
        .add(ST().add_reg(R4).add_reg(R0).add_imm(56).add_reg(R7))
        .add(LD().add_reg(R3).add_reg(R28).add_imm(68).add_reg(R7))
        .add(ST().add_reg(R5).add_reg(R0).add_imm(60).add_reg(R7))

        .add(ST().add_reg(R2).add_reg(R0).add_imm(64).add_reg(R7))
        .add(ST().add_reg(R3).add_reg(R0).add_imm(68).add_reg(R7))

        .add(ADDU().add_reg(R28).add_reg(R28).add_imm(256).add_reg(R7))
        .add(PUSH().add_reg(R28).add_reg(R7));

    auto connector = codes.enc();
    uint32_t pad_size = 0;
    if (entry2 < connector.size() * 4)
    {
        pad_size = connector.size() * 4 - entry2;
    }
    out.bin_size = text1.bin_size + text2.bin_size + pad_size;
    out.bin = new char[out.bin_size];
    memcpy(out.bin, text1.bin, text1.bin_size);
    memcpy(out.bin + text1.bin_size + pad_size, text2.bin, text2.bin_size);
    size_t rewrite_offset = entry1 + get_long_call_size();
    new_entry = entry2 + text1.bin_size + pad_size;
    auto re0 = text1.bin_size + pad_size + entry2 - connector.size() * 4;
    for (auto ins : connector)
    {
        *(uint32_t *)(out.bin + re0) = ins;
        re0 += 4;
    }
    size_t offset = new_entry - rewrite_offset - connector.size() * 4;
    AsmZY::Call call_text2;
    call_text2.add_imm(offset >> 4);
    auto text = call_text2.enc();
    for (auto ins : text)
    {
        *(uint32_t *)(out.bin + rewrite_offset) = ins;
        rewrite_offset += 4;
    }
    return out;
}

} // namespace aipubt