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
 * @Date: 2024-06-21 16:38:09 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-08-01 11:43:07
 */
#include "scalar.h"

namespace aipubt
{
void bl(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx)
{
    using namespace Xbyak::util;
    reg_map.flush(cg);
    prologue(cg, reg_map);
    cg.mov(rax, (uint64_t)jump);
    if (ins.type == IT::BL ||
        ins.type == IT::BRL ||
        ins.type == IT::BRRL)
    {
        // update r30
        cg.mov(esi, (ctx.current_pc + 1) << 4);
        cg.mov(word[rdi + ROFFSET + 30 * 4], esi);
    }
    if (ins.operands[0].is_imm())
    {

        cg.mov(rsi, (ins.operands[0].imm() << 3) + (ctx.current_pc << 4));
    }
    else if (ins.type == IT::BRR ||
             ins.type == IT::BRRL)
    {
        cg.mov(esi, word[rdi + ROFFSET + ins.operands[0].value * 4]);
        cg.add(esi, ctx.current_pc << 4);
    }
    else
    {
        cg.mov(rsi, word[rdi + ROFFSET + ins.operands[0].value * 4]);
    }
    cg.call(rax); // call jump to get the function
    epilogue(cg, reg_map, rax);
    cg.jmp(rax);
}

void cbz(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx)
{
    using namespace Xbyak::util;
    reg_map.flush(cg);
    prologue(cg, reg_map);
    cg.cmp(reg_map.get_reg32(ins.operands[0], cg), 0);
    auto l = get_local_temp_label();
    if (ins.type == IT::CBNZ)
    {
        cg.je(l);
    }
    else
    {
        cg.jne(l);
    }
    cg.mov(esi, (ins.operands[1].imm() << 3) + ((ctx.current_pc) << 4));
    cg.mov(rax, (uint64_t)jump);
    cg.call(rax); // call jump to get the function
    epilogue(cg, reg_map, rax);
    cg.jmp(rax);
    cg.L(l);
    cg.mov(esi, ((ctx.current_pc + 1) << 4));
    cg.mov(rax, (uint64_t)jump);
    cg.call(rax); // call jump to get the function
    epilogue(cg, reg_map, rax);
    cg.jmp(rax);
}

void loop(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx)
{
    using namespace Xbyak::util;
    UNUSED(ctx);
    if (ins.operands[0].is_imm())
    {
        reg_map.flush(cg);
        // auto rd = reg_map.temp_reg32(cg);
        // cg.mov(rd, );
        cg.mov(word[rdi + LOOP_COUNT_OFFSET], ins.operands[0].imm());
    }
    else
    {
        auto rd = reg_map.get_reg32(ins.operands[0], cg);
        reg_map.flush(cg);
        cg.mov(word[rdi + LOOP_COUNT_OFFSET], rd.cvt32());
    }
}
void loopend(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx)
{
    using namespace Xbyak::util;
    UNUSED(ins);
    auto rtemp = reg_map.temp_reg32(cg);
    cg.mov(rtemp, word[rdi + LOOP_COUNT_OFFSET]);
    cg.sub(rtemp, 1);
    cg.mov(word[rdi + LOOP_COUNT_OFFSET], rtemp);
    cg.cmp(rtemp, 0);
    // jump to loop+1
    reg_map.flush(cg);
    auto l = get_local_temp_label();
    cg.jle(l);
    prologue(cg, reg_map);
    cg.mov(esi, (uint32_t)ctx.loop_start);
    cg.mov(rax, (uint64_t)jump);
    cg.call(rax); // call jump to get the function
    epilogue(cg, reg_map, rax);
    cg.jmp(rax);
    cg.L(l);
}

static void wfe2(TecStatus &status)
{
    if (status.core)
    {
        status.core->tec_sync();
    }
    return;
}
void wfe(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx)
{
    // Wait for event
    // we only implement wfe1 and wfe2
    // wfe0/wfe3 is by default will just by pass
    using namespace Xbyak::util;
    int mode = ins.operands[1].imm();
    if (mode == 0 || mode == 3)
    {
        return; // bypass
    }
    //TODO FIXME implement the wfe1
    if (mode == 2)
    {
        using namespace Xbyak::util;
        prologue(cg, reg_map);
        cg.mov(rax, (uint64_t)wfe2);
        cg.call(rax);
        epilogue(cg, reg_map);
    }
}

}; //namespace aipubt