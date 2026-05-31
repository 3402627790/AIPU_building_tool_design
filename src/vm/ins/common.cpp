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
 * @Date: 2024-06-21 16:38:59 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-12-16 11:39:20
 */
#include "common.h"

namespace aipubt
{
const std::string get_local_temp_label()
{
    static uint64_t local_id = 0;
    return ".tmplocal" + std::to_string(local_id++);
}
void prologue(Xbyak::CodeGenerator &cg, RegMap &reg_map) //backup mapping register
{
    using namespace Xbyak::util;
    // calling convention: DI (RDI), SI (RSI), DX (RDX), CX (RCX), R8, R9
    // rdi is always TecStatus
    // stack pointer rsp must be 16byte aligned
    cg.push(rbp);
    cg.mov(rbp, rsp);
    //rdi must be saved
    //!TODO support Ymm and float mapping
    auto mapped = reg_map.get_mapped();
    cg.sub(rsp, ((8 * mapped.size() + 8) + 15) & ~0xf); //align to 16bytes
    cg.mov(qword[rbp - 8], rdi);
    for (auto i = 0u; i < mapped.size(); ++i)
    {
        cg.mov(qword[rbp - 16 - i * 8], mapped[i].cvt64());
    }
}
void epilogue(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Xbyak::Reg &ret_reg)
{
    using namespace Xbyak::util;
    //rdi must be saved
    //!TODO support Ymm and float mapping
    auto mapped = reg_map.get_mapped();
    cg.mov(rdi, qword[rbp - 8]);
    int rax_mapped = -1;
    // we need very calfully to deal with rax, when rax is mapped, since rax is the return value of function
    for (auto i = 0u; i < mapped.size(); ++i)
    {
        if (mapped[i].cvt64() == rax)
        {
            rax_mapped = i;
            continue;
        }
        cg.mov(mapped[i].cvt64(), qword[rbp - 16 - i * 8]);
    }
    if (rax_mapped > -1)
    {
        if (ret_reg.isNone())
        {
            cg.mov(rax, qword[rbp - 16 - rax_mapped * 8]);
        }
        else if (ret_reg.cvt64() != rax)
        {
            cg.mov(ret_reg.cvt64(), rax);
            cg.mov(rax, qword[rbp - 16 - rax_mapped * 8]);
        }
    }
    cg.leave();
}

void NEXT(Xbyak::CodeGenerator &cg, RegMap &reg_map, uint32_t current_pc)
{
    using namespace Xbyak::util;
    reg_map.flush(cg);
    prologue(cg, reg_map);
    cg.mov(rax, (uint64_t)jump);
    cg.mov(rsi, (1 << 4) + (current_pc << 4));
    cg.call(rax); // call jump to get the function
    epilogue(cg, reg_map, rax);
    cg.jmp(rax);
}

void nop(Xbyak::CodeGenerator &, RegMap &reg_map, const Instruction &, const TransContext &ctx){};

void EXIT(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx)
{
    UNUSED(reg_map);
    UNUSED(ins);
    UNUSED(ctx);
    using namespace Xbyak::util;
    reg_map.flush(cg);
    cg.ret();
}

void *jump(TecStatus &status, uint32_t offset)
{
    // std::cout << "jump to " << (offset) << std::endl;
    auto tec = status.tec;
    return (void *)(tec->get_code((offset + 7) & 0xfffffff8));
}

const YMMValue *get_pmsk_ref(uint8_t dtype)
{
    static const YMMValue pmskref1 = {0xff, 0xff, 0xff, 0xff,
                                      0xff, 0xff, 0xff, 0xff,
                                      0xff, 0xff, 0xff, 0xff,
                                      0xff, 0xff, 0xff, 0xff,
                                      0xff, 0xff, 0xff, 0xff,
                                      0xff, 0xff, 0xff, 0xff,
                                      0xff, 0xff, 0xff, 0xff,
                                      0xff, 0xff, 0xff, 0xff};
    static const YMMValue pmskref2 = {0xff, 0, 0xff, 0,
                                      0xff, 0, 0xff, 0,
                                      0xff, 0, 0xff, 0,
                                      0xff, 0, 0xff, 0,
                                      0xff, 0, 0xff, 0,
                                      0xff, 0, 0xff, 0,
                                      0xff, 0, 0xff, 0,
                                      0xff, 0, 0xff, 0};
    static const YMMValue pmskref4 = {0xff, 0, 0, 0,
                                      0xff, 0, 0, 0,
                                      0xff, 0, 0, 0,
                                      0xff, 0, 0, 0,
                                      0xff, 0, 0, 0,
                                      0xff, 0, 0, 0,
                                      0xff, 0, 0, 0,
                                      0xff, 0, 0, 0};
    if (dtype == 1)
    {
        return &pmskref1;
    }
    else if (dtype == 2)
    {
        return &pmskref2;
    }
    else if (dtype == 4)
    {
        return &pmskref4;
    }
    else
    {
        return nullptr;
    }
}

void apply_pg_mask(Xbyak::CodeGenerator &cg, RegMap &reg_map, Xbyak::Ymm &td, Xbyak::Ymm &new_ymm, uint32_t pg, uint32_t p_type)
{
    assert(pg < 7 && "pg must be less than 7");
    assert((p_type == 1 || p_type == 2 || p_type == 4) && "p_type accepts 1/2/4");
    using namespace Xbyak::util;
    auto msk = reg_map.temp_treg(cg);
    auto p = reg_map.temp_reg32(cg).cvt64();
    cg.vmovaps(msk, ptr[rdi + pg * 32 + PMskOFFSET]);
    size_t ref = (uint64_t)get_pmsk_ref(4);
    if (p_type == 1)
    {
        ref = (uint64_t)get_pmsk_ref(1);
        cg.mov(p, ref);
        cg.vpand(msk, msk, ptr[p]);
        cg.vpcmpeqb(msk, msk, ptr[p]);
    }
    else if (p_type == 2)
    {
        ref = (uint64_t)get_pmsk_ref(2);
        cg.mov(p, ref);
        cg.vpand(msk, msk, ptr[p]);
        cg.vpcmpeqw(msk, msk, ptr[p]);
    }
    else
    {
        cg.mov(p, ref);
        cg.vpand(msk, msk, ptr[p]);
        cg.vpcmpeqd(msk, msk, ptr[p]);
    }

    cg.vpandn(td, msk, td);
    cg.vpand(new_ymm, new_ymm, msk);
    cg.vpor(td, td, new_ymm);
    reg_map.release_reg(p);
    reg_map.release_reg(msk);
}

} //namespace aipubt