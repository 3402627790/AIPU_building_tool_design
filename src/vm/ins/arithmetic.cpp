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
 * @Last Modified time: 2024-06-24 10:48:51
 */
#include "scalar.h"

namespace aipubt
{

void add(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx)
{
    using namespace Xbyak::util;
    auto rn = reg_map.get_reg32(ins.operands[1], cg);
    // must get rn first, if we want to use no mov for rd, to avoid rd==rn
    if (ins.operands[2].is_imm())
    {
        auto rd = reg_map.get_reg32(ins.operands[0], cg, true);
        if (rd != rn)
        {
            cg.mov(rd, rn);
        }
        cg.add(rd, ins.operands[2].imm());
    }
    else
    {
        auto rm = reg_map.get_reg32(ins.operands[2], cg);
        auto rd = reg_map.get_reg32(ins.operands[0], cg, true);
        if (rd == rn) // add r0 r0, r1
        {
            cg.add(rd, rm);
        }
        else if (rd == rm) // add r0 r1, r0
        {
            cg.add(rd, rn);
        }
        else // add r0, r1,r2
        {
            cg.mov(rd, rn);
            cg.add(rd, rm);
        }
        if (ins.type == IT::ADDS_U)
        {
            auto tmp = reg_map.temp_reg32(cg);
            cg.mov(tmp, -1);
            cg.cmovc(rd, tmp);
        }
    }
}

void sadds(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx)
{
    using namespace Xbyak::util;
    auto rn = reg_map.get_reg32(ins.operands[1], cg);
    // must get rn first, if we want to use no mov for rd, to avoid rd==rn
    auto rm = reg_map.get_reg32(ins.operands[2], cg);
    auto rd = reg_map.get_reg32(ins.operands[0], cg, true);
    auto is_sub = ins.type == IT::SUBS_S;
    cg.movsxd(rn.cvt64(), rn);
    if (rn != rm)
    {
        cg.movsxd(rm.cvt64(), rm);
    }
    if (rd == rn) // add r0 r0, r1
    {
        if (is_sub)
        {
            cg.sub(rd.cvt64(), rm.cvt64());
        }
        else
        {
            cg.add(rd.cvt64(), rm.cvt64());
        }
    }
    else if (rd == rm) // add r0 r1, r0
    {
        if (is_sub)
        {
            cg.sub(rd.cvt64(), rn.cvt64());
            cg.neg(rd.cvt64());
        }
        else
        {
            cg.add(rd.cvt64(), rn.cvt64());
        }
    }
    else // add r0, r1,r2
    {
        cg.mov(rd.cvt64(), rn.cvt64());
        if (is_sub)
        {
            cg.sub(rd.cvt64(), rm.cvt64());
        }
        else
        {
            cg.add(rd.cvt64(), rm.cvt64());
        }
    }
    auto overflow = get_local_temp_label();
    auto end = get_local_temp_label();
    cg.cmp(rd.cvt64(), 2147483647);
    cg.jg(overflow);
    cg.cmp(rd.cvt64(), -2147483648);
    cg.jg(end);
    cg.mov(rd.cvt64(), -2147483648);
    cg.jmp(end);
    cg.L(overflow);
    cg.mov(rd.cvt64(), 2147483647);
    cg.L(end);
}
void sub(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx)
{
    using namespace Xbyak::util;

    if (ins.operands[2].is_imm())
    {
        auto rn = reg_map.get_reg32(ins.operands[1], cg);
        auto rd = reg_map.get_reg32(ins.operands[0], cg, true);
        if (rd != rn)
        {
            cg.mov(rd, rn);
        }
        cg.sub(rd, ins.operands[2].imm());
    }
    else if (ins.operands[2] == ins.operands[1])
    {
        auto rd = reg_map.get_reg32(ins.operands[1], cg, true);
        cg.mov(rd, 0);
    }
    else
    {
        auto rn = reg_map.get_reg32(ins.operands[1], cg);
        auto rm = reg_map.get_reg32(ins.operands[2], cg);
        auto rd = reg_map.get_reg32(ins.operands[0], cg, true);
        if (rd == rm)
        {
            cg.sub(rd, rn);
            if (ins.type == IT::SUBS_U)
            {
                auto tmp = reg_map.temp_reg32(cg);
                cg.mov(tmp, 0);
                cg.cmovnb(rd, tmp);
            }
            cg.neg(rd);
        }
        else
        {
            if (rd != rn)
            {
                cg.mov(rd, rn);
            }
            cg.sub(rd, rm);

            if (ins.type == IT::SUBS_U)
            {
                auto tmp = reg_map.temp_reg32(cg);
                cg.mov(tmp, 0);
                cg.cmovb(rd, tmp);
            }
        }
    }
}

void maxmin(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx)
{
    using namespace Xbyak::util;
    auto rn = reg_map.get_reg32(ins.operands[1], cg);
    // must get rn first, if we want to use no mov for rd, to avoid rd==rn
    auto rm = reg_map.get_reg32(ins.operands[2], cg);
    auto rd = reg_map.get_reg32(ins.operands[0], cg, true);

    if (rd == rm) // max r0 r1, r0
    {
        rm = rn;
    }
    else if (rd != rn) // max r0 r0, r1
    {
        cg.mov(rd, rn);
    }
    cg.cmp(rm, rd);
    if (ins.type == IT::MAX_S)
    {
        cg.cmovg(rd, rm);
    }
    else if (ins.type == IT::MIN_S)
    {
        cg.cmovl(rd, rm);
    }
    else if (ins.type == IT::MAX_U)
    {
        cg.cmova(rd, rm);
    }
    else if (ins.type == IT::MIN_U)
    {
        cg.cmovb(rd, rm);
    }
}

void absins(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx)
{
    using namespace Xbyak::util;
    auto rn = reg_map.get_reg32(ins.operands[1], cg);
    auto rd = reg_map.get_reg32(ins.operands[0], cg, true);
    cg.cmp(rn, 0);
    if (rd == rn) // max r0 r1, r0
    {
        auto l = get_local_temp_label();
        cg.jns(l);
        cg.neg(rd);
        cg.L(l);
    }
    else
    {
        cg.mov(rd, rn);
        cg.neg(rd);
        cg.cmovs(rd, rn);
    }
}

void mul(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx)
{
    using namespace Xbyak::util;
    // must get rn first, if we want to use no mov for rd, to avoid rd==rn
    if (ins.type == IT::MUL)
    {
        reg_map.get_reg32(ins.operands[2], cg, edx); //expected should be first
        auto rn = reg_map.get_reg32(ins.operands[1], cg);
        auto rd = reg_map.get_reg32(ins.operands[0], cg, true);
        auto tmp = reg_map.temp_reg32(cg);
        cg.mulx(tmp, rd, rn);
    }
    else if (ins.type == IT::MULH_U)

    {
        reg_map.get_reg32(ins.operands[2], cg, edx);
        auto rn = reg_map.get_reg32(ins.operands[1], cg);
        auto rd = reg_map.get_reg32(ins.operands[0], cg, true);
        auto tmp = reg_map.temp_reg32(cg);
        cg.mulx(rd, tmp, rn);
    }
    else if (ins.type == IT::MULH_S)
    {
        auto rn = reg_map.get_reg32(ins.operands[1], cg);
        auto rm = reg_map.get_reg32(ins.operands[2], cg);
        auto rd = reg_map.get_reg32(ins.operands[0], cg, true);
        cg.movsxd(rn.cvt64(), rn);
        if (rn != rm)
        {
            cg.movsxd(rm.cvt64(), rm);
        }
        if (rd == rn) // add r0 r0, r1
        {
            cg.imul(rd.cvt64(), rm.cvt64());
        }
        else if (rd == rm) // add r0 r1, r0
        {
            cg.imul(rd.cvt64(), rn.cvt64());
        }
        else // add r0, r1,r2
        {
            cg.mov(rd.cvt64(), rn.cvt64());
            cg.imul(rd.cvt64(), rm.cvt64());
        }
        cg.sar(rd.cvt64(), 32);
    }
}

void mov(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx)
{
    using namespace Xbyak::util;
    auto rd = reg_map.get_reg32(ins.operands[0], cg, true);
    // std::cout << "imm=" << ins.operands[1].imm() << std::endl;
    cg.mov(rd, ins.operands[1].imm());
}
void movh(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx)
{
    using namespace Xbyak::util;
    auto rd = reg_map.get_reg32(ins.operands[0], cg);
    cg.and_(rd, 0xfff);
    cg.or_(rd, (int32_t)ins.operands[1].imm() << 16);
}

}; //namespace aipubt