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
 * @Last Modified time: 2024-07-31 14:27:52
 */
#include "scalar.h"

namespace aipubt
{
/*
scalar bitwise

AND
AND_L_IMM
AND_H_IMM
OR
OR_L_IMM
OR_H_IMM
XOR
XOR_L_IMM
XOR_H_IMM
ASR
ASR_IMM
LSR
LSR_IMM
ROR
ROR_IMM
LSL
LSL_IMM
BFE_S
BFE_U
BFC
BFI
BFS
BFD
*/
void bitwise(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx)
{
    using namespace Xbyak::util;

    auto rn = reg_map.get_reg32(ins.operands[1], cg);
    if (ins.type == IT::CLZ)
    {
        auto rd = reg_map.get_reg32(ins.operands[0], cg, true);
        cg.lzcnt(rd, rn);
    }
    else if (ins.type == IT::CLS)
    {
        auto rd = reg_map.get_reg32(ins.operands[0], cg, true);
        if (rd != rn)
        {
            cg.mov(rd, rn);
        }
        cg.not_(rd);
        cg.lzcnt(rd, rd);
    }
    else if (ins.type == IT::PCNT)
    {
        auto rd = reg_map.get_reg32(ins.operands[0], cg, true);
        cg.popcnt(rd, rn);
    }
    else if (ins.type == IT::BFD)
    {
        auto rd = reg_map.get_reg32(ins.operands[0], cg);
        auto index = ins.operands[2].imm();
        auto w = ins.operands[3].imm() + 1;
        auto mask = ((1 << w) - 1) << index;
        auto temp = reg_map.temp_reg32(cg);
        cg.mov(temp, rn);
        cg.and_(rd, ~mask);
        cg.and_(temp, mask);
        cg.or_(rd, temp);
    }
    else if (ins.operands[2].is_imm())
    {
        auto rd = reg_map.get_reg32(ins.operands[0], cg);
        if (rd != rn)
        {
            cg.mov(rd, rn);
        }
        if (ins.type == IT::AND_L_IMM)
        {
            cg.and_(rd, ins.operands[2].imm());
        }
        else if (ins.type == IT::AND_H_IMM)
        {
            cg.and_(rd, ~ins.operands[2].imm());
        }
        else if (ins.type == IT::OR_L_IMM)
        {
            cg.or_(rd, ins.operands[2].imm());
        }
        else if (ins.type == IT::OR_H_IMM)
        {
            cg.or_(rd, ~ins.operands[2].imm());
        }
        else if (ins.type == IT::XOR_L_IMM)
        {
            cg.xor_(rd, ins.operands[2].imm());
        }
        else if (ins.type == IT::XOR_H_IMM)
        {
            cg.xor_(rd, ~ins.operands[2].imm());
        }
        else if (ins.type == IT::ASR_IMM)
        {
            cg.sar(rd, ins.operands[2].imm());
        }
        else if (ins.type == IT::LSR_IMM)
        {
            cg.shr(rd, ins.operands[2].imm());
        }
        else if (ins.type == IT::LSL_IMM)
        {
            cg.shl(rd, ins.operands[2].imm());
        }
        else if (ins.type == IT::ROR_IMM)
        {
            cg.rorx(rd, rn, ins.operands[2].imm());
        }
        else if (ins.type == IT::BFS)
        {
            auto index = ins.operands[2].imm();
            auto w = ins.operands[3].imm() + 1;
            auto value = ((1 << w) - 1) << index;
            cg.or_(rd, value);
        }
        else if (ins.type == IT::BFC)
        {
            auto index = ins.operands[2].imm();
            auto w = ins.operands[3].imm() + 1;
            auto value = ((1 << w) - 1) << index;
            auto mask = ~value;
            cg.and_(rd, mask);
        }
        else if (ins.type == IT::BFI)
        {
            auto index = ins.operands[2].imm();
            auto w = ins.operands[3].imm() + 1;
            auto value = ((1 << w) - 1) << index;
            cg.xor_(rd, value);
        }
        else if (ins.type == IT::BFE_U)
        {
            auto index = ins.operands[2].imm();
            auto w = ins.operands[3].imm() + 1;
            auto mask = ((1 << w) - 1);
            cg.shr(rd, index);
            cg.and_(rd, mask);
        }
        else if (ins.type == IT::BFE_S)
        {
            //todo
            auto index = ins.operands[2].imm();
            auto w = ins.operands[3].imm() + 1;
            auto mask = ((1 << w) - 1);
            cg.shr(rd, index);
            cg.and_(rd, mask);
            cg.bt(rd, w - 1);
            auto l = get_local_temp_label();
            cg.jnc(l);
            cg.or_(rd, ~mask);
            cg.L(l);
        }

        // else if (ins.type == IT::BRV)
        // {
        //     auto temp = reg_map.temp_reg32();
        //     cg.pdep(rd,rn,0xf0000000);
        // }
    }
    else if (ins.type == IT::ROR)
    {
        auto rd = reg_map.get_reg32(ins.operands[0], cg);
        if (ins.operands[0] == ins.operands[2])
        {
            reg_map.unmap(ecx, cg);
            cg.mov(ecx, rd);
        }
        else
        {
            reg_map.get_reg32(ins.operands[2], cg, ecx);
        }
        if (ins.operands[0] != ins.operands[1])
        {
            cg.mov(rd, rn);
        }
        cg.ror(rd, cl);

        return;
    }
    else
    {
        auto rm = reg_map.get_reg32(ins.operands[2], cg);
        auto rd = reg_map.get_reg32(ins.operands[0], cg, true);

        if (ins.type == IT::AND)
        {
            if (rd == rm)
            {
                rm = rn;
            }
            else if (rd != rn)
            {
                cg.mov(rd, rn);
            }
            cg.and_(rd, rm);
        }
        else if (ins.type == IT::OR)
        {
            if (rd == rm)
            {
                rm = rn;
            }
            else if (rd != rn)
            {
                cg.mov(rd, rn);
            }
            cg.or_(rd, rm);
        }
        else if (ins.type == IT::XOR)
        {
            if (rd == rm)
            {
                rm = rn;
            }
            else if (rd != rn)
            {
                cg.mov(rd, rn);
            }
            cg.xor_(rd, rm);
        }
        else if (ins.type == IT::ASR)
        {
            cg.sarx(rd, rn, rm);
        }
        else if (ins.type == IT::LSR)
        {
            cg.shrx(rd, rn, rm);
        }
        else if (ins.type == IT::LSL)
        {
            cg.shlx(rd, rn, rm);
        }
    }
}

static uint32_t bit_reverse(TecStatus &status, uint32_t v)
{
    static const uint8_t map[] = {
        0b0000,
        0b1000,
        0b0100,
        0b1100,
        0b0010,
        0b1010,
        0b0110,
        0b1110,
        0b0001,
        0b1001,
        0b0101,
        0b1101,
        0b0011,
        0b1011,
        0b0111,
        0b1111,
    };
    uint32_t ret = map[v >> 28];
    ret = ret | (map[(v >> 24) & 0xf] << 4);
    ret = ret | (map[(v >> 20) & 0xf] << 8);
    ret = ret | (map[(v >> 16) & 0xf] << 12);
    ret = ret | (map[(v >> 12) & 0xf] << 16);
    ret = ret | (map[(v >> 8) & 0xf] << 20);
    ret = ret | (map[(v >> 4) & 0xf] << 24);
    ret = ret | (map[(v)&0xf] << 28);
    return ret;
}
void brv(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx)
{
    using namespace Xbyak::util;
    reg_map.get_reg32(ins.operands[1], cg, esi);
    reg_map.sync_reg(cg, rax);            // free rax
    reg_map.release_reg(ins.operands[0]); // will remap to rax
    prologue(cg, reg_map);
    cg.mov(rax, (uint64_t)bit_reverse);
    cg.call(rax);
    epilogue(cg, reg_map);
    reg_map.map(ins.operands[0], eax);
}
}; //namespace aipubt