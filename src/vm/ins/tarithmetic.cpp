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

void tadd(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx)
{
    using namespace Xbyak::util;
    auto td = reg_map.get_treg(ins.operands[0], cg);
    auto tn = reg_map.get_treg(ins.operands[1], cg);
    if (ins.operands[2].is_imm())
    {
        auto r = reg_map.temp_reg32(cg);
        uint64_t imm = ins.operands[2].imm();
        auto xmm = Xbyak::Xmm(td.getIdx());
        if (ins.type == IT::ADD_IMM_B_UNSH8_UNPRED)
        {
            cg.mov(r, imm);
            cg.vmovd(xmm, r);
            cg.vpbroadcastb(td, xmm);
            cg.vpaddb(td, td, tn);
        }
        else if (ins.type == IT::ADD_IMM_H_UNSH8_UNPRED)
        {
            cg.mov(r, imm << 16ul | imm);
            cg.vmovd(xmm, r);
            cg.vpbroadcastd(td, xmm);
            cg.vpaddw(td, tn, td);
        }
        else if (ins.type == IT::ADD_IMM_W_UNSH8_UNPRED)
        {
            cg.mov(r, imm);
            cg.movd(xmm, r);
            cg.vpbroadcastd(td, xmm);
            cg.vpaddd(td, tn, td);
        }
        reg_map.release_reg(r);
    }
    else
    {
        auto tm = reg_map.get_treg(ins.operands[2], cg);
        if (ins.type == IT::ADD_SS_REG_B_UNPRED ||
            (ins.type == IT::ADD_REG_B_PRED && ins.operands[3].value == 7))
        {
            cg.vpaddb(td, tn, tm);
        }
        else if (ins.type == IT::ADD_SS_REG_H_UNPRED ||
                 (ins.type == IT::ADD_REG_H_PRED && ins.operands[3].value == 7))
        {
            cg.vpaddw(td, tn, tm);
        }
        else if (ins.type == IT::ADD_SS_REG_W_UNPRED ||
                 (ins.type == IT::ADD_REG_W_PRED && ins.operands[3].value == 7))
        {
            cg.vpaddd(td, tn, tm);
        }
        else if (ins.type == IT::ADD_REG_B_PRED)
        {
            auto tmp = reg_map.temp_treg(cg);
            auto pg = ins.operands[3].value;
            cg.vpaddb(tmp, tn, tm);
            apply_pg_mask(cg, reg_map, td, tmp, pg, 1);
            reg_map.release_reg(tmp);
        }
        else if (ins.type == IT::ADD_REG_H_PRED)
        {
            auto tmp = reg_map.temp_treg(cg);
            auto pg = ins.operands[3].value;
            cg.vpaddw(tmp, tn, tm);
            apply_pg_mask(cg, reg_map, td, tmp, pg, 2);
            reg_map.release_reg(tmp);
        }
        else if (ins.type == IT::ADD_REG_W_PRED)
        {
            auto tmp = reg_map.temp_treg(cg);
            auto pg = ins.operands[3].value;
            cg.vpaddd(tmp, tn, tm);
            apply_pg_mask(cg, reg_map, td, tmp, pg, 4);
            reg_map.release_reg(tmp);
        }
    }
}

}; //namespace aipubt