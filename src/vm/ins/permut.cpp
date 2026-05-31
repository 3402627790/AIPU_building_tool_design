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

void exte(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx)
{
    // "EXTL_B_UNPRED", "EXTL_H_UNPRED", "EXTL_W_UNPRED",
    // "EXTH_B_UNPRED", "EXTH_H_UNPRED", "EXTH_W_UNPRED",
    // "EXTO_B_UNPRED", "EXTO_H_UNPRED", "EXTO_W_UNPRED",
    // "EXTE_B_UNPRED", "EXTE_H_UNPRED", "EXTE_W_UNPRED",

    // "PEXTL_B_UNPRED", "PEXTL_H_UNPRED", "PEXTL_W_UNPRED",
    // "PEXTH_B_UNPRED", "PEXTH_H_UNPRED", "PEXTH_W_UNPRED",
    // "PEXTO_B_UNPRED", "PEXTO_H_UNPRED", "PEXTO_W_UNPRED",
    // "PEXTE_B_UNPRED", "PEXTE_H_UNPRED", "PEXTE_W_UNPRED",
    using namespace Xbyak::util;
    auto td = reg_map.get_treg(ins.operands[0], cg); //ymm0  <-> t10
    auto tn = reg_map.get_treg(ins.operands[1], cg); //ymm1       t11
    auto tm = reg_map.get_treg(ins.operands[2], cg);

    if (ins.type == IT::EXTL_B_UNPRED || ins.type == IT::EXTL_H_UNPRED || ins.type == IT::EXTL_W_UNPRED)
    {

        cg.vperm2i128(td, tn, tm, 0b100000);
    }

    else if (ins.type == IT::EXTH_B_UNPRED || ins.type == IT::EXTH_H_UNPRED || ins.type == IT::EXTH_W_UNPRED)
    {
        cg.vperm2i128(td, tn, tm, 0b110001);
    }
    else if (ins.type == IT::EXTO_B_UNPRED ||
             ins.type == IT::EXTE_B_UNPRED ||
             ins.type == IT::EXTO_H_UNPRED ||
             ins.type == IT::EXTE_H_UNPRED ||
             ins.type == IT::EXTE_W_UNPRED ||
             ins.type == IT::EXTO_W_UNPRED)
    {
        auto idx = reg_map.temp_reg32(cg).cvt64();
        auto temp = reg_map.temp_treg(cg);
        static const YMMValue index[] = {
            {0, 2, 4, 6, 8, 10, 12, 14, //exte
             1, 3, 5, 7, 9, 11, 13, 15,
             0, 2, 4, 6, 8, 10, 12, 14,
             1, 3, 5, 7, 9, 11, 13, 15},

            {1, 3, 5, 7, 9, 11, 13, 15, //exto
             0, 2, 4, 6, 8, 10, 12, 14,
             1, 3, 5, 7, 9, 11, 13, 15,
             0, 2, 4, 6, 8, 10, 12, 14},

            {0, 1, 4, 5, 8, 9, 12, 13, //exte_h
             2, 3, 6, 7, 10, 11, 14, 15,
             0, 1, 4, 5, 8, 9, 12, 13,
             2, 3, 6, 7, 10, 11, 14, 15},

            {2, 3, 6, 7, 10, 11, 14, 15, //exto_h
             0, 1, 4, 5, 8, 9, 12, 13,
             2, 3, 6, 7, 10, 11, 14, 15,
             0, 1, 4, 5, 8, 9, 12, 13},

            {0, 1, 2, 3, 8, 9, 10, 11, //exte_w
             4, 5, 6, 7, 12, 13, 14, 15,
             0, 1, 2, 3, 8, 9, 10, 11,
             4, 5, 6, 7, 12, 13, 14, 15},

            {4, 5, 6, 7, 12, 13, 14, 15, //exto_w
             0, 1, 2, 3, 8, 9, 10, 11,
             4, 5, 6, 7, 12, 13, 14, 15,
             0, 1, 2, 3, 8, 9, 10, 11},

        };
        uint64_t idx_ = (uint64_t)&index;
        if (ins.type == (uint32_t)IT::EXTO_B_UNPRED)
        {
            idx_ += 32;
        }
        else if (ins.type == (uint32_t)IT::EXTE_H_UNPRED)
        {
            idx_ += 64;
        }
        else if (ins.type == (uint32_t)IT::EXTO_H_UNPRED)
        {
            idx_ += 96;
        }
        else if (ins.type == (uint32_t)IT::EXTE_W_UNPRED)
        {
            idx_ += 128;
        }
        else if (ins.type == (uint32_t)IT::EXTO_W_UNPRED)
        {
            idx_ += 160;
        }

        cg.mov(idx, idx_);
        uint8_t imm = 2 << 4;
        if (tn != td && tm == td)
        {
            tm = tn;
            tn = td;
            imm = 2;
        }
        cg.vpshufb(temp, tn, ptr[idx]);
        cg.vpermq(td, temp, 0b10001000);
        if (tm != td)
        {
            cg.vpshufb(temp, tm, ptr[idx]);
            cg.vpermq(temp, temp, 0b10001000);
            cg.vperm2i128(td, td, temp, imm);
        }
        reg_map.release_reg(idx);
        reg_map.release_reg(temp);
    }
}

}; //namespace aipubt