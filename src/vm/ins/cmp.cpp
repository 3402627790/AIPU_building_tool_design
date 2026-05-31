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
 * @Last Modified time: 2024-06-24 10:48:41
 */
#include "scalar.h"

namespace aipubt
{
/*
scalar cmp

CMP_EQ_SIMM
CMP_NEQ_SIMM
CMP_EQ_UIMM
CMP_NEQ_UIMM
CMP_SGE_IMM
CMP_SGT_IMM
CMP_SLE_IMM
CMP_SLT_IMM
CMP_UGE_IMM
CMP_UGT_IMM
CMP_ULE_IMM
CMP_ULT_IMM
CMP_EQ_REG
CMP_NEQ_REG
CMP_SGE_REG
CMP_SGT_REG
CMP_SLE_REG
CMP_SLT_REG
CMP_UGE_REG
CMP_UGT_REG
CMP_ULE_REG
CMP_ULT_REG
CMP_QGT_SP
CMP_QGE_SP
CMP_QLT_SP
CMP_QLE_SP
CMP_QEQ_SP
CMP_QNE_SP
CMP_QOD_SP
CMP_QUO_SP
CMP_SGT_SP
CMP_SGE_SP
CMP_SLT_SP
CMP_SLE_SP
CMP_SEQ_SP
CMP_SNE_SP
CMP_SOD_SP
CMP_SUO_SP
CMP_QGT_HP
CMP_QGE_HP
CMP_QLT_HP
CMP_QLE_HP
CMP_QEQ_HP
CMP_QNE_HP
CMP_QOD_HP
CMP_QUO_HP
CMP_SGT_HP
CMP_SGE_HP
CMP_SLT_HP
CMP_SLE_HP
CMP_SEQ_HP
CMP_SNE_HP
CMP_SOD_HP
CMP_SUO_HP
*/
void cmp(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx)
{
    using namespace Xbyak::util;
    auto rd = reg_map.get_reg32(ins.operands[0], cg);
    auto rn = reg_map.get_reg32(ins.operands[1], cg);
    if (ins.operands[2].is_imm())
    {
        cg.cmp(rn, ins.operands[2].imm());
    }
    else
    {
        auto rm = reg_map.get_reg32(ins.operands[2], cg);
        cg.cmp(rn, rm);
    }
    if (ins.type == IT::CMP_EQ_SIMM || ins.type == IT::CMP_EQ_UIMM || ins.type == IT::CMP_EQ_REG)
    {
        cg.sete(rd.cvt8());
    }
    else if (ins.type == IT::CMP_NEQ_SIMM || ins.type == IT::CMP_NEQ_UIMM || ins.type == IT::CMP_NEQ_REG)
    {
        cg.setne(rd.cvt8());
    }
    else if (ins.type == IT::CMP_SLT_IMM || ins.type == IT::CMP_SLT_REG)
    {
        cg.setl(rd.cvt8());
    }
    else if (ins.type == IT::CMP_SLE_IMM || ins.type == IT::CMP_SLE_REG)
    {
        cg.setle(rd.cvt8());
    }
    else if (ins.type == IT::CMP_SGT_IMM || ins.type == IT::CMP_SGT_REG)
    {
        cg.setg(rd.cvt8());
    }
    else if (ins.type == IT::CMP_SGE_IMM || ins.type == IT::CMP_SGE_REG)
    {
        cg.setge(rd.cvt8());
    }
    else if (ins.type == IT::CMP_UGE_REG || ins.type == IT::CMP_UGE_IMM) //unsigned
    {
        cg.setae(rd.cvt8());
    }
    else if (ins.type == IT::CMP_UGT_REG || ins.type == IT::CMP_UGT_IMM) //unsigned
    {
        cg.seta(rd.cvt8());
    }
    else if (ins.type == IT::CMP_ULE_REG || ins.type == IT::CMP_ULE_IMM) //unsigned
    {
        cg.setbe(rd.cvt8());
    }
    else if (ins.type == IT::CMP_ULT_REG || ins.type == IT::CMP_ULT_IMM) //unsigned
    {
        cg.setb(rd.cvt8());
    }

    cg.movzx(rd.cvt64(), rd.cvt8());
    // todo support more compare
}

}; //namespace aipubt