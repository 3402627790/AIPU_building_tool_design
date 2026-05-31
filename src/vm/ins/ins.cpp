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
 * @Last Modified time: 2024-12-16 17:18:35
 */
#include "fp_ins.h"
#include "scalar.h"

namespace aipubt
{

std::map<uint32_t, CGFunc> &get_trans_map()
{
    static std::map<uint32_t, CGFunc> trans_map;
    return trans_map;
}

//common
RegisterTransCode(NOP, nop);
RegisterTransCode(EXIT, EXIT);

//arithmetic
RegisterTransCode(SUB_IMM, sub);
RegisterTransCode(SUB, sub);
RegisterTransCode(ADD, add);
RegisterTransCode(ADDS_U, add);
RegisterTransCode(ADDS_S, sadds);
RegisterTransCode(SUBS_S, sadds);
RegisterTransCode(SUBS_U, sub);
RegisterTransCode(ADD_IMM, add);
RegisterTransCode(MAX_S, maxmin);
RegisterTransCode(MAX_U, maxmin);
RegisterTransCode(MIN_S, maxmin);
RegisterTransCode(MIN_U, maxmin);
RegisterTransCode(MOVIL, mov);
RegisterTransCode(MOVIH, movh);
RegisterTransCode(ABS, absins);
RegisterTransCode(MUL, mul);
RegisterTransCode(MULH_U, mul);
RegisterTransCode(MULH_S, mul);

//bitwise
RegisterTransCode(AND, bitwise);
RegisterTransCode(AND_L_IMM, bitwise);
RegisterTransCode(AND_H_IMM, bitwise);
RegisterTransCode(OR, bitwise);
RegisterTransCode(OR_L_IMM, bitwise);
RegisterTransCode(OR_H_IMM, bitwise);
RegisterTransCode(XOR, bitwise);
RegisterTransCode(XOR_L_IMM, bitwise);
RegisterTransCode(XOR_H_IMM, bitwise);
RegisterTransCode(ASR, bitwise);
RegisterTransCode(ASR_IMM, bitwise);
RegisterTransCode(LSR, bitwise);
RegisterTransCode(LSR_IMM, bitwise);
RegisterTransCode(LSL, bitwise);
RegisterTransCode(LSL_IMM, bitwise);
RegisterTransCode(ROR, bitwise);
RegisterTransCode(ROR_IMM, bitwise);
RegisterTransCode(BFE_S, bitwise);
RegisterTransCode(BFE_U, bitwise);
RegisterTransCode(BFC, bitwise);
RegisterTransCode(BFI, bitwise);
RegisterTransCode(BFS, bitwise);
RegisterTransCode(BFD, bitwise);
RegisterTransCode(CLZ, bitwise);
RegisterTransCode(CLS, bitwise);
RegisterTransCode(PCNT, bitwise);
RegisterTransCode(BRV, brv);

//branch
RegisterTransCode(B, bl);
RegisterTransCode(BL, bl);
RegisterTransCode(BR, bl);
RegisterTransCode(BRL, bl);
RegisterTransCode(BRRL, bl);
RegisterTransCode(BRR, bl);
RegisterTransCode(CBZ, cbz);
RegisterTransCode(CBNZ, cbz);
RegisterTransCode(LOOP_IMM, loop);
RegisterTransCode(LOOP_REG, loop);
RegisterTransCode(LOOPEND, loopend);
RegisterTransCode(WFE, wfe);

//lsu
RegisterTransCode(LD, load);
RegisterTransCode(LDB_U, load);
RegisterTransCode(LDB_S, load);
RegisterTransCode(LDH_S, load);
RegisterTransCode(LDH_U, load);
RegisterTransCode(LD_HP, tload);
RegisterTransCode(LD_SP, tload);
RegisterTransCode(TLD_UNPRED, tload);
RegisterTransCode(ST, st);
RegisterTransCode(STB, st);
RegisterTransCode(STH, st);
RegisterTransCode(TST_UNPRED, tst);

//compare
RegisterTransCode(CMP_EQ_SIMM, cmp);
RegisterTransCode(CMP_EQ_UIMM, cmp);
RegisterTransCode(CMP_EQ_REG, cmp);
RegisterTransCode(CMP_NEQ_SIMM, cmp);
RegisterTransCode(CMP_NEQ_UIMM, cmp);
RegisterTransCode(CMP_NEQ_REG, cmp);
RegisterTransCode(CMP_SLT_IMM, cmp);
RegisterTransCode(CMP_SLE_IMM, cmp);
RegisterTransCode(CMP_SGT_IMM, cmp);
RegisterTransCode(CMP_SGE_IMM, cmp);
RegisterTransCode(CMP_SGE_REG, cmp);
RegisterTransCode(CMP_SGT_REG, cmp);
RegisterTransCode(CMP_SLE_REG, cmp);
RegisterTransCode(CMP_SLT_REG, cmp);
RegisterTransCode(CMP_ULT_IMM, cmp);
RegisterTransCode(CMP_ULE_IMM, cmp);
RegisterTransCode(CMP_UGT_IMM, cmp);
RegisterTransCode(CMP_UGE_IMM, cmp);
RegisterTransCode(CMP_UGE_REG, cmp);
RegisterTransCode(CMP_UGT_REG, cmp);
RegisterTransCode(CMP_ULE_REG, cmp);
RegisterTransCode(CMP_ULT_REG, cmp);

//vector arith
RegisterTransCode(ADD_IMM_B_UNSH8_UNPRED, tadd);
RegisterTransCode(ADD_IMM_H_UNSH8_UNPRED, tadd);
RegisterTransCode(ADD_IMM_W_UNSH8_UNPRED, tadd);
RegisterTransCode(ADD_SS_REG_B_UNPRED, tadd);
RegisterTransCode(ADD_SS_REG_H_UNPRED, tadd);
RegisterTransCode(ADD_SS_REG_W_UNPRED, tadd);
RegisterTransCode(ADD_REG_B_PRED, tadd);
RegisterTransCode(ADD_REG_H_PRED, tadd);
RegisterTransCode(ADD_REG_W_PRED, tadd);

//FP arith
RegisterTransCode(ADD_SP_PRED, fp_vadd);
RegisterTransCode(ADD_HP_PRED, fp16_vadd);

//vector perm
RegisterTransCode(EXTL_B_UNPRED, exte);
RegisterTransCode(EXTL_H_UNPRED, exte);
RegisterTransCode(EXTL_W_UNPRED, exte);
RegisterTransCode(EXTH_B_UNPRED, exte);
RegisterTransCode(EXTH_H_UNPRED, exte);
RegisterTransCode(EXTH_W_UNPRED, exte);
RegisterTransCode(EXTO_B_UNPRED, exte);
RegisterTransCode(EXTO_H_UNPRED, exte);
RegisterTransCode(EXTO_W_UNPRED, exte);
RegisterTransCode(EXTE_B_UNPRED, exte);
RegisterTransCode(EXTE_H_UNPRED, exte);
RegisterTransCode(EXTE_W_UNPRED, exte);
// RegisterTransCode(PEXTL_B_UNPRED, exte);
// RegisterTransCode(PEXTH_B_UNPRED, exte);
// // RegisterTransCode(EXTE_B_UNPRED, exte);
// RegisterTransCode(PEXTO_B_UNPRED, exte);
// // RegisterTransCode(EXTE_W_UNPRED, exte);
// RegisterTransCode(PEXTE_B_UNPRED, exte);
// RegisterTransCode(PEXTL_H_UNPRED, exte);
// RegisterTransCode(PEXTH_H_UNPRED, exte);
// RegisterTransCode(PEXTO_H_UNPRED, exte);
// RegisterTransCode(PEXTE_H_UNPRED, exte);
// RegisterTransCode(PEXTL_W_UNPRED, exte);
// RegisterTransCode(PEXTH_W_UNPRED, exte);
// RegisterTransCode(PEXTO_W_UNPRED, exte);
// RegisterTransCode(PEXTE_W_UNPRED, exte);

// 10.190.12.27
}; //namespace aipubt