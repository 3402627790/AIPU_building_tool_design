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
 * @Last Modified time: 2024-06-24 10:48:32
 */
#pragma once
#include <vm/asmtranscode.h>
#include <vm/vm.h>

namespace aipubt
{
struct alignas(32) YMMValue
{
    uint8_t value[32];
};
using IT = InstructionType;
const std::string get_local_temp_label();
void prologue(Xbyak::CodeGenerator &cg, RegMap &reg_map);
void epilogue(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Xbyak::Reg &ret_reg = Xbyak::Reg());
void nop(Xbyak::CodeGenerator &, RegMap &reg_map, const Instruction &, const TransContext &);
const YMMValue *get_pmsk_ref(uint8_t dtype);
/**
 * @brief Apply predication mask to the destination register
 *  will apply td =  td&(~msk) | (new_ymm & msk)
 * @param cg       code generator
 * @param reg_map    register map
 * @param td     destination ymm register
 * @param new_ymm  new value of ymm register
 * @param pg [0-6] pg register number 
 * @param p_type [1/2/4] predication type, bytes/half/word, default is 4
 * 
 */
void apply_pg_mask(Xbyak::CodeGenerator &cg, RegMap &reg_map, Xbyak::Ymm &td, Xbyak::Ymm &new_ymm, uint32_t pg, uint32_t p_type = 4);
} //namespace aipubt