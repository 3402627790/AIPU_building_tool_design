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
 * @Last Modified time: 2024-12-16 13:45:21
 */
#include "scalar.h"
#include <immintrin.h>
namespace aipubt
{

static inline uint8_t *check_offset(TecStatus &status, uint32_t offset, size_t s, bool &valid)
{
    valid = true;
    if (offset > 0xfa000000 && offset <= 0xfb000000) // accessing LM
    {
        if (offset - 0xfa000000 - 4 >= LMSIZE)
        {
            std::cout << "LM out of range\n";
            valid = false;
            return 0;
        }
        return (uint8_t *)(status.lm + (offset - 0xfa000000));
    }
    else if (offset > 0xf8000000 && offset <= 0xf9000000) // accessing SM
    {
        if (offset - 0xf8000000 - 4 >= SMSIZE)
        {
            std::cout << "SM out of range\n";
            valid = false;
            return 0;
        }
        if (status.core == 0)
        {
            std::cout << "SM core not initialized, due to Core is not initialized.\n";
            valid = false;
            return 0;
        }
        return (uint8_t *)(status.core->get_sm() + (offset - 0xfa000000));
    }
    else if (offset > 0xe0000000)
    {
        std::cout << "read reserve memory ... not implemented!\n";
        valid = false;
        return 0;
    }
    return 0;
}
template <typename T>
uint32_t load32(TecStatus &status, uint32_t offset)
{
    bool valid;
    auto src = check_offset(status, offset, 4, valid);
    if (!valid)
    {
        return 0;
    }
    if (src)
    {
        return *(T *)src;
    }
    auto &mem = *status.mem;
    return mem.read<T>(status.asx.as0 + offset);
}

void load256(TecStatus &status, uint32_t offset, int32_t tn)
{
    bool valid;
    auto src = check_offset(status, offset, 32, valid);
    if (!valid)
    {
        return;
    }
    if (src)
    {
        memcpy((char *)&(status.TReg[tn].u8), (char *)(src), 32);
    }
    else
    {
        auto &mem = *status.mem;
        mem.read(status.asx.as0 + offset, (char *)&(status.TReg[tn].u8), 32);
    }
}

template <uint32_t dtype>
void load256s(TecStatus &status, uint32_t offset, int32_t tn, uint32_t s)
{
    bool valid;
    auto src = check_offset(status, offset, 32 * s, valid);
    if (!valid)
    {
        return;
    }
    if (src)
    {
        for (int i = 0; i < 32 / dtype; ++i)
        {
            memcpy((&status.TReg[tn].u8) + i * dtype, src + i * s, dtype);
        }
    }
    else
    {
        auto &mem = *status.mem;
        for (int i = 0; i < 32 / dtype; ++i)
        {
            mem.read(status.asx.as0 + offset + i * s, (char *)&(status.TReg[tn].u8) + i * dtype, dtype);
        }
    }
}

// p lower 8bits is the pg number
template <uint32_t dtype>
void load256p(TecStatus &status, uint32_t offset, int32_t tn, uint32_t p)
{
    if (p > 7)
    {
        std::cout << "invalid Pg register number!\n";
    }
    if (p == 7)
    {
        return load256(status, offset, tn);
    }
    __m256i &t = *(__m256i *)&status.TReg[tn];
    __m256i &msk = *(__m256i *)&status.PRegMask[p];
    load256(status, offset, tn);
    __m256i &ref = *(__m256i *)get_pmsk_ref(dtype);
    auto m = _mm256_and_si256(msk, ref);
    if (dtype == 1)
    {
        m = _mm256_cmpeq_epi8(m, ref);
    }
    else if (dtype == 2)
    {
        m = _mm256_cmpeq_epi16(m, ref);
    }
    else if (dtype == 4)
    {
        m = _mm256_cmpeq_epi32(m, ref);
    }
    auto td = _mm256_andnot_si256(m, t);
    auto v = _mm256_and_si256(m, *(__m256i *)status.TReg[tn]);
    td = _mm256_or_si256(td, v);
    status.TReg[tn] = *(V256 *)&td;
}

template <uint32_t S>
void loadf(TecStatus &status, uint32_t offset, int32_t tn)
{
    // std::cout << "load256 offset = " << offset << ", tn=" << tn << "\n";
    bool valid;
    auto src = check_offset(status, offset, S, valid);
    if (!valid)
    {
        return;
    }
    if (src)
    {
        memcpy((char *)&(status.TReg[tn].u8), (char *)(src), S);
    }
    else
    {
        auto &mem = *status.mem;
        mem.read(status.asx.as0 + offset, (char *)&(status.TReg[tn].u8), S);
    }
}

void write_memory(TecStatus &status, uint32_t offset, uint64_t value, uint32_t size)
{

    bool valid;
    auto ptr = check_offset(status, offset, size, valid);
    if (!valid)
    {
        return;
    }
    if (ptr)
    {
        for (auto i = 0u; i < size; ++i)
        {
            ptr[i] = ((uint8_t *)&value)[i];
        }
        return;
    }
    auto &mem = *status.mem;
    mem.write(status.asx.as0 + offset, (char *)(&value), size);
}

void store256(TecStatus &status, uint32_t offset, int32_t tn)
{
    bool valid;
    auto ptr = check_offset(status, offset, 32, valid);
    if (!valid)
    {
        return;
    }
    if (ptr)
    {
        for (auto i = 0u; i < 32; ++i)
        {
            ptr[i] = ((uint8_t *)&status.TReg[tn])[i];
        }
        return;
    }
    auto &mem = *status.mem;
    mem.write(status.asx.as0 + offset, (char *)&(status.TReg[tn]), sizeof(V256));
}

void load(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx)
{
    using namespace Xbyak::util;
    // calling convention: DI (RDI), SI (RSI), DX (RDX), CX (RCX), R8, R9
    // rdi is always TecStatus
    //uint64_t read_memory(TecStatus& status,int32_t offset)
    // ld rd,rn ,offset
    reg_map.get_reg32(ins.operands[1], cg, esi);
    reg_map.sync_reg(cg, rax);            // free rax
    reg_map.release_reg(ins.operands[0]); // will remap to rax
    prologue(cg, reg_map);
    cg.add(esi, ins.operands[2].imm());
    if (ins.type == IT::LD)
    {
        cg.mov(rax, (uint64_t)load32<uint32_t>);
    }
    else if (ins.type == IT::LDB_S)
    {
        cg.mov(rax, (uint64_t)load32<int8_t>);
    }
    else if (ins.type == IT::LDB_U)
    {
        cg.mov(rax, (uint64_t)load32<uint8_t>);
    }
    else if (ins.type == IT::LDH_S)
    {
        cg.mov(rax, (uint64_t)load32<int16_t>);
    }
    else if (ins.type == IT::LDH_U)
    {
        cg.mov(rax, (uint64_t)load32<uint16_t>);
    }

    else
    {
        //error TODO!!!
    }
    cg.call(rax);
    epilogue(cg, reg_map);
    reg_map.map(ins.operands[0], eax);
}

void tload(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx)
{
    using namespace Xbyak::util;
    // calling convention: DI (RDI), SI (RSI), DX (RDX), CX (RCX), R8, R9
    // rdi is always TecStatus
    //uint64_t read_memory(TecStatus& status,int32_t offset)
    // ld rd,rn ,offset
    reg_map.get_reg32(ins.operands[1], cg, esi);
    reg_map.release_reg(ins.operands[0]); //load256 will directly write to tec status
    prologue(cg, reg_map);
    cg.add(esi, ins.operands[2].imm());
    cg.mov(edx, ins.operands[0].value);
    if (ins.type == IT::TLD_UNPRED)
    {
        cg.mov(rax, (uint64_t)load256);
    }
    else if (ins.type == IT::LD_HP)
    {
        cg.mov(rax, (uint64_t)loadf<2>);
    }
    else if (ins.type == IT::LD_SP)
    {
        cg.mov(rax, (uint64_t)loadf<4>);
    }
    cg.call(rax);
    epilogue(cg, reg_map);
}

void st(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx)
{
    using namespace Xbyak::util;
    //set_memory(TecStatus &status, int32_t offset, uint64_t value, uint32_t size
    // st rn, rm, offset
    auto rm = reg_map.get_reg32(ins.operands[1], cg, esi);
    reg_map.get_reg32(ins.operands[0], cg, edx);
    prologue(cg, reg_map);
    cg.add(rm, ins.operands[2].imm()); //offset
    if (ins.type == IT::ST)
    {
        cg.mov(ecx, 4); // size more store here
    }
    else if (ins.type == IT::STH)
    {
        cg.mov(ecx, 2);
    }
    else if (ins.type == IT::STB)
    {
        cg.mov(ecx, 1);
    }
    else
    {
        //error !!!!!!! TODO
    }
    cg.mov(rax, (uint64_t)write_memory);
    cg.call(rax);
    epilogue(cg, reg_map);
}

void tst(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx)
{
    using namespace Xbyak::util;
    //set_memory(TecStatus &status, int32_t offset, uint64_t value, uint32_t size)
    // st rn, rm, offset
    reg_map.sync_reg(cg, ins.operands[0]); // flush to memory
    auto rm = reg_map.get_reg32(ins.operands[1], cg, esi);
    prologue(cg, reg_map);
    cg.add(rm, ins.operands[2].imm());  //offset
    cg.mov(edx, ins.operands[0].value); // size more store here
    cg.mov(rax, (uint64_t)store256);
    cg.call(rax);
    epilogue(cg, reg_map);
}

}; //namespace aipubt