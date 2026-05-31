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
 * @Author: Lang Hu
 * @Date: 2024-08-14 10:25:08
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-08-14 10:25:08
 */
#pragma once
#include "asm/asm.h"
#include "core/logger.h"
#include "op.h"
#include "reg.h"
#include "utils/utils.h"

namespace tongue
{
namespace sbe
{
    using InstructionType = aipubt::InstructionType;
    using Instruction = aipubt::Instruction;
    using InsPackage = aipubt::InsPackage;
    using InsPackages = aipubt::InsPackages;

    struct CBZ : public Instruction
    {
        CBZ(RCode rd, int32_t offset) : Instruction((uint32_t)InstructionType::CBZ, 0)
        {
            operands.push_back(Instruction::Operand(Instruction::Operand::RReg, rd));
            operands.push_back(Instruction::Operand(Instruction::Operand::Imm, *(uint32_t *)(&offset)));
        }
        CBZ(const Reg &rd, int32_t offset) : CBZ((RCode)rd.index(), offset) {}
        CBZ(RCode rd, const std::string &label) : Instruction((uint32_t)InstructionType::CBZ, 0)
        {
            operands.push_back(Instruction::Operand(Instruction::Operand::RReg, rd));
            operands.push_back(Instruction::Operand(label));
        }
        CBZ(const Reg &rd, const std::string &label) : CBZ((RCode)rd.index(), label) {}
    };

    struct CBNZ : public CBZ
    {
        CBNZ(RCode rd, int32_t offset) : CBZ(rd, offset)
        {
            type = (uint32_t)InstructionType::CBNZ;
        }
        CBNZ(const Reg &rd, int32_t offset) : CBNZ((RCode)rd.index(), offset) {}
        CBNZ(RCode rd, const std::string &label) : CBZ(rd, label)
        {
            type = (uint32_t)InstructionType::CBNZ;
        }
        CBNZ(const Reg &rd, const std::string &label) : CBNZ((RCode)rd.index(), label) {}
    };

    struct BL : public Instruction
    {
        BL(int32_t offset) : Instruction((uint32_t)InstructionType::BL, 0)
        {
            operands.push_back(Instruction::Operand(Instruction::Operand::Imm, *(uint32_t *)(&offset)));
        }
        BL(const std::string &label) : Instruction((uint32_t)InstructionType::BL, 0)
        {
            operands.push_back(Instruction::Operand(label));
        }
    };

    struct BR : public Instruction
    {
        BR(RCode reg) : Instruction((uint32_t)InstructionType::BR, 0)
        {
            operands.push_back(Instruction::Operand(Instruction::Operand::RReg, reg));
        }
        BR(const Reg &reg) : BR((RCode)reg.index()) {}
    };

    inline Instruction::Operand::OperandType get_operand_type(const Reg &reg)
    {
        auto type = Instruction::Operand::None;
        if (reg.is_rreg())
        {
            type = Instruction::Operand::RReg;
        }
        else if (reg.is_freg())
        {
            type = Instruction::Operand::FReg;
        }
        aipu_assert(type != Instruction::Operand::None);
        return type;
    }

    struct MOV : public Instruction
    {
        MOV(const Reg &rd, const Reg &rn) : Instruction((uint32_t)InstructionType::NOP, 0)
        {
            if (rd.is_rreg() && rn.is_rreg())
            {
                type = (uint32_t)InstructionType::ADD_IMM;
            }
            else if (rd.is_rreg() && rn.is_freg())
            {
                type = (uint32_t)InstructionType::MOV_SP_RD_FN;
            }
            else if (rd.is_freg() && rn.is_rreg())
            {
                type = (uint32_t)InstructionType::MOV_SP_FD_RN;
            }
            else if (rd.is_freg() && rn.is_freg())
            {
                type = (uint32_t)InstructionType::MOV_SP_FD_FN;
            }
            aipu_assert(type != (uint32_t)InstructionType::NOP);
            auto op_type = get_operand_type(rd);
            operands.push_back(Instruction::Operand(op_type, rd.index(), true));
            op_type = get_operand_type(rn);
            operands.push_back(Instruction::Operand(op_type, rn.index()));
            if (rd.is_rreg() && rn.is_rreg())
            {
                operands.push_back(Instruction::Operand(Instruction::Operand::UImm, 0));
            }
        }
        MOV(RCode rd, RCode rn) : MOV(RReg(rd), RReg(rn)) {}
        MOV(RCode rd, int32_t imm) : Instruction((uint32_t)InstructionType::MOVIL, 0)
        {
            operands.push_back(Instruction::Operand(Instruction::Operand::RReg, rd, true));
            operands.push_back(Instruction::Operand(Instruction::Operand::Imm, *(uint32_t *)(&imm)));
        }
        MOV(const Reg &rd, int32_t imm) : MOV((RCode)rd.index(), imm) {}
    };

    struct MOVH : public Instruction
    {
        MOVH(RCode rd, int32_t imm) : Instruction((uint32_t)InstructionType::MOVIH, 0)
        {
            operands.push_back(Instruction::Operand(Instruction::Operand::RReg, rd, true));
            operands.push_back(Instruction::Operand(Instruction::Operand::Imm, *(uint32_t *)(&imm)));
        }
        MOVH(const Reg &rd, int32_t imm) : MOVH((RCode)rd.index(), imm) {}
    };

    struct ST : public Instruction
    {
        ST(const Reg &rd, RCode rn, int32_t offset) : Instruction((uint32_t)InstructionType::ST, 0)
        {
            if (rd.is_freg())
            {
                type = (uint32_t)InstructionType::ST_SP;
            }
            auto op_type = get_operand_type(rd);
            operands.push_back(Instruction::Operand(op_type, rd.index()));
            operands.push_back(Instruction::Operand(Instruction::Operand::RReg, rn));
            operands.push_back(Instruction::Operand(Instruction::Operand::Imm, *(uint32_t *)(&offset)));
        }
        ST(const Reg &rd, const Reg &rn, int32_t offset) : ST(rd, (RCode)rn.index(), offset) {}
        ST(RCode rd, RCode rn, int32_t offset) : ST(RReg(rd), rn, offset) {}
    };

    struct LD : public ST
    {
        LD(const Reg &rd, RCode rn, int32_t offset) : ST(rd, rn, offset)
        {
            if (rd.is_rreg())
            {
                type = (uint32_t)InstructionType::LD;
            }
            else if (rd.is_freg())
            {
                type = (uint32_t)InstructionType::LD_SP;
            }
            operands[0].is_target = true;
        }
        LD(const Reg &rd, const Reg &rn, int32_t offset) : LD(rd, (RCode)rn.index(), offset) {}
        LD(RCode rd, RCode rn, int32_t offset) : LD(RReg(rd), rn, offset) {}
    };

    inline InstructionType get_instruction_type(Op op, Dtype rd_d, Dtype rn_d, Dtype rm_d)
    {
        auto type = InstructionType::NOP;
        switch (op)
        {
        case Op::add:
        {
            if ((rn_d == Dtype::i32 && rm_d == Dtype::i32) ||
                (rn_d == Dtype::u32 && rm_d == Dtype::u32))
            {
                type = InstructionType::ADD;
            }
            else if (rn_d == Dtype::f32 && rm_d == Dtype::f32)
            {
                type = InstructionType::ADD_SP;
            }
            break;
        }
        case Op::sub:
        {
            if ((rn_d == Dtype::i32 && rm_d == Dtype::i32) ||
                (rn_d == Dtype::u32 && rm_d == Dtype::u32))
            {
                type = InstructionType::SUB;
            }
            else if (rn_d == Dtype::f32 && rm_d == Dtype::f32)
            {
                type = InstructionType::SUB_SP;
            }
            break;
        }
        case Op::mul:
        {
            if ((rn_d == Dtype::i32 && rm_d == Dtype::i32) ||
                (rn_d == Dtype::u32 && rm_d == Dtype::u32))
            {
                type = InstructionType::MUL;
            }
            else if (rn_d == Dtype::f32 && rm_d == Dtype::f32)
            {
                type = InstructionType::MUL_SP;
            }
            break;
        }
        case Op::mulh:
        {
            if (rn_d == Dtype::i32 && rm_d == Dtype::i32)
            {
                type = InstructionType::MULH_S;
            }
            else if (rn_d == Dtype::u32 && rm_d == Dtype::u32)
            {
                type = InstructionType::MULH_U;
            }
            break;
        }
        case Op::div:
        {
            if (rn_d == Dtype::i32 && rm_d == Dtype::i32)
            {
                type = InstructionType::DIV_S;
            }
            else if (rn_d == Dtype::u32 && rm_d == Dtype::u32)
            {
                type = InstructionType::DIV_U;
            }
            else if (rn_d == Dtype::f32 && rm_d == Dtype::f32)
            {
                type = InstructionType::DIV_SP;
            }
            break;
        }
        case Op::mod:
        {
            if (rn_d == Dtype::i32 && rm_d == Dtype::i32)
            {
                type = InstructionType::MOD_S;
            }
            else if (rn_d == Dtype::u32 && rm_d == Dtype::u32)
            {
                type = InstructionType::MOD_U;
            }
            break;
        }
        case Op::slt:
        {
            if (rn_d == Dtype::i32 && rm_d == Dtype::i32)
            {
                type = InstructionType::CMP_SLT_REG;
            }
            else if (rn_d == Dtype::u32 && rm_d == Dtype::u32)
            {
                type = InstructionType::CMP_ULT_REG;
            }
            else if (rn_d == Dtype::f32 && rm_d == Dtype::f32)
            {
                type = InstructionType::CMP_QLT_SP;
            }
            break;
        }
        case Op::sle:
        {
            if (rn_d == Dtype::i32 && rm_d == Dtype::i32)
            {
                type = InstructionType::CMP_SLE_REG;
            }
            else if (rn_d == Dtype::u32 && rm_d == Dtype::u32)
            {
                type = InstructionType::CMP_ULE_REG;
            }
            else if (rn_d == Dtype::f32 && rm_d == Dtype::f32)
            {
                type = InstructionType::CMP_QLE_SP;
            }
            break;
        }
        case Op::sgt:
        {
            if (rn_d == Dtype::i32 && rm_d == Dtype::i32)
            {
                type = InstructionType::CMP_SGT_REG;
            }
            else if (rn_d == Dtype::u32 && rm_d == Dtype::u32)
            {
                type = InstructionType::CMP_UGT_REG;
            }
            else if (rn_d == Dtype::f32 && rm_d == Dtype::f32)
            {
                type = InstructionType::CMP_QGT_SP;
            }
            break;
        }
        case Op::sge:
        {
            if (rn_d == Dtype::i32 && rm_d == Dtype::i32)
            {
                type = InstructionType::CMP_SGE_REG;
            }
            else if (rn_d == Dtype::u32 && rm_d == Dtype::u32)
            {
                type = InstructionType::CMP_UGE_REG;
            }
            else if (rn_d == Dtype::f32 && rm_d == Dtype::f32)
            {
                type = InstructionType::CMP_QGE_SP;
            }
            break;
        }
        case Op::seq:
        {
            if ((rn_d == Dtype::i32 && rm_d == Dtype::i32) ||
                (rn_d == Dtype::u32 && rm_d == Dtype::u32))
            {
                type = InstructionType::CMP_EQ_REG;
            }
            else if (rn_d == Dtype::f32 && rm_d == Dtype::f32)
            {
                type = InstructionType::CMP_QEQ_SP;
            }
            break;
        }
        case Op::sne:
        {
            if ((rn_d == Dtype::i32 && rm_d == Dtype::i32) ||
                (rn_d == Dtype::u32 && rm_d == Dtype::u32))
            {
                type = InstructionType::CMP_NEQ_REG;
            }
            else if (rn_d == Dtype::f32 && rm_d == Dtype::f32)
            {
                type = InstructionType::CMP_QNE_SP;
            }
            break;
        }
        case Op::max:
        {
            if (rn_d == Dtype::i32 && rm_d == Dtype::i32)
            {
                type = InstructionType::MAX_S;
            }
            else if (rn_d == Dtype::u32 && rm_d == Dtype::u32)
            {
                type = InstructionType::MAX_U;
            }
            else if (rn_d == Dtype::f32 && rm_d == Dtype::f32)
            {
                type = InstructionType::MAX_SP;
            }
            break;
        }
        case Op::min:
        {
            if (rn_d == Dtype::i32 && rm_d == Dtype::i32)
            {
                type = InstructionType::MIN_S;
            }
            else if (rn_d == Dtype::u32 && rm_d == Dtype::u32)
            {
                type = InstructionType::MIN_U;
            }
            else if (rn_d == Dtype::f32 && rm_d == Dtype::f32)
            {
                type = InstructionType::MIN_SP;
            }
            break;
        }
        default:
            break;
        }
        if (type == InstructionType::NOP)
        {
            ERROR() << "unsupported dtype: " << op << " rd(" << rd_d << "),rn(" << rn_d << "),rm(" << rm_d << ")";
        }
        aipu_assert(type != InstructionType::NOP);
        return type;
    }

    struct Arithmetic : public Instruction
    {
        Arithmetic(Op op, RCode rd, RCode rn, uint32_t imm) : Instruction((uint32_t)InstructionType::NOP, 0)
        {
            if (op == Op::add)
            {
                type = (uint32_t)InstructionType::ADD_IMM;
            }
            else if (op == Op::sub)
            {
                type = (uint32_t)InstructionType::SUB_IMM;
            }
            aipu_assert(type != (uint32_t)InstructionType::NOP);
            operands.push_back(Instruction::Operand(Instruction::Operand::RReg, rd, true));
            operands.push_back(Instruction::Operand(Instruction::Operand::RReg, rn));
            operands.push_back(Instruction::Operand(Instruction::Operand::UImm, imm));
        }
        Arithmetic(Op op, const Reg &rd, const Reg &rn, const Reg &rm, Dtype rd_d, Dtype rn_d, Dtype rm_d)
            : Instruction((uint32_t)InstructionType::NOP, 0)
        {
            type = (uint32_t)get_instruction_type(op, rd_d, rn_d, rm_d);
            auto op_type = get_operand_type(rd);
            operands.push_back(Instruction::Operand(op_type, rd.index(), true));
            operands.push_back(Instruction::Operand(op_type, rn.index()));
            operands.push_back(Instruction::Operand(op_type, rm.index()));
        }
        Arithmetic(Op op, RCode rd, RCode rn, RCode rm)
            : Arithmetic(op, RReg(rd), RReg(rn), RReg(rm), Dtype::i32, Dtype::i32, Dtype::i32) {}
        Arithmetic(Op op, FCode fd, FCode fn, FCode fm)
            : Arithmetic(op, FReg(fd), FReg(fn), FReg(fm), Dtype::f32, Dtype::f32, Dtype::f32) {}
        Arithmetic(Op op, const Reg &rd, const Reg &rn) : Instruction((uint32_t)InstructionType::NOP, 0)
        {
            switch (op)
            {
            case Op::itof:
                type = (uint32_t)InstructionType::CVT_SP_W_UNPRED;
                break;
            case Op::ftoi:
                type = (uint32_t)InstructionType::CVT_W_SP_UNPRED;
                break;
            case Op::rint:
                type = (uint32_t)InstructionType::RINT_SP;
                break;
            case Op::log2:
                type = (uint32_t)InstructionType::LOG2_SP;
                break;
            case Op::sqrt:
                type = (uint32_t)InstructionType::SQRT_SP;
                break;
            case Op::rsqrt:
                type = (uint32_t)InstructionType::RSQRT_SP;
                break;
            default:
                break;
            }
            aipu_assert(type != (uint32_t)InstructionType::NOP);
            auto op_type = get_operand_type(rd);
            operands.push_back(Instruction::Operand(op_type, rd.index(), true));
            op_type = get_operand_type(rn);
            operands.push_back(Instruction::Operand(op_type, rn.index()));
        }
        Arithmetic(Op op, FCode fd, FCode fn) : Arithmetic(op, FReg(fd), FReg(fn)) {}
    };

    struct ADD : public Arithmetic
    {
        ADD(RCode rd, RCode rn, uint32_t imm) : Arithmetic(Op::add, rd, rn, imm) {}
        ADD(const Reg &rd, const Reg &rn, uint32_t imm)
            : Arithmetic(Op::add, (RCode)rd.index(), (RCode)rn.index(), imm) {}
        ADD(RCode rd, RCode rn, RCode rm) : Arithmetic(Op::add, rd, rn, rm) {}
        ADD(FCode fd, FCode fn, FCode fm) : Arithmetic(Op::add, fd, fn, fm) {}
    };

    struct SUB : public Arithmetic
    {
        SUB(RCode rd, RCode rn, uint32_t imm) : Arithmetic(Op::sub, rd, rn, imm) {}
        SUB(const Reg &rd, const Reg &rn, uint32_t imm)
            : Arithmetic(Op::sub, (RCode)rd.index(), (RCode)rn.index(), imm) {}
        SUB(RCode rd, RCode rn, RCode rm) : Arithmetic(Op::sub, rd, rn, rm) {}
        SUB(FCode fd, FCode fn, FCode fm) : Arithmetic(Op::sub, fd, fn, fm) {}
    };

    struct MUL : public Arithmetic
    {
        MUL(RCode rd, RCode rn, RCode rm) : Arithmetic(Op::mul, rd, rn, rm) {}
        MUL(FCode fd, FCode fn, FCode fm) : Arithmetic(Op::mul, fd, fn, fm) {}
    };

    struct DIV : public Arithmetic
    {
        DIV(RCode rd, RCode rn, RCode rm) : Arithmetic(Op::div, rd, rn, rm) {}
        DIV(FCode fd, FCode fn, FCode fm) : Arithmetic(Op::div, fd, fn, fm) {}
    };

    struct MOD : public Arithmetic
    {
        MOD(RCode rd, RCode rn, RCode rm) : Arithmetic(Op::mod, rd, rn, rm) {}
        MOD(FCode fd, FCode fn, FCode fm) : Arithmetic(Op::mod, fd, fn, fm) {}
    };

    struct CVT : public Arithmetic
    {
        CVT(FCode fd, RCode rn) : Arithmetic(Op::itof, FReg(fd), RReg(rn)) {}
        CVT(RCode rd, FCode fn) : Arithmetic(Op::ftoi, RReg(rd), FReg(fn)) {}
    };

    struct RINT : public Arithmetic
    {
        RINT(FCode fd, FCode fn) : Arithmetic(Op::rint, fd, fn) {}
    };

    struct LOG2 : public Arithmetic
    {
        LOG2(FCode fd, FCode fn) : Arithmetic(Op::log2, fd, fn) {}
    };

    struct SQRT : public Arithmetic
    {
        SQRT(FCode fd, FCode fn) : Arithmetic(Op::sqrt, fd, fn) {}
    };

    struct RSQRT : public Arithmetic
    {
        RSQRT(FCode fd, FCode fn) : Arithmetic(Op::rsqrt, fd, fn) {}
    };

    struct Bitwise : public Instruction
    {
        Bitwise(Op op, const Reg &rd, const Reg &rn, const Reg &rm)
            : Instruction((uint32_t)InstructionType::NOP, 0)
        {
            switch (op)
            {
            case Op::lshit:
                type = (uint32_t)InstructionType::LSL;
                break;
            case Op::rshit:
                type = (uint32_t)InstructionType::LSR;
                break;
            case Op::band:
                type = (uint32_t)InstructionType::AND;
                break;
            case Op::bor:
                type = (uint32_t)InstructionType::OR;
                break;
            case Op::bxor:
                type = (uint32_t)InstructionType::XOR;
                break;
            default:
                break;
            }
            aipu_assert(type != (uint32_t)InstructionType::NOP);
            operands.push_back(Instruction::Operand(Instruction::Operand::RReg, rd.index(), true));
            operands.push_back(Instruction::Operand(Instruction::Operand::RReg, rn.index()));
            operands.push_back(Instruction::Operand(Instruction::Operand::RReg, rm.index()));
        }
    };

    struct LSL : public Bitwise
    {
        LSL(const Reg &rd, const Reg &rn, const Reg &rm) : Bitwise(Op::lshit, rd, rn, rm) {}
        LSL(RCode rd, RCode rn, RCode rm) : LSL(RReg(rd), RReg(rn), RReg(rm)) {}
    };

    struct LSR : public Bitwise
    {
        LSR(const Reg &rd, const Reg &rn, const Reg &rm) : Bitwise(Op::rshit, rd, rn, rm) {}
        LSR(RCode rd, RCode rn, RCode rm) : LSR(RReg(rd), RReg(rn), RReg(rm)) {}
    };

    struct AND : public Bitwise
    {
        AND(const Reg &rd, const Reg &rn, const Reg &rm) : Bitwise(Op::band, rd, rn, rm) {}
        AND(RCode rd, RCode rn, RCode rm) : AND(RReg(rd), RReg(rn), RReg(rm)) {}
    };

    struct OR : public Bitwise
    {
        OR(const Reg &rd, const Reg &rn, const Reg &rm) : Bitwise(Op::bor, rd, rn, rm) {}
        OR(RCode rd, RCode rn, RCode rm) : OR(RReg(rd), RReg(rn), RReg(rm)) {}
    };

    struct XOR : public Bitwise
    {
        XOR(const Reg &rd, const Reg &rn, const Reg &rm) : Bitwise(Op::bxor, rd, rn, rm) {}
        XOR(RCode rd, RCode rn, RCode rm) : XOR(RReg(rd), RReg(rn), RReg(rm)) {}
    };

    struct Compare : public Instruction
    {
        Compare(Op op, const Reg &rd, const Reg &rn, const Reg &rm, Dtype rd_d, Dtype rn_d, Dtype rm_d)
            : Instruction((uint32_t)InstructionType::NOP, 0)
        {
            type = (uint32_t)get_instruction_type(op, rd_d, rn_d, rm_d);
            operands.push_back(Instruction::Operand(Instruction::Operand::RReg, rd.index(), true));
            auto op_type = get_operand_type(rn);
            operands.push_back(Instruction::Operand(op_type, rn.index()));
            op_type = get_operand_type(rm);
            operands.push_back(Instruction::Operand(op_type, rm.index()));
        }
        Compare(Op op, RCode rd, RCode rn, RCode rm)
            : Compare(op, RReg(rd), RReg(rn), RReg(rm), Dtype::i32, Dtype::i32, Dtype::i32) {}
        Compare(Op op, FCode fd, FCode fn, FCode fm)
            : Compare(op, FReg(fd), FReg(fn), FReg(fm), Dtype::f32, Dtype::f32, Dtype::f32) {}
        Compare(Op op, const Reg &rd, const Reg &rn, const Reg &rm, Dtype rn_d, Dtype rm_d)
            : Instruction((uint32_t)InstructionType::NOP, 0)
        {
            type = (uint32_t)get_instruction_type(op, Dtype::none, rn_d, rm_d);
            auto op_type = get_operand_type(rd);
            operands.push_back(Instruction::Operand(op_type, rd.index(), true));
            op_type = get_operand_type(rn);
            operands.push_back(Instruction::Operand(op_type, rn.index()));
            op_type = get_operand_type(rm);
            operands.push_back(Instruction::Operand(op_type, rm.index()));
        }
    };

    struct SLT : public Compare
    {
        SLT(RCode rd, RCode rn, RCode rm) : Compare(Op::slt, rd, rn, rm) {}
        SLT(FCode fd, FCode fn, FCode fm) : Compare(Op::slt, fd, fn, fm) {}
    };

    struct SLE : public Compare
    {
        SLE(RCode rd, RCode rn, RCode rm) : Compare(Op::sle, rd, rn, rm) {}
        SLE(FCode fd, FCode fn, FCode fm) : Compare(Op::sle, fd, fn, fm) {}
    };

    struct SGT : public Compare
    {
        SGT(RCode rd, RCode rn, RCode rm) : Compare(Op::sgt, rd, rn, rm) {}
        SGT(FCode fd, FCode fn, FCode fm) : Compare(Op::sgt, fd, fn, fm) {}
    };

    struct SGE : public Compare
    {
        SGE(RCode rd, RCode rn, RCode rm) : Compare(Op::sge, rd, rn, rm) {}
        SGE(FCode fd, FCode fn, FCode fm) : Compare(Op::sge, fd, fn, fm) {}
    };

    struct SEQ : public Compare
    {
        SEQ(RCode rd, RCode rn, RCode rm) : Compare(Op::seq, rd, rn, rm) {}
        SEQ(FCode fd, FCode fn, FCode fm) : Compare(Op::seq, fd, fn, fm) {}
    };

    struct SNE : public Compare
    {
        SNE(RCode rd, RCode rn, RCode rm) : Compare(Op::sne, rd, rn, rm) {}
        SNE(FCode fd, FCode fn, FCode fm) : Compare(Op::sne, fd, fn, fm) {}
    };
} // namespace sbe
} // namespace tongue
