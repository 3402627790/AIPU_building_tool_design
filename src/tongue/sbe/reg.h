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
 * @Date: 2023-04-03 17:22:56 
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-08-08 11:45:20
 */
#pragma once
#include <cstdint>

namespace tongue
{
namespace sbe
{
    enum RCode : int
    {
        R0 = 0,
        R1 = 1,
        R2 = 2,
        R3 = 3,
        R4 = 4,
        R5 = 5,
        R6 = 6,
        R7 = 7,
        R8 = 8,
        R9 = 9,
        R10 = 10,
        R11 = 11,
        R12 = 12,
        R13 = 13,
        R14 = 14,
        R15 = 15,
        R16 = 16,
        R17 = 17,
        R18 = 18,
        R19 = 19,
        R20 = 20,
        R21 = 21,
        R22 = 22,
        FP = 22,
        R23 = 23,
        R24 = 24,
        R25 = 25,
        R26 = 26,
        R27 = 27,
        R28 = 28,
        R29 = 29,
        SP = 29,
        R30 = 30,
        RA = 30,
        R31 = 31,
    };

    enum FCode : int
    {
        F0 = 0,
        F1 = 1,
        F2 = 2,
        F3 = 3,
        F4 = 4,
        F5 = 5,
        F6 = 6,
        F7 = 7,
        F8 = 8,
        F9 = 9,
        F10 = 10,
        F11 = 11,
        F12 = 12,
        F13 = 13,
        F14 = 14,
        F15 = 15,
        F16 = 16,
        F17 = 17,
        F18 = 18,
        F19 = 19,
        F20 = 20,
        F21 = 21,
        F22 = 22,
        F23 = 23,
        F24 = 24,
        F25 = 25,
        F26 = 26,
        F27 = 27,
        F28 = 28,
        F29 = 29,
        F30 = 30,
        F31 = 31,
    };

    class Operand
    {
    public:
        enum Kind
        {
            NONE = 0,
            RREG = 1,
            FREG = 2,
            TREG = 3,
            ACCREG = 4,
            AREG = 5,
            PREG = 6,
            EREG = 7,
            PREG_Z = 8,
            PREG_M = 9,
            OPMASK = 10,
        };

        explicit Operand(Kind kind, uint32_t bit)
            : m_kind(kind), m_bit(bit) {}
        Operand() = delete;

        Kind kind() const { return m_kind; }
        uint32_t bit() const { return m_bit; }
        bool is_none() const { return is_kind(NONE); }
        bool is_rreg() const { return is_kind(RREG); }
        bool is_freg() const { return is_kind(FREG); }
        bool is_treg() const { return is_kind(TREG); }

    private:
        bool is_kind(Kind kind) const { return (m_kind == kind); }

    protected:
        Kind m_kind = NONE;
        uint32_t m_bit = 0;
    };

    class Reg : public Operand
    {
    public:
        explicit Reg(uint32_t index, Kind kind, uint32_t bit)
            : Operand(kind, bit), m_index(index) {}
        Reg() = delete;

        uint32_t index() const { return m_index; }
        void set_index(uint32_t index) { m_index = index; }
        int offset() const { return m_offset; }
        void set_offset(int offset) { m_offset = offset; }

        bool operator==(const Reg &o) const
        {
            return (m_kind == o.m_kind && m_bit == o.m_bit) &&
                   (m_index == o.m_index && m_offset == o.m_offset);
        }
        bool operator!=(const Reg &o) const
        {
            return *this == o ? false : true;
        }

        bool spilled() const { return m_offset != 0; }

    protected:
        uint32_t m_index{0};
        int m_offset{0}; // stack relative offset
    };

    class RReg : public Reg
    {
    public:
        explicit RReg(uint32_t index) : Reg(index, RREG, 32) {}
    };

    class FReg : public Reg
    {
    public:
        explicit FReg(uint32_t index) : Reg(index, FREG, 32) {}
    };

    class TReg : public Reg
    {
    public:
        explicit TReg(uint32_t index) : Reg(index, TREG, 256) {}
    };

    class AccReg : public Reg
    {
    public:
        explicit AccReg(uint32_t index) : Reg(index, ACCREG, 256) {}
    };

    class AddrReg : public Reg
    {
    public:
        explicit AddrReg(uint32_t index) : Reg(index, AREG, 256) {}
    };

    class PReg : public Reg
    {
    public:
        explicit PReg(uint32_t index) : Reg(index, PREG, 256) {}
    };

    class EReg : public Reg
    {
    public:
        explicit EReg(uint32_t index) : Reg(index, EREG, 256) {}
    };

    inline std::ostream &operator<<(std::ostream &oss, const Reg &reg)
    {
        if (reg.is_rreg())
        {
            switch (reg.index())
            {
            case FP:
                return oss << "fp";
            case SP:
                return oss << "sp";
            case RA:
                return oss << "ra";
            default:
                return oss << "r" << reg.index();
            }
        }
        else if (reg.is_freg())
        {
            return oss << "f" << reg.index() << ".fp32";
        }
        return oss << "unkown";
    }
} // namespace sbe
} // namespace tongue
