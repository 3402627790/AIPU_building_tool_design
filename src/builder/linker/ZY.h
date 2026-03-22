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
 * @Date: 2020-01-29 14:24:59 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-11-21 11:01:27
 */
#ifndef BUILDER_LINKER_ZY_H
#define BUILDER_LINKER_ZY_H
#include <assert.h>
#include <iostream>
#include <sstream>
#include <stdint.h>
#include <vector>
// a simple assembler for linker and codegen
// here we only deal with the scalar part
namespace AsmZY
{

enum GPReg : uint32_t
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
    ZR = 31,
};

using Operand = uint32_t;

struct IMMshift
{
    uint8_t lower;
    uint8_t margin;
    uint8_t shift;
    int8_t bits; // [0-31] is valid, if negative, this imm is signed.
};
struct VLIWPackage;
struct Instruction32
{
    std::vector<VLIWPackage> pre;
    uint32_t opcode = 0x5f0001f; //nop
    // the operand bitflied is guranted by user!!
    std::vector<Operand> operands;
    std::vector<Operand> imms;
    std::vector<uint8_t> shifts;
    // imm may be split into two part, so
    // the first uint8_t is mean the bit number of lower part
    // the second is the margin of lower/higher bit
    // the third is the shift
    std::vector<IMMshift> ishifts;
    virtual uint32_t encode()
    {
        uint32_t ret = opcode;
        assert(operands.size() == shifts.size() || operands.size() + 1 == shifts.size());
        assert(imms.size() == ishifts.size());
        for (uint32_t i = 0; i < operands.size(); ++i)
        {
            ret |= (operands[i] << shifts[i]);
        }
        //auto pad R7
        if (operands.size() + 1 == shifts.size())
        {
            ret |= (R7 << shifts[shifts.size() - 1]);
        }
        for (uint32_t i = 0; i < imms.size(); ++i)
        {
            uint32_t imm = imms[i];
            auto s = ishifts[i];
            uint32_t mask = (1 << (s.lower)) - 1;
            ret |= (((imm & (~mask)) << s.margin) | (imm & mask)) << s.shift;
        }
        return ret;
    }
    virtual Instruction32 &add_operand(Operand op)
    {
        if (operands.size() == shifts.size())
        {
            assert(operands.size() < shifts.size());
        }
        operands.push_back(op);
        return *this;
    }
    virtual Instruction32 &add_reg(Operand op)
    {
        return add_operand(op);
    }

    virtual Instruction32 &add_imm(int64_t op)
    {

        assert(imms.size() < ishifts.size());
        auto s = ishifts[imms.size()];
        uint32_t imm = op;
        if (s.bits < 0 && op < 0) // convert to unsigned
        {
            assert((1 << -s.bits) + op > 0);
            imm = (1 << (-s.bits)) + op;
        }
        imms.push_back(imm);
        add_imm_hook(op);
        return *this;
    }
    virtual void add_imm_hook(int64_t op)
    {
    }
};
struct VLIWPackage
{
    Instruction32 ins[4];
    // VLIWPackage()
    // {
    //     ;
    // }
    // VLIWPackage(const std::initializer_list<Instruction32> &l)
    // {
    //     assert(l.size() == 4);
    //     int i = 0;
    //     for (auto ii : l)
    //     {
    //         ins[i] = ii;
    //         i++;
    //     }
    // }
};
struct nop : Instruction32
{
    nop() { opcode = 0x5f0001f; }
};
struct IRTN : Instruction32
{
    IRTN() { opcode = 0x4220380; }
};
struct EXIT_ins : Instruction32
{
    EXIT_ins() { opcode = 0x4120380; }
};
struct scalar0 : Instruction32
{
    scalar0() { shifts = {0}; }
};

struct scalar2 : Instruction32
{
    scalar2()
    {
        shifts = {0, 20};
        ishifts = {{0, 0, 5, 15}};
    }
};

struct scalar3 : Instruction32
{
    scalar3()
    {
        shifts = {0};
        ishifts = {{0, 0, 5, 10}};
    }
};

struct scalar4 : Instruction32
{
    scalar4()
    {
        shifts = {0, 10, 20};
        ishifts = {{5, 5, 5, 10}};
    }
};

struct scalar5 : Instruction32
{
    scalar5()
    {
        shifts = {0, 10, 20};
        ishifts = {{5, 5, 5, 9}};
    }
};

struct scalar6 : Instruction32
{
    scalar6()
    {
        shifts = {0, 10, 20};
        ishifts = {{0, 0, 5, 5}};
    }
};

struct scalar8 : Instruction32
{
    scalar8()
    {
        shifts = {0, 10, 5, 20};
    }
};

struct scalar12 : Instruction32
{
    scalar12()
    {
        shifts = {0, 10, 5};
    }
};
struct scalar14 : Instruction32
{
    scalar14()
    {
        shifts = {0, 20};
    }
};

struct scalar15 : Instruction32
{
    scalar15()
    {
        shifts = {0, 10};
        ishifts = {{5, 5, 5, 9}};
    }
};
struct scalar16 : Instruction32
{
    scalar16()
    {
        shifts = {0};
        ishifts = {{0, 0, 15, 5}, {0, 0, 5, -10}};
    }
};

struct scalar19 : Instruction32
{
    scalar19()
    {
        shifts = {0};
        ishifts = {{0, 0, 5, -15}};
    }
};

struct scalar20 : Instruction32
{
    scalar20()
    {
        shifts = {0};
    }
};

struct TBZ : scalar16
{
    TBZ()
    {
        opcode = 0x4C00000;
    }
};

struct GPC : scalar14
{
    GPC()
    {
        opcode = 0x4020280;
    }
};

struct ADDU : scalar5
{
    ADDU()
    {
        opcode = 0x9800000;
    }
};

struct ADD : scalar8
{
    ADD()
    {
        opcode = 0x40a0000;
    }
};

struct SUB : scalar8
{
    SUB()
    {
        opcode = 0x4060000;
    }
};

struct SUBsi : scalar5
{
    SUBsi()
    {
        opcode = 0xA080000;
        ishifts = {{5, 5, 5, -9}};
    }
};

struct LSLi : scalar6
{
    LSLi()
    {
        opcode = 0x40c8000;
    }
};

struct SLTs : scalar12
{
    SLTs()
    {
        opcode = 0x4080000;
    }
};

struct SLTu : scalar12
{
    SLTu()
    {
        opcode = 0x4000000;
    }
};

struct SLTsi : scalar15
{
    SLTsi()
    {
        opcode = 0x5080000;
        ishifts = {{5, 5, 5, -9}};
    }
};

struct SLEsi : scalar15
{
    SLEsi()
    {
        opcode = 0x5180000;
        ishifts = {{5, 5, 5, -9}};
    }
};
struct CBZ : scalar19
{
    CBZ()
    {
        opcode = 0x4a00000;
    }
};
struct CBNZ : scalar19
{
    CBNZ()
    {
        opcode = 0x4b00000;
    }
};
struct MVI : scalar2
{
    MVI()
    {
        opcode = 0x5800000;
    }
    void codegen()
    {
        auto op = imms[0];
        if (op >= (1 << 16))
        {
            auto rd = operands[0];
            auto rc = operands[1];
            if (op >= (1 << 25))
            {
                pre.push_back({MVI().add_reg(rd).add_imm(op >> 17).add_reg(rc), nop(), nop(), nop()});
                uint32_t lshift = 17;
                if ((op >> 8) & (0x1ff))
                {
                    pre.push_back({LSLi().add_reg(rd).add_reg(rd).add_imm(9).add_reg(rc), nop(), nop(), nop()});
                    pre.push_back({ADDU().add_reg(rd).add_reg(rd).add_imm((op >> 8) & (0x1ff)).add_reg(rc), nop(), nop(), nop()});
                    lshift = 8;
                }
                pre.push_back({LSLi().add_reg(rd).add_reg(rd).add_imm(lshift).add_reg(rc), nop(), nop(), nop()});
                if ((op) & (0xff))
                {

                    pre.push_back({ADDU().add_reg(rd).add_reg(rd).add_imm((op) & (0xff)).add_reg(rc), nop(), nop(), nop()});
                }
            }
            else
            {
                pre = {{MVI().add_reg(rd).add_imm(op >> 9).add_reg(rc), nop(), nop(), nop()},
                       {LSLi().add_reg(rd).add_reg(rd).add_imm(9).add_reg(rc), nop(), nop(), nop()}};
                if (op & 0x1ff)
                {
                    pre.push_back({ADDU().add_reg(rd).add_reg(rd).add_imm((op) & (0x1ff)).add_reg(rc), nop(), nop(), nop()});
                }
            }
            Instruction32 m_ins = pre[pre.size() - 1].ins[0];
            opcode = m_ins.opcode;
            operands = m_ins.operands;
            imms = m_ins.imms;
            shifts = m_ins.shifts;
            ishifts = m_ins.ishifts;
            // *this=m_ins;
            pre.erase(pre.begin() + pre.size() - 1);
        }
    }
    Instruction32 &add_operand(Operand op) override
    {
        Instruction32::add_operand(op);
        if (operands.size() == 2)
        {
            codegen();
        }
        return *this;
    }
};

struct BR : scalar20
{
    BR()
    {
        opcode = 0x4420180;
    }
};
struct BRL : scalar20
{
    BRL()
    {
        opcode = 0x4520180;
    }
};

struct PUSH : scalar14
{
    PUSH()
    {
        opcode = 0x4020000;
    }
};

struct POP : scalar14
{
    POP()
    {
        opcode = 0x4020200;
    }
};
struct LD : scalar5
{
    LD()
    {
        opcode = 0xF800000;
    }
};
struct ST : scalar5
{
    ST()
    {
        opcode = 0xF880000;
    }
};

struct MFCTRL0HR : scalar3
{
    MFCTRL0HR()
    {
        opcode = 0x4740000;
    }
};

struct MTCTRL0HR : scalar3
{
    MTCTRL0HR()
    {
        opcode = 0x4540000;
    }
};

struct MFCTRL2HR : scalar3
{
    MFCTRL2HR()
    {
        opcode = 0x47C0000;
    }
};

struct MTCTRL2HR : scalar3
{
    MTCTRL2HR()
    {
        opcode = 0x45C0000;
    }
};

struct ANDi : scalar4
{
    ANDi()
    {
        opcode = 0xA800000;
    }
};

struct AND : scalar8
{
    AND()
    {
        opcode = 0x4068000;
    }
};
struct LSRi : scalar6
{
    LSRi()
    {
        opcode = 0x40A8000;
    }
};

struct CodeBlock
{
    std::vector<VLIWPackage> packages;
    size_t size()
    {
        return packages.size();
    }
    CodeBlock &add(const VLIWPackage &p)
    {
        packages.push_back(p);
        return *this;
    }
    CodeBlock &add(const Instruction32 &ins, int slot = 0)
    {
        VLIWPackage p = {nop(), nop(), nop(), nop()};
        p.ins[slot] = ins;
        for (auto l : ins.pre)
        {
            packages.push_back(l);
        }
        return add(p);
    }
    CodeBlock &add(const CodeBlock &b)
    {
        for (auto p : b.packages)
        {
            packages.push_back(p);
        }
        return *this;
    }
    virtual void enc(std::ostream &oss)
    {
        for (auto pkg : packages)
        {
            for (int i = 0; i < 4; ++i)
            {
                auto value = pkg.ins[i].encode();
                oss.write((char *)(&value), sizeof(uint32_t));
            }
        }
    }
    virtual std::vector<uint32_t> enc()
    {
        std::vector<uint32_t> ret;
        for (auto pkg : packages)
        {
            for (int i = 0; i < 4; ++i)
            {
                ret.push_back(pkg.ins[i].encode());
            }
        }
        return ret;
    }
};

class MOV : public scalar2
{
public:
    MOV()
    {
        opcode = 0x5800000;
    }

    void add_imm_hook(int64_t imm) override
    {
        auto op = imms[0];
        auto rd = operands[0];
        Operand rc = R7;
        if (operands.size() == 2)
        {
            rc = operands[1];
        }
        pre.clear();
        Instruction32 last;
        if (op >= (1 << 16))
        {

            if (op >= (1 << 24))
            {
                pre.push_back({MVI().add_reg(rd).add_imm(op >> 17).add_reg(rc), nop(), nop(), nop()});
                uint32_t lshift = 17;
                if ((op >> 8) & (0x1ff))
                {
                    pre.push_back({LSLi().add_reg(rd).add_reg(rd).add_imm(9).add_reg(rc), nop(), nop(), nop()});
                    pre.push_back({ADDU().add_reg(rd).add_reg(rd).add_imm((op >> 8) & (0x1ff)).add_reg(rc), nop(), nop(), nop()});
                    lshift = 8;
                }
                pre.push_back({LSLi().add_reg(rd).add_reg(rd).add_imm(lshift).add_reg(rc), nop(), nop(), nop()});
                last = LSLi().add_reg(rd).add_reg(rd).add_imm(lshift).add_reg(rc);
                if ((op) & (0xff))
                {

                    pre.push_back({ADDU().add_reg(rd).add_reg(rd).add_imm((op) & (0xff)).add_reg(rc), nop(), nop(), nop()});
                    last = ADDU().add_reg(rd).add_reg(rd).add_imm((op) & (0xff)).add_reg(rc);
                }
            }
            else
            {
                pre = {{MVI().add_reg(rd).add_imm(op >> 9).add_reg(rc), nop(), nop(), nop()},
                       {LSLi().add_reg(rd).add_reg(rd).add_imm(9).add_reg(rc), nop(), nop(), nop()}};
                last = LSLi().add_reg(rd).add_reg(rd).add_imm(9).add_reg(rc);
                if (op & 0x1ff)
                {
                    pre.push_back({ADDU().add_reg(rd).add_reg(rd).add_imm((op) & (0x1ff)).add_reg(rc), nop(), nop(), nop()});
                    last = ADDU().add_reg(rd).add_reg(rd).add_imm((op) & (0x1ff)).add_reg(rc);
                }
            }
            pre.pop_back();
            opcode = last.opcode;
            operands = last.operands;
            imms = last.imms;
            shifts = last.shifts;
            ishifts = last.ishifts;
        }
    }
};

class Call : public CodeBlock, public scalar2
{
    // relative_offset is after shift 4,
    // and the relative_offset is an int32_t, so it could be negative
public:
    Call()
    {
        ishifts = {{0, 0, 0}};
    }
    void enc(std::ostream &oss) override
    {
        code_gen();
        CodeBlock::enc(oss);
    }
    std::vector<uint32_t> enc() override
    {
        code_gen();
        return CodeBlock::enc();
    }

private:
    void code_gen()
    {
        assert(imms.size() == 1);
        int relative_offset = imms[0];
        if ((relative_offset - 2 <= 0x3fff) && (relative_offset - 3 >= -0x3fff))
        {
            // correct the relative offset: add two line before call
            relative_offset -= 0x2;
            // using CBZ to call, cbz has imm15 signed
            uint32_t m_offset = relative_offset & 0x3fff;
            if (relative_offset < 0)
            {
                relative_offset += 1;
                m_offset = relative_offset + 0x7fff;
            }
            packages.clear();
            /*
            GPC R30,R7 & nop & nop & nop
            ADD.U R30,R30, #0x20,R7 & nop & nop & nop
            CBZ R31,%FUNC_LABEL%
            */
            packages = {
                {GPC().add_operand(R30).add_operand(R7),
                 nop(), nop(), nop()},
                {ADDU().add_operand(R30).add_operand(R30).add_imm(0x20).add_operand(R7),
                 nop(), nop(), nop()},
                {CBZ().add_operand(R31).add_imm(m_offset),
                 nop(), nop(), nop()},
            };
        }
        else
        {
            relative_offset = relative_offset << 4;
            bool is_negative = (relative_offset < 0);
            if (is_negative)
            {
                relative_offset = -relative_offset;
            }
            packages.clear();
            packages.push_back({PUSH().add_operand(R0).add_operand(R7), nop(), nop(), nop()}); //10
            if (relative_offset - 0x70 > 0xffffff)                                             //>24bit
            {
                if (is_negative)
                {
                    relative_offset += 0x70;
                }
                else
                {
                    relative_offset -= 0x70;
                }

                //14bit+9bit +9bit
                //r0=rela>>18   get the high 14bit
                packages.push_back({MVI().add_operand(R0).add_imm(relative_offset >> 18).add_operand(R7), nop(), nop(), nop()});
                //r0=(rela>>18)<<9
                packages.push_back({LSLi().add_operand(R0).add_operand(R0).add_imm(9).add_operand(R7), nop(), nop(), nop()});
                //r0= ((rela>>18)<<9)+(rela>>9 & 0x1ff)   //get the middle 9 bit
                packages.push_back({ADDU().add_operand(R0).add_operand(R0).add_imm((relative_offset >> 9) & (0x1ff)).add_operand(R7), nop(), nop(), nop()});
                //r0=(((rela>>18)<<9)+(rela>>9 & 0x1ff))<<9
                packages.push_back({LSLi().add_operand(R0).add_operand(R0).add_imm(9).add_operand(R7), nop(), nop(), nop()});
                //r0=((((rela>>18)<<9)+(rela>>9 & 0x1ff))<<9)+rela&0x1ff
                packages.push_back({ADDU().add_operand(R0).add_operand(R0).add_imm((relative_offset) & (0x1ff)).add_operand(R7), nop(), nop(), nop()});
            }
            else
            {
                if (is_negative)
                {
                    relative_offset += 0x50;
                }
                else
                {
                    relative_offset -= 0x50;
                }
                //15bit+9bit
                packages.push_back({MVI().add_operand(R0).add_imm(relative_offset >> 9).add_operand(R7), nop(), nop(), nop()});
                packages.push_back({LSLi().add_operand(R0).add_operand(R0).add_imm(9).add_operand(R7), nop(), nop(), nop()});
                packages.push_back({ADDU().add_operand(R0).add_operand(R0).add_imm((relative_offset) & (0x1ff)).add_operand(R7), nop(), nop(), nop()});
            }
            packages.push_back({GPC().add_operand(R30).add_operand(R7), nop(), nop(), nop()}); //10

            //R0 is the "relative offset"
            if (is_negative)
            {
                packages.push_back({SUB().add_operand(R30).add_operand(R30).add_operand(R0).add_operand(R7), nop(), nop(), nop()}); //20
            }
            else
            {
                packages.push_back({ADD().add_operand(R30).add_operand(R30).add_operand(R0).add_operand(R7), nop(), nop(), nop()}); //20
            }
            packages.push_back({POP().add_operand(R0).add_operand(R7), nop(), nop(), nop()}); //30
            packages.push_back({BRL().add_operand(R30), nop(), nop(), nop()});                //40
            //50

            //20+10+20
            /*
            Push R1
            MVI R1,#imm15
            LSL R1,R1,#15, R7
            ADDU R1,R1,#imm9, r7
            //if >24 other addu meeded
            GPC R30, R7
            ADD R30,R30,R1, R7
            pop R1
            BLR R30
            */
        }
    }
};

};     // namespace AsmZY
#endif //!BUILDER_LINKER_ZY_H