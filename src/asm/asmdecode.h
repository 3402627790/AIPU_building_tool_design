#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <core/fp16.h>

namespace aipubt
{
#include "asmopcode.h"
#define DMA_TYPE 0x10000
#define DMA_OPCODE 0x10000
#define AIFF_TYPE 0x20000
#define AIFF_OPCODE 0x20000

struct Instruction
{

    struct Format // a format is 64bit
    {
        struct Operand
        {
            uint8_t target : 2; // is destination, if the type is imm, target means signed of imm
            uint8_t type : 4;   // 0 R reg,1 Freg, 2 T reg,3 preg 4:stride 5: imm, 6:imm_split
            uint8_t bits : 5;   // number of bits
            // if imm_split, we need to check imm table, which store the shift2
            // shift: lower 5bit is the lower part shift, high 3bit is the lower part size
            uint8_t shift : 5;
            Operand(uint16_t c = 0)
            {
                target = c >> 14;
                type = (c >> 10) & 0xf;
                bits = (c >> 5) & 0x1f;
                shift = c & 0x1f;
            }
        };
        Operand ops[4];
        Format(uint64_t code)
        {
            uint16_t *c = (uint16_t *)(&code);
            ops[0] = Operand(c[0]);
            ops[1] = Operand(c[1]);
            ops[2] = Operand(c[2]);
            ops[3] = Operand(c[3]);
        }
    };

    struct Operand
    {
        enum OperandType // must be same as gen_opcode.py
        {
            None = -1,
            RReg = 0,
            FReg = 1,
            TReg = 2,
            PReg = 3,
            StReg = 4,
            UImm = 5,
            Imm = 6,
        };
        OperandType type = OperandType::None;
        bool is_target = false;
        uint32_t value = 0;
        std::string label;
        Operand() = default;
        Operand(const Operand &other)
        {
            type = other.type;
            is_target = other.is_target;
            value = other.value;
            label = other.label;
        }
        Operand(OperandType t, uint32_t v, bool target = false) : type(t), is_target(target), value(v) {}
        Operand(const std::string &l) : type(Imm), is_target(false), value(0), label(l) {}
        bool is_none() const
        {
            return type == OperandType::None;
        }
        bool is_rreg() const
        {
            return type == OperandType::RReg;
        }
        bool is_preg() const
        {
            return type == OperandType::PReg;
        }
        bool is_treg() const
        {
            return type == OperandType::TReg;
        }
        bool is_freg() const
        {
            return type == OperandType::FReg;
        }
        bool is_imm() const
        {
            return (type == OperandType::UImm) | (type == OperandType::Imm);
        }
        bool is_simm() const
        {
            return type == OperandType::Imm;
        }
        bool is_label() const
        {
            return is_simm() && !label.empty();
        }
        int32_t imm() const
        {
            if (is_imm())
                return *(int32_t *)(&value); // imm should not be 32bit
            return 0;
        }
        operator bool() const
        {
            return !is_none();
        }
        bool operator==(const Operand &other) const
        {
            return type == other.type && value == other.value;
        }
        bool operator<(const Operand &other) const
        {
            if (type == other.type)
            {
                return value < other.value;
            }
            return type < other.type;
        }
        bool operator!=(const Operand &other) const
        {
            return !(*this == other);
        }
        std::string to_string() const
        {
            static const std::vector<std::string> prefix = {"r", "f", "t", "p", "st", "", ""};
            if (is_simm())
            {
                if (!label.empty())
                {
                    return label;
                }
                return std::to_string(imm());
            }
            return prefix[(int)(type)] + std::to_string(value);
        }
    };

    uint64_t encoding = 0;
    uint32_t type = 0;
    uint8_t slots = 0;
    std::vector<Operand> operands;
    void _init()
    {
        if (type == UINT32_MAX)
        {
            return;
        }
        uint32_t length = sizeof(VLIWSlots) / sizeof(VLIWSlots[0]);
        if (type < length)
        {
            slots = VLIWSlots[type];
        }
        if (is_dma() || is_aiff())
        {
            return _init64();
        }
        if ((encoding & BaseMask[type]) != BaseCode[type])
        {
            return;
        }
        Format fmt(FormatCoding[type]);
        uint32_t imm = 0;
        int imm_bits = 0;
        for (auto op : fmt.ops)
        {
            if (op.bits != 0)
            {
                Operand::OperandType opt;
                uint32_t v = (encoding >> op.shift) & ((1 << op.bits) - 1);
                if (op.type > 4)
                {
                    opt = Operand::OperandType::UImm;
                    if (op.target)
                    {
                        opt = Operand::OperandType::Imm;
                    }
                    if (op.type == 6) // split imm
                    {
                        if (imm_bits)
                        {
                            v = imm << op.bits | v;
                        }
                        else
                        {
                            imm_bits = op.bits;
                            continue;
                        }
                    }
                    imm_bits += op.bits;
                }
                else
                {
                    opt = (Operand::OperandType)op.type;
                }
                if (opt == Operand::OperandType::Imm) // convert into signed
                {
                    if (v & (1 << (imm_bits - 1))) // is negative
                    {
                        v = v | ((0xffffffff >> imm_bits) << imm_bits);
                    }
                }
                operands.emplace_back(opt, v, (bool)op.target);
            }
        }
    }

    void _init64()
    {
        /**
         * operand   : bits
         * rd        : 0-4
         * rn        : 5-9
         * rm        : 10-14
         * despctrl  : 27-30  // zero for dma
         * rsync     : 32-36
         * opmode    : 42-47
         * ---------------------
         * -----dma only--------
         * size      : 15-17  // byte/half/word
         * direction : 18
         * ext_type  : 19-22 //asid
         * int_type  : 23-26 //asid  or type of memory
         *
         * Syntax:
         * dma <op_mode_uimm6>, <rsync>, <desp_ctrl_uimm4>, <int_type_uimm4>, <ext_type_uimm4>, <dir_uimm1>,<size_uimm3>,<rm>, <rn>, <rd>
         * aiff <op_mode_uimm6>, <rsync>, <desp_ctrl_imm4>, <rm>, <rn>, <rd>
         *
         */
        Operand rd(Operand::RReg, encoding & 0x1f);
        Operand rn(Operand::RReg, (encoding >> 5) & 0x1f);
        Operand rm(Operand::RReg, (encoding >> 10) & 0x1f);
        Operand despctrl(Operand::UImm, (encoding >> 27) & 0xf);
        Operand rsync(Operand::UImm, (encoding >> 32) & 0x1f);
        Operand opmode(Operand::UImm, (encoding >> 42) & 0x3f);
        operands.push_back(opmode);
        operands.push_back(rsync);
        operands.push_back(despctrl);
        if (is_dma())
        {
            Operand size_(Operand::UImm, (encoding >> 15) & 0x7);
            Operand direction(Operand::UImm, (encoding >> 18) & 0x1);
            Operand ext_type(Operand::UImm, (encoding >> 19) & 0xf);
            Operand int_type(Operand::UImm, (encoding >> 23) & 0xf);
            operands.push_back(int_type);
            operands.push_back(ext_type);
            operands.push_back(direction);
            operands.push_back(size_);
        }
        operands.push_back(rm);
        operands.push_back(rn);
        operands.push_back(rd);
    }
    uint32_t get_type_with_code(uint32_t code)
    {
        if ((code & 0x3fff0000) == DMA_OPCODE)
        {
            return DMA_TYPE;
        }
        else if ((code & 0x3fff0000) == AIFF_OPCODE)
        {
            return AIFF_OPCODE;
        }
        size_t length = sizeof(BaseCode) / sizeof(BaseCode[0]);
        for (size_t i = 0; i < length; ++i)
        {
            if ((code & BaseMask[i]) == BaseCode[i])
            {
                return i;
            }
        }
        return UINT32_MAX;
    }
    static inline bool is_64bit_ins(uint64_t code)
    {
        uint32_t opcode = (code >> 32) & 0x3fff0000;
        return (opcode == DMA_OPCODE || opcode == AIFF_OPCODE);
    }
    Instruction() = default;
    Instruction(uint32_t t, uint64_t code) : encoding(code), type(t)
    {
        _init();
    }
    Instruction(uint32_t code) : encoding(code)
    {
        type = get_type_with_code(code);
        _init();
    }

    Instruction(uint64_t code) : encoding(code)
    {
        uint32_t high = code >> 32;

        uint32_t opcode = high & 0x3fff0000;
        if (opcode == DMA_OPCODE)
        {
            type = DMA_TYPE;
        }
        else if (opcode == AIFF_OPCODE)
        {
            type = AIFF_OPCODE;
        }
        else if (high == 0) // 32bit ins
        {
            type = get_type_with_code(code);
        }

        _init();
    }

    bool is_jmp() const
    {
        static const std::set<uint32_t> jmp_ins = {
            (uint32_t)InstructionType::B,
            (uint32_t)InstructionType::BL,
            (uint32_t)InstructionType::BR,
            (uint32_t)InstructionType::BRL,
            (uint32_t)InstructionType::BRR,
            (uint32_t)InstructionType::BRRL,
            (uint32_t)InstructionType::LOOP_REG,
            (uint32_t)InstructionType::LOOP_IMM,
            (uint32_t)InstructionType::LOOPEND,
            (uint32_t)InstructionType::CBZ,
            (uint32_t)InstructionType::CBNZ,
        };
        return jmp_ins.count(type);
    }
    bool is_barrier() const
    {
        return type == (uint32_t)InstructionType::WFE || type == (uint32_t)InstructionType::EXIT || type == (uint32_t)InstructionType::IRTN || is_jmp();
    }
    bool is_nop() const
    {
        return type == (uint32_t)InstructionType::NOP;
    }
    bool is_ret() const
    {
        // br r30;
        return type == (uint32_t)InstructionType::BR && operands[0].value == 30;
    }
    bool is_dma() const
    {
        return type == DMA_TYPE;
    }
    bool is_aiff() const
    {
        return type == AIFF_TYPE;
    }
    bool is_64() const
    {
        return is_dma() || is_aiff();
    }
    bool is_load() const
    {
        static const std::set<uint32_t> load = {
            (uint32_t)InstructionType::LD,
            (uint32_t)InstructionType::LD_HP,
            (uint32_t)InstructionType::LD_SP,
            (uint32_t)InstructionType::LDB_S,
            (uint32_t)InstructionType::LDB_U,
            (uint32_t)InstructionType::LDH_S,
            (uint32_t)InstructionType::LDH_U,
            (uint32_t)InstructionType::TLD_B_PRED,
            (uint32_t)InstructionType::TLD_H_PRED,
            (uint32_t)InstructionType::TLD_W_PRED,
            (uint32_t)InstructionType::TLD_UNPRED,
            (uint32_t)InstructionType::TLDB_D_UNPRED,
            (uint32_t)InstructionType::TLDB_Q_UNPRED,
            (uint32_t)InstructionType::TLDG_REG_B_PRED,
            (uint32_t)InstructionType::TLDG_REG_B_UNPRED,
            (uint32_t)InstructionType::TLDG_REG_D_PRED,
            (uint32_t)InstructionType::TLDG_REG_D_UNPRED,
            (uint32_t)InstructionType::TLDG_REG_H_PRED,
            (uint32_t)InstructionType::TLDG_REG_H_UNPRED,
            (uint32_t)InstructionType::TLDG_REG_Q_PRED,
            (uint32_t)InstructionType::TLDG_REG_Q_UNPRED,
            (uint32_t)InstructionType::TLDG_REG_W_PRED,
            (uint32_t)InstructionType::TLDG_REG_W_UNPRED,
            (uint32_t)InstructionType::TLDS_IMM_B_PRED,
            (uint32_t)InstructionType::TLDS_IMM_B_UNPRED,
            (uint32_t)InstructionType::TLDS_IMM_D_PRED,
            (uint32_t)InstructionType::TLDS_IMM_D_UNPRED,
            (uint32_t)InstructionType::TLDS_IMM_H_PRED,
            (uint32_t)InstructionType::TLDS_IMM_H_UNPRED,
            (uint32_t)InstructionType::TLDS_IMM_Q_PRED,
            (uint32_t)InstructionType::TLDS_IMM_Q_UNPRED,
            (uint32_t)InstructionType::TLDS_IMM_W_PRED,
            (uint32_t)InstructionType::TLDS_IMM_W_UNPRED,
            (uint32_t)InstructionType::TLDS_REG_B_PRED,
            (uint32_t)InstructionType::TLDS_REG_B_UNPRED,
            (uint32_t)InstructionType::TLDS_REG_D_PRED,
            (uint32_t)InstructionType::TLDS_REG_D_UNPRED,
            (uint32_t)InstructionType::TLDS_REG_H_PRED,
            (uint32_t)InstructionType::TLDS_REG_H_UNPRED,
            (uint32_t)InstructionType::TLDS_REG_Q_PRED,
            (uint32_t)InstructionType::TLDS_REG_Q_UNPRED,
            (uint32_t)InstructionType::TLDS_REG_W_PRED,
            (uint32_t)InstructionType::TLDS_REG_W_UNPRED,
            // (uint32_t)InstructionType::MFCTRL0,
            // (uint32_t)InstructionType::MFCTRL1,
            // (uint32_t)InstructionType::MFCTRL2,
        };
        return load.count(type);
    }
    bool is_store() const
    {
        static const std::set<uint32_t> store = {
            (uint32_t)InstructionType::ST,
            (uint32_t)InstructionType::ST_HP,
            (uint32_t)InstructionType::ST_SP,
            (uint32_t)InstructionType::STB,
            (uint32_t)InstructionType::STH,
            (uint32_t)InstructionType::TST_B_PRED,
            (uint32_t)InstructionType::TST_H_PRED,
            (uint32_t)InstructionType::TST_W_PRED,
            (uint32_t)InstructionType::TST_UNPRED,
            (uint32_t)InstructionType::TSTC_REG_B_PRED,
            (uint32_t)InstructionType::TSTC_REG_B_UNPRED,
            (uint32_t)InstructionType::TSTC_REG_D_PRED,
            (uint32_t)InstructionType::TSTC_REG_D_UNPRED,
            (uint32_t)InstructionType::TSTC_REG_H_PRED,
            (uint32_t)InstructionType::TSTC_REG_H_UNPRED,
            (uint32_t)InstructionType::TSTC_REG_Q_PRED,
            (uint32_t)InstructionType::TSTC_REG_Q_UNPRED,
            (uint32_t)InstructionType::TSTC_REG_W_PRED,
            (uint32_t)InstructionType::TSTC_REG_W_UNPRED,
            (uint32_t)InstructionType::TSTS_IMM_B_PRED,
            (uint32_t)InstructionType::TSTS_IMM_B_UNPRED,
            (uint32_t)InstructionType::TSTS_IMM_D_PRED,
            (uint32_t)InstructionType::TSTS_IMM_D_UNPRED,
            (uint32_t)InstructionType::TSTS_IMM_H_PRED,
            (uint32_t)InstructionType::TSTS_IMM_H_UNPRED,
            (uint32_t)InstructionType::TSTS_IMM_Q_PRED,
            (uint32_t)InstructionType::TSTS_IMM_Q_UNPRED,
            (uint32_t)InstructionType::TSTS_IMM_W_PRED,
            (uint32_t)InstructionType::TSTS_IMM_W_UNPRED,
            (uint32_t)InstructionType::TSTS_REG_B_PRED,
            (uint32_t)InstructionType::TSTS_REG_B_UNPRED,
            (uint32_t)InstructionType::TSTS_REG_D_PRED,
            (uint32_t)InstructionType::TSTS_REG_D_UNPRED,
            (uint32_t)InstructionType::TSTS_REG_H_PRED,
            (uint32_t)InstructionType::TSTS_REG_H_UNPRED,
            (uint32_t)InstructionType::TSTS_REG_Q_PRED,
            (uint32_t)InstructionType::TSTS_REG_Q_UNPRED,
            (uint32_t)InstructionType::TSTS_REG_W_PRED,
            (uint32_t)InstructionType::TSTS_REG_W_UNPRED,
            (uint32_t)InstructionType::TLDS_REG_W_UNPRED,
        };
        return store.count(type);
    }
    bool is_lsu() const
    {
        return is_load() || is_store();
    }
    bool depends(const Instruction &other) const
    {
        // 1. memory is  always a barrier
        // 2. write a same target register
        // 3. write/read on same register
        auto t0 = get_target();
        auto t1 = get_target();

        if (t0.type != Operand::OperandType::None && (std::find(other.operands.begin(), other.operands.end(), t0) != other.operands.end()))
        {
            return true;
        }
        if (t1.type != Operand::OperandType::None && std::find(std::begin(operands), std::end(operands), t1) != std::end(operands))
        {
            return true;
        }
        if (is_lsu() || is_jmp() || other.is_lsu() || other.is_jmp())
        {
            return true;
        }
        auto i0 = (InstructionType)type;
        auto i1 = (InstructionType)other.type;
        if (i0 == InstructionType::WFE || i0 == InstructionType::EXIT ||
            i0 == InstructionType::IRTN || i0 == InstructionType::BREAK ||
            i1 == InstructionType::WFE || i1 == InstructionType::EXIT ||
            i1 == InstructionType::IRTN || i1 == InstructionType::BREAK)
        {
            return true;
        }
        // check mftcl/mtctl
        if ((i0 == InstructionType::MFCTRL0 && i1 == InstructionType::MTCTRL0) ||
            (i0 == InstructionType::MFCTRL1 && i1 == InstructionType::MTCTRL1) ||
            (i0 == InstructionType::MFCTRL2 && i1 == InstructionType::MTCTRL2) ||
            (i0 == InstructionType::MTCTRL0 && i1 == InstructionType::MFCTRL0) ||
            (i0 == InstructionType::MTCTRL1 && i1 == InstructionType::MFCTRL1) ||
            (i0 == InstructionType::MTCTRL2 && i1 == InstructionType::MFCTRL2))
        {
            uint32_t addr0 = operands[1].imm();
            uint32_t addr1 = other.operands[1].imm();
            return addr0 == addr1;
        }
        return false;
    }

    uint32_t encode()
    {
        encoding = BaseCode[type];
        Format fmt(FormatCoding[type]);
        int high_bit_idx = -1;
        int idx = 0;
        for (auto i = 0; i < 4; ++i)
        {
            auto f = fmt.ops[i];
            if (f.bits != 0)
            {
                auto op = operands[idx];
                if (f.type == 6) // split imm
                {
                    if (high_bit_idx == -1)
                    {
                        high_bit_idx = i;
                        continue;
                    }
                    else
                    {
                        auto hf = fmt.ops[high_bit_idx];
                        int bits = hf.bits + f.bits;
                        auto v = (op.value & ((1 << bits) - 1));
                        encoding |= (v >> f.bits) << hf.shift;
                        encoding += (v & ((1 << f.bits) - 1)) << f.shift;
                    }
                }
                else
                {
                    encoding |= (op.value & ((1 << f.bits) - 1)) << f.shift;
                }
                idx++;
            }
        }
        return encoding;
    }

    void _setup_target()
    {
        Format fmt(FormatCoding[type]);
        for (auto i = 0u; i < operands.size(); i++)
        {
            operands[i].is_target = fmt.ops[i].target;
        }
    }

    // return the target register, if no target register, will return a None operand
    Operand get_target() const
    {
        for (auto &op : operands)
        {
            if (op.is_target)
            {
                return op;
            }
        }
        return Operand();
    }

    void add_op(const Operand &op)
    {
        operands.push_back(op);
        _setup_target();
    }
    /**
     * @brief encode64 for dma and aiff instruction, for normal instruction, encode64 will return same value as encode, but in 64bit;
     *
     * @return uint64_t
     */
    uint64_t encode64()
    {
        if (is_64())
        {
            if (is_dma())
            {
                encoding = DMA_OPCODE;
            }
            else
            {
                encoding = AIFF_OPCODE;
            }
            auto midx = operands.size();
            auto opmode = operands[0];
            auto rsync = operands[1];
            auto descp = operands[2];
            auto rd = operands[midx - 1];
            auto rn = operands[midx - 2];
            auto rm = operands[midx - 3];
            encoding |= rd.value & 0x1ful;
            encoding |= (rn.value & 0x1ful) << 5;
            encoding |= (rm.value & 0x1ful) << 10;
            encoding |= (descp.imm() & 0xful) << 27;
            encoding |= (rsync.imm() & 0x1ful) << 32;
            encoding |= (opmode.imm() & 0x3ful) << 42;
            if (is_dma())
            {
                encoding |= (operands[3].imm() & 0xful) << 23; // int_type
                encoding |= (operands[4].imm() & 0xful) << 19; // ext_type
                encoding |= (operands[5].imm() & 0x1ul) << 18; // direction
                encoding |= (operands[6].imm() & 0x7ul) << 15; // size
            }
            return encoding;
        }
        return encode();
    }
    // TODO: delete this when postfix table complete
    std::vector<std::string> _get_operand_postfix() const
    {
        switch ((InstructionType)type)
        {
        case InstructionType::MOV_SP_RD_FN:
            return {"", ".fp32"};
        case InstructionType::MOV_SP_FD_RN:
            return {".fp32", ""};
        case InstructionType::MOV_SP_FD_FN:
            return {".fp32", ".fp32"};
        case InstructionType::ST_SP:
        case InstructionType::LD_SP:
            return {".fp32", "", ""};
        case InstructionType::ADD_SP:
        case InstructionType::SUB_SP:
        case InstructionType::MUL_SP:
        case InstructionType::DIV_SP:
            return {".fp32", ".fp32", ".fp32"};
        case InstructionType::MULH_U:
        case InstructionType::DIV_U:
        case InstructionType::MOD_U:
            return {".u", ".u", ".u"};
        case InstructionType::CMP_ULT_REG:
        case InstructionType::CMP_ULE_REG:
        case InstructionType::CMP_UGT_REG:
        case InstructionType::CMP_UGE_REG:
            return {"", ".u", ".u"};
        case InstructionType::CMP_QLT_SP:
        case InstructionType::CMP_QLE_SP:
        case InstructionType::CMP_QGT_SP:
        case InstructionType::CMP_QGE_SP:
        case InstructionType::CMP_QEQ_SP:
        case InstructionType::CMP_QNE_SP:
            return {"", ".fp32", ".fp32"};
        case InstructionType::MAX_U:
        case InstructionType::MIN_U:
            return {".u", ".u", ".u"};
        case InstructionType::MAX_SP:
        case InstructionType::MIN_SP:
            return {".fp32", ".fp32", ".fp32"};
        case InstructionType::CVT_SP_W_UNPRED:
            return {".fp32", ".w"};
        case InstructionType::CVT_W_SP_UNPRED:
        case InstructionType::LOG2_SP:
            return {".w", ".fp32"};
        case InstructionType::RINT_SP:
        case InstructionType::SQRT_SP:
        case InstructionType::RSQRT_SP:
            return {".fp32", ".fp32"};
        default:
            break;
        }
        return std::vector<std::string>();
    }
    std::string disasm() const
    {
        std::ostringstream oss;
        if (is_dma())
        {
            oss << "dma";
        }
        else if (is_aiff())
        {
            oss << "aiff";
        }
        else if (type == UINT32_MAX)
        {
            std::cerr << "unkown instruction: encoding 0x" << std::hex << encoding;
            return "";
        }
        else
        {
            oss << Mnemonic[type];
        }
        if (operands.size())
        {
            oss << " ";
            auto postfix = _get_operand_postfix();
            oss << operands[0].to_string();
            if (postfix.size())
            {
                oss << postfix[0];
            }
            for (auto i = 1u; i < operands.size(); ++i)
            {
                oss << "," << operands[i].to_string();
                if (postfix.size())
                {
                    oss << postfix[i];
                }
            }
        }
        return oss.str();
    }
};

const Instruction Nop = Instruction((uint32_t)InstructionType::NOP, 0);
const Instruction RET = Instruction((uint32_t)InstructionType::BR, 0x140001e);
struct InsPackage
{
    InsPackage() = default;
    InsPackage(const std::vector<Instruction> &pkg)
    {
        int n = pkg.size() <= 4 ? pkg.size() : 4;
        for (int i = 0; i < n; ++i)
        {
            ins[i] = pkg[i];
        }
        for (int i = n; i < 4; ++i)
        {
            ins[i] = Nop;
        }
    }
    InsPackage(const std::initializer_list<Instruction> &&l)
        : InsPackage(std::vector<Instruction>(l)) {}
    InsPackage(const Instruction &c) : InsPackage({c}) {}
    InsPackage(const std::string &l) : label(l) {}
    InsPackage(uint64_t code0, uint64_t code1 = 0)
    {
        if (Instruction::is_64bit_ins(code0))
        {
            ins[0] = Instruction(code0);
            ins[1] = Nop;
        }
        else
        {
            ins[0] = Instruction((uint32_t)(code0 & 0xffffffff));
            ins[1] = Instruction((uint32_t)(code0 >> 32));
        }
        if (code1)
        {
            if (Instruction::is_64bit_ins(code1))
            {
                ins[2] = Instruction(code1);
                ins[3] = Nop;
            }
            else
            {
                uint32_t h0 = code1 & 0xffffffff;
                uint32_t h1 = code1 >> 32;
                ins[2] = Instruction(h0);
                ins[3] = Instruction(h1);
            }
        }
    }

    Instruction ins[4];
    std::string label;

    bool is_label() const
    {
        return !label.empty();
    }
    bool has_jmp() const
    {
        return ins[0].is_jmp() | ins[1].is_jmp() | ins[2].is_jmp() | ins[3].is_jmp();
    }
    bool is_barrier() const
    {
        return ins[0].is_barrier() | ins[1].is_barrier() | ins[2].is_barrier() | ins[3].is_barrier();
    }
    bool is_legal() const
    {
        for (auto i = 0; i < 4; ++i)
        {
            if (!(ins[i].slots & (1 << i)))
            {
                return false;
            }
        }
        return true;
    }

    void _backtrace(int n, std::vector<int> &assignment, std::vector<std::vector<int>> &result, uint8_t &occupied, const std::vector<uint8_t> &mat)
    {
        if (n == 4)
        {
            result.push_back(assignment);
            return;
        }
        auto &slot = mat[n];
        for (auto i = 0; i < 4; ++i)
        {
            if ((~occupied) & slot & (1 << i))
            {
                assignment[n] = i;
                occupied |= (1 << i);
                _backtrace(n + 1, assignment, result, occupied, mat);
                occupied &= ~(1 << i);
            }
        }
    }
    // try to legalize, if success, return true
    bool legalize()
    {
        if (is_legal())
        {
            return true;
        }
        std::vector<int> assignment(4);
        std::vector<std::vector<int>> result;
        uint8_t occupied = 0;
        std::vector<uint8_t> mat = {ins[0].slots, ins[1].slots, ins[2].slots, ins[3].slots};
        _backtrace(0, assignment, result, occupied, mat);
        if (result.empty())
        {
            return false;
        }
        Instruction new_ins[4];
        for (auto i = 0; i < 4; ++i)
        {
            new_ins[result[0][i]] = ins[i];
        }
        ins[0] = new_ins[0];
        ins[1] = new_ins[1];
        ins[2] = new_ins[2];
        ins[3] = new_ins[3];
        return true;
    }
    const std::array<std::vector<int>, 4> get_deps() const
    {
        Instruction::Operand targets[4];
        std::array<std::vector<int>, 4> deps;
        // suppose one instruction only has one target
        for (auto i = 0; i < 4; ++i)
        {
            for (auto o : ins[i].operands)
            {
                if (o.is_target)
                {
                    targets[i] = o;
                    continue;
                }
            }
        }
        for (auto i = 0; i < 4; ++i)
        {
            for (auto o : ins[i].operands)
            {
                for (auto j = 0; j < 4; ++j)
                {
                    if (i != j && o == targets[j])
                    {
                        deps[j].push_back(i);
                    }
                }
            }
        }
        return deps;
    }
    // move jmp to latest and resolve simple conflict
    // if return false, that it is a complex conflict
    bool resolve()
    {
        // suppose one pkg only  has one jmp
        for (auto i = 0; i < 3; ++i)
        {
            if (ins[i].is_jmp())
            {
                std::swap(ins[3], ins[i]);
            }
        }
        auto deps = get_deps();
        if (conflict(deps)) // can resolve
        {
            if (complex_conflict(deps))
            {
                return false;
            }
            std::vector<int> order;
            auto dfs = [&](int j, auto &self) {
                if (order.size() == 4)
                {
                    return;
                }
                if (deps[j].size())
                {
                    for (auto i : deps[j])
                    {
                        self(i, self);
                    }
                    order.push_back(j);
                }
                else if (std::find(order.begin(), order.end(), j) == order.end())
                {
                    order.push_back(j);
                }
            };
            for (auto i = 0; i < 4; ++i)
            {
                dfs(i, dfs);
            }
            Instruction old[4] = {ins[0], ins[1], ins[2], ins[3]};
            for (auto i = 0; i < 4; ++i)
            {
                ins[order[i]] = old[i];
            }
        }
        return true;
    }
    bool conflict(const std::array<std::vector<int>, 4> &deps) const
    {
        for (auto i = 0; i < 4; ++i)
        {
            if (deps[i].size())
            {
                return true;
            }
        }
        return false;
    }
    // instruction interweave with others
    // for example: ld r0, r1,0;  ld r1,r0,0;
    bool complex_conflict(const std::array<std::vector<int>, 4> &deps) const
    {
        for (auto i = 0; i < 4; ++i)
        {
            if (deps[i].size())
            {
                for (auto j : deps[i])
                {
                    for (auto k : deps[j])
                    {
                        if (k == i)
                        {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }
    bool complex_conflict() const
    {
        return complex_conflict(get_deps());
    }
    std::string disasm(bool vliw64 = false) const
    {
        std::ostringstream oss;
        if (is_label())
        {
            oss << label << ":";
        }
        else
        {
            if (ins[0].is_64())
            {
                oss << "{" << ins[0].disasm();
            }
            else
            {
                oss << "{" << ins[0].disasm() << ";"
                    << ins[1].disasm();
            }
            if (!vliw64)
            {
                if (ins[2].is_64())
                {
                    oss << ";" << ins[2].disasm();
                }
                else
                {
                    oss << ";" << ins[2].disasm() << ";"
                        << ins[3].disasm();
                }
            }
            oss << "}";
        }
        return oss.str();
    }
};

struct InsPackages
{
    InsPackages() = default;
    InsPackages(const uint64_t *codes, uint32_t len)
    {
        uint32_t i = 0;
        vliw64 = len & 0x1;
        if (vliw64)
        {
            while (i < len)
            {
                pkgs.emplace_back(codes[i]);
                i += 1;
            }
        }
        else
        {
            while (i < len)
            {
                pkgs.emplace_back(codes[i], codes[i + 1]);
                i += 2;
            }
        }
    }
    InsPackages(const std::vector<InsPackage> &inss) : pkgs(inss) {}

    InsPackages &add(const InsPackage &pkg)
    {
        pkgs.push_back(pkg);
        return *this;
    }

#define FWD_FUNC(funcname)                                               \
    template <typename... T>                                             \
    inline auto funcname(T &&... params)                                 \
    {                                                                    \
        return pkgs.funcname(std::forward<decltype(params)>(params)...); \
    }

    // update label offset for BL/CBZ/CBN
    void update_offsets()
    {
        std::map<std::string, int> labels;
        int index = 0;
        for (auto &&pkg : pkgs)
        {
            if (pkg.is_label())
            {
                labels[pkg.label] = index;
                continue; // skip label
            }
            index += 4;
        }
        index = 0;
        for (auto &pkg : pkgs)
        {
            if (pkg.is_label())
            {
                continue; // skip label
            }
            for (auto &ins : pkg.ins)
            {
                if (ins.is_jmp())
                {
                    for (auto &op : ins.operands)
                    {
                        if (op.is_label() && labels.count(op.label))
                        {
                            int offset = (labels[op.label] - index) / 2;
                            op.value = *(uint32_t *)(&offset);
                        }
                    }
                }
            }
            index += 4;
        }
    }

    void remove_labels()
    {
        std::vector<InsPackage> inss;
        for (auto &pkg : pkgs)
        {
            if (pkg.is_label())
            {
                continue;
            }
            inss.push_back(pkg);
        }
        pkgs = std::move(inss);
    }

    inline InsPackage &operator[](uint32_t i)
    {
        return pkgs[i];
    }
    size_t size() const
    {
        return pkgs.size();
    }

    FWD_FUNC(emplace_back);
    FWD_FUNC(emplace);
    FWD_FUNC(begin);
    FWD_FUNC(end);
    FWD_FUNC(push_back);

    std::string disasm() const
    {
        std::ostringstream oss;
        for (auto &&pkg : pkgs)
        {
            oss << pkg.disasm() << "\n";
        }
        return oss.str();
    }
    std::vector<uint32_t> split_basic_block(std::vector<std::pair<uint32_t, uint32_t>> &loop_pair)
    {
        std::set<uint32_t> ret;
        ret.insert(0);
        uint32_t i = 0;
        int32_t last_loop = -1;
        loop_pair.clear();
        for (auto pkg : pkgs)
        {
            if (pkg.is_barrier())
            {
                ret.insert(i + 1);
                for (auto ins : pkg.ins)
                {
                    if (ins.is_jmp())
                    {
                        InstructionType t = (InstructionType)ins.type;
                        int32_t relative_offset = 0;
                        if ((t == InstructionType::B) | (t == InstructionType::BL))
                        {
                            relative_offset = ins.operands[0].imm() / 2;
                        }
                        else if ((t == InstructionType::CBZ) | (t == InstructionType::CBNZ))
                        {
                            relative_offset = ins.operands[1].imm() / 2;
                        }
                        else if ((t == InstructionType::LOOP_REG) | (t == InstructionType::LOOP_IMM))
                        {
                            if (last_loop >= 0)
                            {
                                // error
                                std::cout << "error nest loop is not acceptable\n";
                            }
                            else
                            {
                                last_loop = i;
                            }
                        }
                        else if (t == InstructionType::LOOPEND)
                        {
                            if (last_loop < 0)
                            {
                                std::cout << "error no loop ins before loopend\n";
                                // error
                            }
                            else
                            {
                                loop_pair.emplace_back(last_loop, i);
                                last_loop = -1;
                            }
                        }
                        if (relative_offset != 0)
                        {
                            if (relative_offset + (int)i < 0)
                            {
                            }
                            else
                            {
                                ret.insert(relative_offset + i);
                            }
                        }
                        break; // only one brach is allow in one pkg
                    }
                }
            }
            i++;
        }
        std::vector<uint32_t> res(ret.begin(), ret.end());

        std::sort(res.begin(), res.end());

        if (res.back() != pkgs.size())
        {
            res.push_back(pkgs.size());
        }

        return res;
    }

    static InsPackages pack_basic_block(InsPackages pkgs)
    {
        // 1. build the dependency graph
        // 2. get all levels
        // 3. pack independent instructions into one package
        std::map<uint32_t, Instruction> ins_map;
        std::map<uint32_t, std::vector<uint32_t>> dep_graph;
        // we use the index of instruction as node id
        std::map<Instruction::Operand, uint32_t> write_reg;
        std::map<Instruction::Operand, std::set<uint32_t>> read_reg;
        std::set<uint32_t> all_ins;
        std::set<uint32_t> viewed;
        std::set<uint32_t> barrier;

        std::vector<std::set<uint32_t>> levels;
        // TODO we need deal with T0/f0, where f0 is alias of T0[0]
        // TODO jmp/wfe  instruction should be a barrier
        for (auto i = 0u; i < pkgs.size(); ++i)
        {
            auto &pkg = pkgs[i];
            pkg.resolve();
            for (auto j = 0; j < 4; ++j)
            {
                auto id = i * 4 + j;
                auto &ins = pkg.ins[j];
                if (ins.is_nop())
                {
                    continue;
                }
                ins_map[id] = ins;
                bool is_barrier = ins.is_barrier();
                if (is_barrier)
                {
                    barrier.insert(id);
                }
                // read before write
                for (auto &op : ins.operands)
                {
                    if (!op.is_imm() && !op.is_target)
                    {
                        if (write_reg.count(op))
                        {
                            dep_graph[id].push_back(write_reg[op]);
                        }
                        read_reg[op].insert(id);
                    }
                }
                auto target = ins.get_target();
                if (target)
                {
                    if (write_reg.count(target))
                    {
                        dep_graph[id].push_back(write_reg[target]);
                    }
                    if (read_reg.count(target))
                    {
                        for (auto ii : read_reg[target])
                        {
                            if (ii != id) // skip self dep
                            {
                                dep_graph[id].push_back(ii);
                            }
                        }
                        read_reg.erase(target);
                    }
                    write_reg[target] = i * 4 + j;
                }
                if (!dep_graph.count(id) && !is_barrier) // barrier should be last level
                {
                    viewed.insert(id);
                }
                else
                {
                    all_ins.insert(id);
                }
            }
        }

        // 2. levels
        if (viewed.size())
        {
            levels.emplace_back(viewed);
        }
        auto s = all_ins.size() + viewed.size();
        std::set<uint32_t> l0;

        while (s > viewed.size())
        {
            for (auto id : all_ins)
            {
                if (barrier.count(id))
                {
                    continue;
                }
                if (!dep_graph.count(id))
                {
                    l0.insert(id);
                    viewed.insert(id);
                }
                else
                {
                    bool ready = true;
                    for (auto d : dep_graph[id])
                    {
                        if (!viewed.count(d) || l0.count(d))
                        {
                            ready = false;
                        }
                    }
                    if (ready)
                    {
                        l0.insert(id);
                        viewed.insert(id);
                    }
                }
            }
            if (l0.size() == 0)
            {
                // FATAL error
            }
            for (auto i : l0)
            {
                all_ins.erase(i);
            }
            if (all_ins.size() == barrier.size())
            {
                l0.insert(barrier.begin(), barrier.end());
                all_ins.clear();
                viewed.insert(barrier.begin(), barrier.end());
            }
            levels.emplace_back(l0);
            l0.clear();
        }

        // 3. package
        InsPackages ret;
        auto get_valid_slots = [](uint8_t slots, uint8_t occupied) {
            std::vector<uint8_t> valid_slots;
            for (auto i = 3; i >= 0; i--)
            {
                if (slots & ~occupied & (1 << i))
                {
                    valid_slots.push_back(i);
                }
            }
            return valid_slots;
        };
        auto pack_free_level = [&](const std::set<uint32_t> &level, bool last_level) {
            std::vector<InsPackage> ret;
            InsPackage vliw;
            uint8_t occupied = 0;
            std::set<uint32_t> skip = level;
            std::set<uint32_t> remain;
            while (skip.size())
            {
                for (auto id : skip)
                {
                    auto ins = ins_map[id];
                    if (ins.is_barrier())
                    {
                        continue;
                    }
                    auto valid = get_valid_slots(ins.slots, occupied);
                    if (valid.empty())
                    {
                        remain.insert(id);
                        continue;
                    }
                    vliw.ins[valid[0]] = ins;
                    occupied |= 1 << valid[0];
                    if (occupied == 0xf)
                    {
                        ret.push_back(vliw);
                        vliw = {};
                        occupied = 0;
                        continue;
                    }
                }
                if (last_level & remain.empty() && barrier.size())
                {
                    for (auto id : barrier)
                    {
                        auto ins = ins_map[id];
                        auto valid = get_valid_slots(ins.slots, occupied);
                        if (valid.empty())
                        {
                            ret.push_back(vliw);
                            vliw = {};
                            occupied = 0;
                            valid = get_valid_slots(ins.slots, occupied);
                        }
                        vliw.ins[valid[0]] = ins;
                        occupied |= 1 << valid[0];
                        if (occupied == 0xf)
                        {
                            ret.push_back(vliw);
                            vliw = {};
                            occupied = 0;
                        }
                    }
                }
                if (occupied != 0)
                {
                    ret.push_back(vliw);
                    vliw = {};
                    occupied = 0;
                }
                skip = remain;
                remain.clear();
            }
            return ret;
        };
        for (auto i = 0u; i < levels.size(); i++)
        {
            auto p = pack_free_level(levels[i], i == levels.size() - 1);
            ret.pkgs.insert(ret.pkgs.end(), p.begin(), p.end());
        }

        // use backtrace to package vliw instructions
        // auto backtrace = [&](auto &&self, InsPackages &result, uint8_t occupied, std::vector<uint32_t> &level, int idx)
        // {
        //     auto n = level.size();
        //     if (idx == n)
        //     {
        //         res.emplace_back(result);
        //         return;
        //     }
        //     auto &ins = ins_map[level[idx]];
        // uint8_t valid = ins.slots & ~occupied;
        // if (valid)
        // {
        //     for (auto i = 0; i < 4; ++i)
        //     {
        //         if ((1 << i) & valid)
        //         {
        //             occupied |= (1 << i);
        //             (*result.end()).ins[i] = ins;
        //         }
        //     }
        // }
        // else
        // {

        // }
        // };
        return ret;
    }

    void vliw_pack() // auto pack to vliw128
    {
        // 1. split basic block
        // 2. basic block pack
        // 3. relocate jmp offset(code size changed)
        std::vector<InsPackages> blocks;
        std::vector<InsPackages> vliw_blocks;
        std::vector<std::pair<uint32_t, uint32_t>> loop_pair;
        auto block_offset = split_basic_block(loop_pair);
        for (auto i = 0u; i < block_offset.size() - 1; ++i)
        {
            if (block_offset[i] != block_offset[i + 1] && block_offset[i] < pkgs.size() && block_offset[i + 1] <= pkgs.size())
            {
                // blocks.emplace_back(pkgs.begin() + block_offset[i], pkgs.begin() + block_offset[i + 1]);
            }
        }

        for (auto &block : blocks)
        {
            vliw_blocks.emplace_back(pack_basic_block(block));
        }

        for (auto pkg : pkgs)
        {
            // TODO ~~auto pack to vliw128
        }
    }

    std::vector<InsPackage> pkgs;
    bool vliw64 = false;
};

// std::string decode(uint32_t code)
// {
//     size_t length = sizeof(BaseCode) / sizeof(BaseCode[0]);
//     for (size_t i = 0; i < length; ++i)
//     {
//         if ((code & BaseMask[i]) == BaseCode[i])
//         {
//             Instruction ins(i, code);
//             return ins.disasm();
//         }
//     }
//     return "UNKNOWN";
// }
} // namespace aipubt