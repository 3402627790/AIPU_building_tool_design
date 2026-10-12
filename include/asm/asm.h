#pragma once
#include "asmdecode.h"
#include <array>
#include <core/constexpr_utils.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
namespace cexpr
{
using namespace aipubt;

constexpr size_t MAX_PARTS = 5;
constexpr size_t MAX_LENGTH = 10;

constexpr int get_ins_type(const char *inst)
{
    int i = 0;
    size_t len = 0;
    inst = strip(inst);
    while (inst[len] != ',' && inst[len] != ' ')
    {
        ++len;
    }
    for (auto tn : InstructionTypeName)
    {

        if (len == str_len(tn) && compare_strings(inst, tn, len))
        {
            break;
        }
        i++;
    }
    return i;
}

// an asm syntax: add rd,rn,rm
constexpr const char *get_op_str(const char *str, size_t len, size_t index)
{
    auto inst = strip(str);
    inst = inst + len_to_splitter(inst, ' ') + 1;
    for (size_t i = 0; i < index && inst < (str + len); ++i)
    {
        inst = inst + len_to_splitter(inst, ',') + 1;
    }
    inst = strip(inst);
    return inst;
}
constexpr Instruction::Operand::OperandType get_op_type(const char *inst)
{
    using OT = Instruction::Operand::OperandType;
    if (inst[0] == 'R' || inst[0] == 'r')
    {
        return OT::RReg;
    }
    else if (inst[0] == 'T' || inst[0] == 't')
    {
        return OT::TReg;
    }
    else if (inst[0] == 'P' || inst[0] == 'p')
    {
        return OT::PReg;
    }
    else if (inst[0] == 'F' || inst[0] == 'f')
    {
        return OT::FReg;
    }
    return OT::Imm;
}

constexpr int get_op_value(const char *inst, Instruction::Operand::OperandType t)
{
    if (t != Instruction::Operand::OperandType::Imm && t != Instruction::Operand::OperandType::UImm)
    {
        inst++;
    }
    auto len = len_to_splitter(inst, ',');
    return str_to_int(inst, len);
}

} //namespace cexpr

namespace aipubt
{
inline bool operator==(const uint32_t a, const InstructionType &b)
{
    return a == (uint32_t)b;
}
inline bool operator==(const InstructionType &b, uint32_t a)
{
    return a == (uint32_t)b;
}
inline bool operator!=(uint32_t a, const InstructionType &b)
{
    return a != (uint32_t)b;
}
inline bool operator!=(const InstructionType &b, uint32_t a)
{
    return a != (uint32_t)b;
}

inline Instruction operator"" _asm(const char *ins, size_t len)
{
    using namespace cexpr;
    auto i = get_ins_type(ins);
    auto ret = Instruction(i, 0);
    for (auto i = 0; i < 4; ++i)
    {
        auto opstr = get_op_str(ins, len, i);
        if (opstr[0] && opstr < (ins + len))
        {
            auto optype = get_op_type(opstr);
            auto v = get_op_value(opstr, optype);
            ret.operands.push_back(Instruction::Operand(optype, v));
        }
    }
    return ret;
}
} //namespace aipubt