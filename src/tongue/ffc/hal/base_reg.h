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
 * @Date: 2024-11-20 10:57:17 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-11-22 12:03:50
 */
#pragma once
#include "desc.h"
#include <algorithm>
#include <cinttypes>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tongue/symbolic.h>
#include <vector>
using namespace tongue;

namespace ffc
{
namespace HAL
{
    struct RegisterValue
    {
        RegisterValue() = default;
        template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
        RegisterValue(T v) : value((uint32_t)v) {}
        RegisterValue(Expr v) : expr(v) {}

        RegisterValue &operator=(const RegisterValue &other)
        {
            value = other.value;
            expr = other.expr;
            return *this;
        }

        RegisterValue mulh(const RegisterValue &v)
        {
            if (expr && v.expr)
            {
                return expr.mulh(v.expr);
            }
            else if (expr)
            {
                return expr.mulh(v.value);
            }
            else if (v.expr)
            {
                return Expr(value).mulh(v.expr);
            }
            uint64_t res = value * v.value;
            return (res >> 32) & 0xffffffff;
        }

        RegisterValue floordiv(const RegisterValue &v)
        {
            if (expr && v.expr)
            {
                return expr.floordiv(v.expr);
            }
            else if (expr)
            {
                return expr.floordiv(v.value);
            }
            else if (v.expr)
            {
                return Expr(value).floordiv(v.expr);
            }
            return value / v.value;
        }

        bool is_constant() const
        {
            if (expr)
            {
                return expr->is_constant();
            }
            return true;
        }

        std::string to_string() const
        {
            if (expr)
            {
                return expr->to_string();
            }
            return std::to_string(value);
        }

        operator uint32_t() const
        {
            if (expr && expr->is_constant())
            {
                return expr->value();
            }
            return value;
        }

        operator Expr() const
        {
            if (expr)
            {
                return expr;
            }
            return Expr(value);
        }

        uint32_t value = 0;
        Expr expr;
    };

    inline std::ostream &operator<<(std::ostream &oss, const RegisterValue &r)
    {
        if (r.expr)
        {
            oss << r.expr->to_string();
        }
        else
        {
            oss << r.value;
        }
        return oss;
    }

#define DECLARE_REGVALUE_OPERATOR(mop)                                                           \
    inline RegisterValue operator mop(const RegisterValue &r, const RegisterValue &v)            \
    {                                                                                            \
        if (r.expr && v.expr)                                                                    \
        {                                                                                        \
            return r.expr mop v.expr;                                                            \
        }                                                                                        \
        else if (r.expr)                                                                         \
        {                                                                                        \
            return r.expr mop v.value;                                                           \
        }                                                                                        \
        else if (v.expr)                                                                         \
        {                                                                                        \
            return r.value mop v.expr;                                                           \
        }                                                                                        \
        return r.value mop v.value;                                                              \
    }                                                                                            \
    template <typename T, typename std::enable_if<std::is_integral<T>::value>::type * = nullptr> \
    inline RegisterValue operator mop(const RegisterValue &r, T v)                               \
    {                                                                                            \
        return r mop RegisterValue((uint32_t)v);                                                 \
    }                                                                                            \
    template <typename T, typename std::enable_if<std::is_integral<T>::value>::type * = nullptr> \
    inline RegisterValue operator mop(T v, const RegisterValue &r)                               \
    {                                                                                            \
        return RegisterValue((uint32_t)v) mop r;                                                 \
    }                                                                                            \
    inline RegisterValue operator mop(const RegisterValue &r, Expr v)                            \
    {                                                                                            \
        return r mop RegisterValue(v);                                                           \
    }                                                                                            \
    inline RegisterValue operator mop(Expr v, const RegisterValue &r)                            \
    {                                                                                            \
        return RegisterValue(v) mop r;                                                           \
    }

    DECLARE_REGVALUE_OPERATOR(+);
    DECLARE_REGVALUE_OPERATOR(-);
    DECLARE_REGVALUE_OPERATOR(*);
    DECLARE_REGVALUE_OPERATOR(/);
    DECLARE_REGVALUE_OPERATOR(%);
    DECLARE_REGVALUE_OPERATOR(<<);
    DECLARE_REGVALUE_OPERATOR(>>);
    DECLARE_REGVALUE_OPERATOR(&);
    DECLARE_REGVALUE_OPERATOR(|);
    DECLARE_REGVALUE_OPERATOR(^);

    // Class to represent a register
    // a special key for address/buffer/tensor value: _tensor
    // if the field is _tensor, then other fields will be ignored
    // and the value is the tensor's id
    // the tensor id only valid when the id < 0x7fffffff, i.e. negative value is invalid
    constexpr const char *DefaultTensorKey = "_tensor";
    class Register
    {
    public:
        RegisterValue &operator[](const char *key)
        {
            return (*this)[std::string(key)];
        }
        RegisterValue operator[](const char *key) const
        {
            return (*this)[std::string(key)];
        }

        RegisterValue &operator[](const std::string &key)
        {
            if (key == DefaultTensorKey)
            {
                if (m_fields.count(DefaultTensorKey))
                {
                    return m_fields[DefaultTensorKey];
                }
                m_fields[DefaultTensorKey] = 0xffffffff; // default value for tensor id
                return m_fields[DefaultTensorKey];
            }
            auto it = m_fmt.find(key);
            if (it != m_fmt.end())
            {
                if (!m_fields.count(key))
                {
                    auto fmt = it->second;
                    auto shift = fmt & 0xff;
                    auto width = fmt >> 16u;
                    auto mask = ((1ul << width) - 1u);
                    m_fields[key] = (m_init_val >> shift) & mask;
                }
                return m_fields[key];
            }
            static RegisterValue wrong;
            return wrong;
        }

        RegisterValue operator[](const std::string &key) const
        {
            if (key == DefaultTensorKey)
            {
                if (m_fields.count(DefaultTensorKey))
                {
                    return m_fields.at(DefaultTensorKey);
                }
                return -1; // default value for tensor id
            }
            auto it = m_fmt.find(key);
            if (it != m_fmt.end())
            {
                if (!m_fields.count(key))
                {
                    auto fmt = it->second;
                    auto shift = fmt & 0xff;
                    auto width = fmt >> 16u;
                    auto mask = ((1ul << width) - 1u);
                    return (m_init_val >> shift) & mask;
                }
                return m_fields.at(key);
            }
            return 0;
        }

        Register(uint32_t init_val = 0,
                 uint32_t address = 0,
                 const std::map<std::string, uint32_t> &fmt = std::map<std::string, uint32_t>(),
                 uint32_t type = 0) : m_init_val(init_val),
                                      m_address(address),
                                      m_type(type),
                                      m_fmt(fmt)
        {
        }

        std::string to_string(uint32_t indent = 4) const
        {
            std::string ind(indent, ' ');
            std::ostringstream oss;
            if (m_fields.count(DefaultTensorKey))
            {
                oss << ind << "Tensor id:\t" << (*this)[DefaultTensorKey] << "[31:0]\n";
                return oss.str();
            }
            std::vector<std::pair<uint32_t, std::string>> shifted_fields;

            for (auto &kv : m_fmt)
            {
                auto shift = kv.second & 0xff;
                shifted_fields.emplace_back(shift, kv.first);
            }

            std::sort(shifted_fields.begin(), shifted_fields.end(), [](const auto &a, const auto &b) {
                return a.first > b.first;
            });

            for (const auto &sf : shifted_fields)
            {
                auto shift = sf.first;
                auto field_name = sf.second;
                auto fmt = m_fmt.at(field_name);
                auto width = fmt >> 16u;
                oss << ind << std::setw(22) << field_name << ":\t0x" << std::hex << (*this)[field_name]
                    << "[" << std::dec << shift + width - 1 << ":" << shift << "]\n";
            }

            return oss.str();
        }

        operator RegisterValue() const
        {
            return val();
        }

        Register &operator=(const RegisterValue &v)
        {
            for (auto &kv : m_fmt)
            {
                auto shift = kv.second & 0xff;
                auto width = kv.second >> 16u;
                auto mask = (1ul << width) - 1u;
                m_fields[kv.first] = (v >> shift) & mask;
            }
            return *this;
        }
        template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
        Register &operator=(T v)
        {
            *this = RegisterValue(v);
            return *this;
        }
        Register &operator=(Expr v)
        {
            *this = RegisterValue(v);
            return *this;
        }

        RegisterValue val() const
        {
            if (m_fields.count(DefaultTensorKey))
            {
                return m_fields.at(DefaultTensorKey);
            }
            RegisterValue ret = m_init_val;
            for (auto &kv : m_fmt)
            {
                if (m_fields.count(kv.first))
                {
                    auto shift = kv.second & 0xff;
                    auto width = kv.second >> 16u;
                    auto max = (1ul << width) - 1u;
                    auto mask = ~(max << shift);
                    auto value = m_fields.at(kv.first);
                    ret = (ret & mask) | (value << shift);
                }
            }
            return ret;
        }

        std::vector<RegisterValue> get_field_values_ordered_by_bit_field() const
        {
            std::vector<std::pair<int, RegisterValue>> fields_with_shifts;

            for (auto &kv : m_fmt)
            {
                auto shift = kv.second & 0xff;
                fields_with_shifts.emplace_back(shift, (*this)[kv.first]);
            }

            std::sort(fields_with_shifts.begin(), fields_with_shifts.end(), [](const auto &a, const auto &b) {
                return a.first < b.first;
            });

            std::vector<RegisterValue> ordered_field_values;
            for (auto &field : fields_with_shifts)
            {
                ordered_field_values.push_back(field.second);
            }

            return ordered_field_values;
        }

        uint32_t address() const
        {
            return m_address;
        }

        const std::map<std::string, uint32_t> &format() const
        {
            return m_fmt;
        }

        uint32_t type() const
        {
            return m_type;
        }

    protected:
        uint32_t m_init_val;                           // Initial value of the register
        uint32_t m_address;                            // Address of the register
        uint32_t m_type;                               // Type of the register (0 for act, 1 for param, 2 for ctrl)
        std::map<std::string, RegisterValue> m_fields; // Fields of the register
        std::map<std::string, uint32_t> m_fmt;         // Format of the register fields
    };

    // Class to represent a group of registers
    class AIFFGroupRegs
    {
    public:
        // Overload operator[] to access registers by key
        Register &operator[](const char *key)
        {
            return (*this)[std::string(key)];
        }
        Register &operator[](const std::string &key)
        {
            if (m_regs.count(key))
                return m_regs[key];
            static Register wrong;
            return wrong;
        }

        AIFFGroupRegs(const std::map<std::string, Register> &regs = {}) : m_regs(regs)
        {
        }

        std::string to_string(uint32_t indent = 4) const
        {
            std::string ind;
            for (auto i = 0u; i < indent; ++i)
            {
                ind += " ";
            }
            std::ostringstream oss;
            for (auto kv : order_by_address())
            {
                oss << ind << kv.first
                    << "\tvalue = 0x" << std::hex << kv.second.val()
                    << "\taddress = " << kv.second.address() << "\n";
                oss << kv.second.to_string(indent + 4);
            }
            return oss.str();
        }
        // Get the map of registers
        const std::map<std::string, Register> &get_map() const
        {
            return m_regs;
        }
        // Order registers by address
        std::vector<std::pair<std::string, Register>> order_by_address() const
        {
            std::vector<std::pair<std::string, Register>> sorted_regs(m_regs.begin(), m_regs.end());

            std::sort(sorted_regs.begin(), sorted_regs.end(), [](const std::pair<std::string, Register> &a, const std::pair<std::string, Register> &b) {
                return a.second.address() < b.second.address();
            });

            return sorted_regs;
        }
        using iterator = std::map<std::string, Register>::iterator;
        iterator begin()
        {
            return m_regs.begin();
        }
        iterator end()
        {
            return m_regs.end();
        }

        std::vector<std::vector<Register>> grouped_by_address() const
        {
            std::vector<std::vector<Register>> grouped;
            std::vector<Register> current_group;
            for (auto &kv : order_by_address())
            {
                auto &&reg = kv.second;
                if (current_group.empty() || reg.address() == current_group.back().address() + 1)
                {
                    current_group.push_back(reg);
                }
                else
                {
                    if (!current_group.empty())
                    {
                        grouped.emplace_back(current_group);
                        current_group.clear();
                        current_group.push_back(reg);
                    }
                }
            }
            if (!current_group.empty())
            {
                grouped.emplace_back(current_group);
            }
            return grouped;
        }

    protected:
        std::map<std::string, Register> m_regs; // Map of registers
    };

    class AIFFRegs
    {
    public:
        using iterator = std::map<std::string, AIFFGroupRegs>::iterator;
        // Constructor from a map<string,AIFFGroupRegs>
        AIFFRegs(const std::map<std::string, AIFFGroupRegs> &groups = {}) : m_regs(groups)
        {
        }
        // Overload operator[] to access groups of registers by key
        AIFFGroupRegs &operator[](const char *key)
        {
            return (*this)[std::string(key)];
        }
        AIFFGroupRegs &operator[](const std::string &key)
        {
            if (m_regs.count(key))
                return m_regs[key];
            static AIFFGroupRegs wrong;
            return wrong;
        }

        const AIFFGroupRegs &operator[](const char *key) const
        {
            return (*this)[std::string(key)];
        }
        const AIFFGroupRegs &operator[](const std::string &key) const
        {
            if (m_regs.count(key))
                return m_regs.at(key);
            static AIFFGroupRegs wrong;
            return wrong;
        }
        std::string to_string(uint32_t indent = 4) const
        {
            std::string ind(indent, ' ');
            std::ostringstream oss;
            for (const auto &kv : m_regs)
            {
                oss << ind << kv.first << "\n";
                oss << kv.second.to_string(indent + 4);
            }
            return oss.str();
        }

        // Add a group of registers to the map
        void add_group(const std::string &key, const AIFFGroupRegs &group)
        {
            m_regs[key] = group;
        }

        const auto &get_map() const
        {
            return m_regs;
        }
        iterator begin()
        {
            return m_regs.begin();
        }
        iterator end()
        {
            return m_regs.end();
        }

        std::string get_modify_desc_func(const std::vector<RegisterValue> &regs,
                                         const std::string &func_name)
        {
            std::vector<std::pair<Expr, std::vector<int>>> exprs;
            for (size_t i = 0; i < regs.size(); ++i)
            {
                auto expr = regs[i].expr;
                if (expr && !expr->is_constant())
                {
                    exprs.push_back({expr, {(int)i}});
                }
            }
            if (exprs.empty())
            {
                return std::string();
            }

            if (exprs.size() >= 2)
            {
                std::set<int> ran_idx;
                std::vector<std::pair<Expr, std::vector<int>>> new_exprs;
                for (size_t i = 0; i < exprs.size(); ++i)
                {
                    if (ran_idx.count(i))
                    {
                        continue;
                    }
                    auto p0 = exprs[i];
                    for (size_t j = i + 1; j < exprs.size(); ++j)
                    {
                        if (ran_idx.count(j))
                        {
                            continue;
                        }
                        auto &p1 = exprs[j];
                        if (p0.first->equal(p1.first))
                        {
                            p0.second.insert(p0.second.end(), p1.second.begin(), p1.second.end());
                            ran_idx.insert(j);
                        }
                    }
                    new_exprs.push_back(p0);
                    ran_idx.insert(i);
                }
                if (new_exprs.size() < exprs.size())
                {
                    exprs = std::move(new_exprs);
                }
            }

            std::set<Symbol> arg_set;
            for (auto &p : exprs)
            {
                auto vars = p.first->variables();
                for (auto &&var : vars)
                {
                    arg_set.insert(var);
                }
            }
            std::vector<Symbol> args{arg_set.begin(), arg_set.end()};

            std::ostringstream oss;
            oss << "// autogenerated DO NOT modify!!!\n";
            auto func = "void " + func_name + "(";
            oss << func << "__local uint *desc,\n";
            auto tab = std::string(func.size(), ' ');
            for (size_t i = 0; i < args.size(); ++i)
            {
                oss << tab << "uint " << args[i].name();
                if (i != args.size() - 1)
                {
                    oss << ",\n";
                }
            }
            oss << ")\n"
                << "{\n";

            tab = std::string(4, ' ');
            std::map<int, std::string> func_args;
            std::map<ExprPtr, std::string> func_vars;
            for (size_t i = 0; i < exprs.size(); ++i)
            {
                auto expr = exprs[i].first;
                auto &idxs = exprs[i].second;

                auto vars = expr->variables();
                for (auto &&var : vars)
                {
                    func_args[var.id()] = var.name();
                }
                std::string res;
                auto codes = expr_to_cl_code(expr, func_args, func_vars, res);
                oss << tab << "// " << expr->to_string() << "\n";
                for (auto &&c : codes)
                {
                    oss << tab << c << "\n";
                }
                for (auto &&idx : idxs)
                {
                    oss << tab << "desc[" << idx << "] = " << res << ";\n";
                }
                if (i != exprs.size() - 1)
                {
                    oss << "\n";
                }
            }
            oss << "}\n";
            return oss.str();
        }

        std::vector<RegisterValue> get_ctrl_regs()
        {
            static const std::set<std::string> excluded = {
                "ITP2",
                "ITP3",
                "AUC",
            };
            std::vector<RegisterValue> ret(8);
            ret[0] = 0x81;
            for (auto &kv : m_regs)
            {
                if (excluded.count(kv.first))
                {
                    continue;
                }
                auto groups = kv.second.grouped_by_address();
                for (auto &g : groups)
                {
                    if (g.size() && g[0].type() == 2)
                    {
                        constexpr uint32_t base_align = 8;
                        uint32_t base = g[0].address();
                        uint32_t len = g.size();
                        uint32_t front_pad_len = 0;
                        if (base % base_align != 0) // base addr alignment
                        {
                            front_pad_len = base % base_align;
                            auto new_base = base / base_align * base_align;
                            auto new_len = len + front_pad_len;
                            // std::cout << "base: " << std::hex << "0x" << base << ", len: " << len
                            //           << ", new_base: " << std::hex << "0x" << new_base << ", new_len: " << new_len
                            //           << ", front_pad_len: " << front_pad_len << std::endl;
                            base = new_base;
                            len = new_len;
                        }
                        ret.push_back(len << 16 | base);
                        for (uint32_t i = 0; i < front_pad_len; ++i)
                        {
                            ret.push_back(0);
                        }
                        for (auto &r : g)
                        {
                            ret.push_back(r.val());
                        }
                        while (ret.size() % 8 != 0)
                        {
                            ret.push_back(0);
                        }
                    }
                }
            }
            ret[3] = (ret.size() / 8);
            return ret;
        }

        std::vector<RegisterValue> get_act_regs()
        {
            std::vector<RegisterValue> ret(72, 0);
            static const std::vector<std::string> units = {
                "MTP0",
                "MTP1",
                "ITPS",
                "ITP0",
                "ITP1",
                "ITP2",
                "ITP3",
                "PTP",
                "WRB",
            };
            constexpr size_t size_per_unit = 8;
            for (size_t i = 0; i < units.size(); ++i)
            {
                auto regs = m_regs.at(units[i]).order_by_address();
                size_t idx = 0;
                for (size_t j = 0; j < regs.size(); ++j)
                {
                    auto &r = regs[j].second;
                    if (r.type() == 0)
                    {
                        ret[i * size_per_unit + idx] = r.val();
                        idx++;
                    }
                }
            }
            return ret;
        }

        std::vector<RegisterValue> get_param_regs()
        {
            std::vector<RegisterValue> ret(88, 0);
            static const std::vector<std::string> units = {
                "MTP0",
                "MTP1",
                "ITP0",
                "ITP1",
                "ITP2",
                "ITP3",
                "ITPS",
                "PTP",
            };
            std::vector<size_t> unit_sizes = {8, 8, 16, 16, 16, 14, 2, 8};
            size_t offset = 0;
            for (size_t i = 0; i < units.size(); ++i)
            {
                auto regs = m_regs.at(units[i]).order_by_address();
                size_t idx = 0;
                for (size_t j = 0; j < regs.size(); ++j)
                {
                    auto &r = regs[j].second;
                    if (r.type() == 1)
                    {
                        ret[offset + idx] = r.val();
                        idx++;
                    }
                }
                offset += unit_sizes[i];
            }
            return ret;
        }

    private:
        std::map<std::string, AIFFGroupRegs> m_regs; // Map of groups of registers
    };

} //namespace HAL
} //namespace ffc
