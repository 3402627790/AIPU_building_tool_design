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
 * @Date: 2023-03-21 11:44:04 
 * @Last Modified by: Lang Hu 
 * @Last Modified time: 2023-07-17 10:25:13 
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace tongue
{

namespace sbe
{
    class Function;
};
class ExprImpl;
class Expr;
using ExprPtr = std::shared_ptr<ExprImpl>;
using ExprRef = std::weak_ptr<ExprImpl>;

class Symbol
{
public:
    using Value = double;
    Symbol(const std::string &name = "x") : m_name(name)
    {
        m_id = get_symbol_id();
    }
    Symbol(Value v)
    {
        m_id = get_symbol_id();
        m_name = "_symbol_" + std::to_string(m_id);
        m_value = v;
    }

    operator std::string() const
    {
        return m_name;
    }

    Value value() const { return m_value; }
    void set_value(Value value) { m_value = value; }
    uint32_t id() const { return m_id; }
    void set_id(uint32_t id) { m_id = id; }
    const std::string &name() const { return m_name; }
    void set_name(const std::string &name) { m_name = name; }

private:
    static uint32_t get_symbol_id()
    {
        static uint32_t id = 0;
        return id++;
    }
    uint32_t m_id;
    std::string m_name;
    Value m_value;
    friend class ExprImpl;
};

Symbol operator"" _sym(const char *name, size_t len);
Symbol operator"" _sym(const char name);
class ExprImpl : public std::enable_shared_from_this<ExprImpl>
{
public:
    enum class Op
    {
        Func = 0,
        Branch = 1,
        Constant,
        Val,
        // memory
        Read,
        Write,
        // unary
        Negate,
        Reciprocal,
        Invert,
        Ceil,
        Floor,
        Round,
        RoundInt,
        Itof,
        Log2,
        Sqrt,
        Rsqrt,
        Max,
        Min,
        Production,
        Pow,
        // binary
        Add,
        Sub,
        Mul,
        Mulh,
        Div,
        FloorDiv,
        Mod,
        LeftShift,
        RightShift,
        And,
        Or,
        Xor,
        // compare
        Less,
        LessEqual,
        Greater,
        GreaterEqual,
        Equal,
        NotEqual,
        _OP_NUM
    };

    static ExprPtr create(const Symbol &s)
    {
        return ExprPtr(new ExprImpl(s));
    }
    static ExprPtr create(Symbol::Value v)
    {
        return ExprPtr(new ExprImpl(v));
    }
    template <typename... Args>
    static ExprPtr create(Op op, Args... args)
    {
        return ExprPtr(new ExprImpl(op, std::vector<ExprPtr>{args...}))->simplify();
    }
    static ExprPtr parse(const std::string &str)
    {
        std::string tmp = str;
        tmp.erase(std::remove_if(tmp.begin(), tmp.end(), ::isspace), tmp.end());
        uint32_t pos = 0;
        std::map<std::string, uint32_t> ids;
        return parse(tmp, pos, ids);
    }
    static ExprPtr replace_symbol(ExprPtr expr, const std::map<std::string, ExprPtr> &symbols);

    static const std::string &op_to_string(Op op);
    std::string to_string() const;
    ExprPtr canonicalize();
    ExprPtr simplify();
    std::vector<Symbol> variables() const;
    bool is_constant() const
    {
        return variables().empty();
    }
    bool is_integral_constant() const
    {
        if (!is_constant())
        {
            return false;
        }
        auto v = value();
        if (std::trunc(v) == v)
        {
            return true;
        }
        return false;
    }
    bool eval(const std::map<Symbol, Symbol::Value> &vars = {});
    bool equal(ExprPtr other) const;

    Op op() const { return m_op; }
    const std::vector<ExprPtr> &args() const { return m_args; }
    const Symbol &symbol() const { return m_symbol; }
    void set_value(Symbol::Value value) { m_symbol.set_value(value); }
    Symbol::Value value() const { return m_symbol.value(); }
    uint32_t depth() const;

protected:
    ExprImpl(const Symbol &s) : m_symbol(s) {}
    ExprImpl(Symbol::Value v) : m_op(Op::Val), m_symbol(Symbol(v)) {}
    ExprImpl(Op op, const std::vector<ExprPtr> &args) : m_op(op), m_args(args) {}
    static const std::map<ExprImpl::Op, std::function<ExprPtr(ExprPtr)>> &get_constant_folding();
    static ExprPtr constant_folding(ExprPtr x);
    static ExprPtr parse(const std::string &str, uint32_t &pos, std::map<std::string, uint32_t> &ids);
    friend class Expr;

    Op m_op{Op::Constant};
    Symbol m_symbol;
    std::vector<ExprPtr> m_args;
};

class Expr
{
public:
    Expr() : m_ptr(nullptr) {}
    Expr(std::nullptr_t p) : m_ptr(p) {}
    Expr(ExprPtr p) : m_ptr(p) {}
    Expr(const Symbol &s) : m_ptr(ExprImpl::create(s)) {}
    template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
    Expr(T v) : m_ptr(ExprImpl::create((Symbol::Value)v)) {}
    Expr(const std::string &s) : m_ptr(ExprImpl::parse(s)) {}

    ExprImpl &operator*()
    {
        return *m_ptr;
    }

    const ExprImpl &operator*() const
    {
        return *m_ptr;
    }

    ExprImpl *operator->()
    {
        return m_ptr.get();
    }

    const ExprImpl *operator->() const
    {
        return m_ptr.get();
    }

    operator ExprPtr() const
    {
        return m_ptr;
    }

    operator bool() const
    {
        return m_ptr != nullptr;
    }

    Expr reciprocal() const
    {
        return ExprImpl::create(ExprImpl::Op::Reciprocal, m_ptr);
    }
    Expr ceil() const
    {
        return ExprImpl::create(ExprImpl::Op::Ceil, m_ptr);
    }
    Expr floor() const
    {
        return ExprImpl::create(ExprImpl::Op::Floor, m_ptr);
    }
    Expr round() const
    {
        return ExprImpl::create(ExprImpl::Op::Round, m_ptr);
    }
    Expr rint() const
    {
        return ExprImpl::create(ExprImpl::Op::RoundInt, m_ptr);
    }
    Expr itof() const
    {
        return ExprImpl::create(ExprImpl::Op::Itof, m_ptr);
    }
    Expr log2() const
    {
        return ExprImpl::create(ExprImpl::Op::Log2, m_ptr);
    }
    Expr sqrt() const
    {
        return ExprImpl::create(ExprImpl::Op::Sqrt, m_ptr);
    }
    Expr rsqrt() const
    {
        return ExprImpl::create(ExprImpl::Op::Rsqrt, m_ptr);
    }
    Expr max(const Expr &o) const
    {
        return ExprImpl::create(ExprImpl::Op::Max, m_ptr, o);
    }
    Expr max(const std::vector<Expr> &args) const
    {
        std::vector<ExprPtr> m_args{m_ptr};
        for (auto arg : args)
        {
            m_args.push_back(arg);
        }
        return ExprImpl::create(ExprImpl::Op::Max, m_args);
    }
    Expr min(const Expr &o) const
    {
        return ExprImpl::create(ExprImpl::Op::Min, m_ptr, o);
    }
    Expr min(const std::vector<Expr> &args) const
    {
        std::vector<ExprPtr> m_args{m_ptr};
        for (auto arg : args)
        {
            m_args.push_back(arg);
        }
        return ExprImpl::create(ExprImpl::Op::Min, m_args);
    }
    Expr prod(const std::vector<Expr> &args) const
    {
        std::vector<ExprPtr> m_args{m_ptr};
        for (auto arg : args)
        {
            m_args.push_back(arg);
        }
        return ExprImpl::create(ExprImpl::Op::Production, m_args);
    }
    Expr pow(int n) const
    {
        return ExprImpl::create(ExprImpl::Op::Pow, m_ptr, Expr(n));
    }
    Expr floordiv(const Expr &o) const
    {
        return ExprImpl::create(ExprImpl::Op::FloorDiv, m_ptr, o);
    }
    Expr mulh(const Expr &o) const
    {
        return ExprImpl::create(ExprImpl::Op::Mulh, m_ptr, o);
    }

    static Expr select(const Expr &cond, const Expr &true_o, const Expr &false_o)
    {
        return ExprImpl::create(ExprImpl::Op::Branch, cond, true_o, false_o);
    }

protected:
    ExprPtr m_ptr;
};

#define DECLARE_BASE_OPERATOR(mtype, mop, mfilter)                                      \
    inline Expr operator mop(const Expr &a, const Expr &b)                              \
    {                                                                                   \
        return ExprImpl::create(mtype, a, b);                                           \
    }                                                                                   \
    inline Expr operator mop(const Expr &o, const Symbol &s)                            \
    {                                                                                   \
        return ExprImpl::create(mtype, o, Expr(s));                                     \
    }                                                                                   \
    inline Expr operator mop(const Symbol &s, const Expr &o)                            \
    {                                                                                   \
        return ExprImpl::create(mtype, Expr(s), o);                                     \
    }                                                                                   \
    template <typename T, typename std::enable_if<mfilter<T>::value>::type * = nullptr> \
    inline Expr operator mop(const Expr &o, T v)                                        \
    {                                                                                   \
        return ExprImpl::create(mtype, o, Expr(v));                                     \
    }                                                                                   \
    template <typename T, typename std::enable_if<mfilter<T>::value>::type * = nullptr> \
    inline Expr operator mop(T v, const Expr &o)                                        \
    {                                                                                   \
        return ExprImpl::create(mtype, Expr(v), o);                                     \
    }                                                                                   \
    template <typename T, typename std::enable_if<mfilter<T>::value>::type * = nullptr> \
    inline Expr operator mop(const Symbol &s, T v)                                      \
    {                                                                                   \
        return ExprImpl::create(mtype, Expr(s), Expr(v));                               \
    }                                                                                   \
    template <typename T, typename std::enable_if<mfilter<T>::value>::type * = nullptr> \
    inline Expr operator mop(T v, const Symbol &s)                                      \
    {                                                                                   \
        return ExprImpl::create(mtype, Expr(v), Expr(s));                               \
    }

#define DECLARE_ARITHMETIC_OPERATOR(mtype, mop) \
    DECLARE_BASE_OPERATOR(mtype, mop, std::is_arithmetic)

#define DECLARE_BITWISE_OPERATOR(mtype, mop) \
    DECLARE_BASE_OPERATOR(mtype, mop, std::is_integral)

#define DECLARE_COMPARE_OPERATOR(mtype, mop) \
    DECLARE_BASE_OPERATOR(mtype, mop, std::is_arithmetic)

#define DECLARE_ARITHMETIC_OPERATOR2(mtype, mop)                                                   \
    inline Expr &operator mop##=(Expr &a, const Expr &b)                                           \
    {                                                                                              \
        a = ExprImpl::create(mtype, a, b);                                                         \
        return a;                                                                                  \
    }                                                                                              \
    inline Expr &operator mop##=(Expr &o, const Symbol &s)                                         \
    {                                                                                              \
        o = ExprImpl::create(mtype, o, Expr(s));                                                   \
        return o;                                                                                  \
    }                                                                                              \
    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr> \
    inline Expr &operator mop##=(Expr &o, T v)                                                     \
    {                                                                                              \
        o = ExprImpl::create(mtype, o, Expr(v));                                                   \
        return o;                                                                                  \
    }

inline Expr operator-(const Expr &o)
{
    return ExprImpl::create(ExprImpl::Op::Negate, o);
}
inline Expr operator-(const Symbol &s)
{
    return ExprImpl::create(ExprImpl::Op::Negate, Expr(s));
}

DECLARE_ARITHMETIC_OPERATOR(ExprImpl::Op::Add, +);
DECLARE_ARITHMETIC_OPERATOR(ExprImpl::Op::Sub, -);
DECLARE_ARITHMETIC_OPERATOR(ExprImpl::Op::Mul, *);
DECLARE_ARITHMETIC_OPERATOR(ExprImpl::Op::Div, /);
DECLARE_ARITHMETIC_OPERATOR(ExprImpl::Op::Mod, %);

DECLARE_ARITHMETIC_OPERATOR2(ExprImpl::Op::Add, +);
DECLARE_ARITHMETIC_OPERATOR2(ExprImpl::Op::Sub, -);
DECLARE_ARITHMETIC_OPERATOR2(ExprImpl::Op::Mul, *);
DECLARE_ARITHMETIC_OPERATOR2(ExprImpl::Op::Div, /);

DECLARE_BITWISE_OPERATOR(ExprImpl::Op::LeftShift, <<);
DECLARE_BITWISE_OPERATOR(ExprImpl::Op::RightShift, >>);
DECLARE_BITWISE_OPERATOR(ExprImpl::Op::And, &);
DECLARE_BITWISE_OPERATOR(ExprImpl::Op::Or, |);
DECLARE_BITWISE_OPERATOR(ExprImpl::Op::Xor, ^);

inline Expr operator~(const Expr &o)
{
    return ExprImpl::create(ExprImpl::Op::Invert, o);
}
inline Expr operator~(const Symbol &s)
{
    return ExprImpl::create(ExprImpl::Op::Invert, Expr(s));
}

DECLARE_COMPARE_OPERATOR(ExprImpl::Op::Less, <);
DECLARE_COMPARE_OPERATOR(ExprImpl::Op::LessEqual, <=);
DECLARE_COMPARE_OPERATOR(ExprImpl::Op::Greater, >);
DECLARE_COMPARE_OPERATOR(ExprImpl::Op::GreaterEqual, >=);
DECLARE_COMPARE_OPERATOR(ExprImpl::Op::Equal, ==);
DECLARE_COMPARE_OPERATOR(ExprImpl::Op::NotEqual, !=);

inline bool operator==(const Symbol &a, const Symbol &b)
{
    return a.id() == b.id();
}
inline bool operator!=(const Symbol &a, const Symbol &b)
{
    return a.id() != b.id();
}
inline bool operator<(const Symbol &a, const Symbol &b)
{
    return a.id() < b.id();
}
inline bool operator<=(const Symbol &a, const Symbol &b)
{
    return a.id() <= b.id();
}
inline bool operator>(const Symbol &a, const Symbol &b)
{
    return a.id() > b.id();
}
inline bool operator>=(const Symbol &a, const Symbol &b)
{
    return a.id() >= b.id();
}

std::ostream &operator<<(std::ostream &os, const Symbol &sym);
std::ostream &operator<<(std::ostream &os, Expr expr);

// relaxed mode: only focus on structural equal
bool compare(ExprPtr a, const std::vector<Symbol> &arga,
             ExprPtr b, const std::vector<Symbol> &argb,
             bool relaxed_mode = false);

sbe::Function expr_to_func(Expr expr, const std::string &func_name);
std::vector<std::string> expr_to_cl_code(Expr expr, std::string &res);
std::string expr_to_cl_func(Expr expr, const std::string &func_name);

} // namespace tongue
