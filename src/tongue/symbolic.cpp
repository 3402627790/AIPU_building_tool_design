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
 * @Date: 2023-03-21 11:44:13
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-11-14 16:00:29
 */
#include "tongue/sbe/ssa.h"
#include <tongue/symbolic.h>
using namespace tongue::sbe;

namespace tongue
{
std::ostream &operator<<(std::ostream &os, const Symbol &sym)
{
    return os << sym.operator std::string();
}
std::ostream &operator<<(std::ostream &os, Expr expr)
{
    if (expr)
    {
        return os << expr->to_string();
    }
    return os;
}

Symbol operator"" _sym(const char *name, size_t len)
{
    return Symbol(std::string(name, len));
}
Symbol operator"" _sym(const char name)
{
    return Symbol(std::string(&name, 1));
}
ExprPtr ExprImpl::parse(const std::string &str, uint32_t &pos, std::map<std::string, uint32_t> &ids)
{
    static std::map<std::string, ExprImpl::Op> funcs = {
        {"ceil", ExprImpl::Op::Ceil},
        {"floor", ExprImpl::Op::Floor},
        {"round", ExprImpl::Op::Round},
        {"rint", ExprImpl::Op::RoundInt},
        {"itof", ExprImpl::Op::Itof},
        {"log2", ExprImpl::Op::Log2},
        {"sqrt", ExprImpl::Op::Sqrt},
        {"rsqrt", ExprImpl::Op::Rsqrt},
        {"max", ExprImpl::Op::Max},
        {"min", ExprImpl::Op::Min},
        {"prod", ExprImpl::Op::Production},
        {"mulh", ExprImpl::Op::Mulh},
    };

    static std::map<std::string, ExprImpl::Op> ops = {
        {"**", ExprImpl::Op::Pow},
        {"+", ExprImpl::Op::Add},
        {"-", ExprImpl::Op::Sub},
        {"*", ExprImpl::Op::Mul},
        {"/", ExprImpl::Op::Div},
        {"//", ExprImpl::Op::FloorDiv},
        {"%", ExprImpl::Op::Mod},
        {"<<", ExprImpl::Op::LeftShift},
        {">>", ExprImpl::Op::RightShift},
        {"&", ExprImpl::Op::And},
        {"|", ExprImpl::Op::Or},
        {"^", ExprImpl::Op::Xor},
        {"<", ExprImpl::Op::Less},
        {"<=", ExprImpl::Op::LessEqual},
        {">", ExprImpl::Op::Greater},
        {">=", ExprImpl::Op::GreaterEqual},
        {"==", ExprImpl::Op::Equal},
        {"!=", ExprImpl::Op::NotEqual},
    };

    // parsing operators
    if (pos < str.size() && str[pos] == '(')
    {
        pos++;
        ExprImpl::Op m_op = ExprImpl::Op::Constant;
        std::vector<ExprPtr> args;
        auto arg = parse(str, pos, ids);
        if (arg)
        {
            args.push_back(arg);
            if (pos < str.size() - 1)
            {
                auto op = str.substr(pos, 2);
                if (ops.count(op))
                {
                    pos += 2;
                    m_op = ops[op];
                }
            }
            if (m_op == ExprImpl::Op::Constant && pos < str.size())
            {
                auto op = str.substr(pos, 1);
                if (ops.count(op))
                {
                    pos++;
                    m_op = ops[op];
                }
                else if (str[pos] == '?')
                {
                    pos++;
                    arg = parse(str, pos, ids);
                    if (arg && pos < str.size() && str[pos] == ':')
                    {
                        pos++;
                        m_op = ExprImpl::Op::Branch;
                        args.push_back(arg);
                    }
                }
            }
        }
        else if (pos < str.size())
        {
            if (str[pos] == '-')
            {
                pos++;
                m_op = ExprImpl::Op::Negate;
            }
            else if (str[pos] == '~')
            {
                pos++;
                m_op = ExprImpl::Op::Invert;
            }
        }
        if (m_op != ExprImpl::Op::Constant)
        {
            arg = parse(str, pos, ids);
            if (arg && pos < str.size() && str[pos] == ')')
            {
                pos++;
                args.push_back(arg);
                return ExprImpl::create(m_op, args);
            }
        }
    }

    // parsing values
    if ((pos < str.size() && isdigit(str[pos])) ||
        (pos < str.size() - 1 && str[pos] == '-' && isdigit(str[pos + 1])))
    {
        std::stringstream ss;
        bool is_hex = false;
        if (pos < str.size() - 1 && str[pos] == '0' && (str[pos + 1] == 'x' || str[pos + 1] == 'X'))
        {
            is_hex = true;
        }
        ss << str[pos];
        pos++;
        while (pos < str.size() &&
               (isdigit(str[pos]) ||
                str[pos] == '.' ||
                str[pos] == 'x' ||
                str[pos] == 'X' ||
                (str[pos] >= 'a' && str[pos] <= 'f') ||
                (str[pos] >= 'A' && str[pos] <= 'F')))
        {
            ss << str[pos];
            pos++;
        }
        Symbol::Value val = 0;
        if (is_hex)
        {
            int64_t num = 0;
            ss >> std::noskipws >> std::hex >> num;
            val = num;
        }
        else
        {
            ss >> std::noskipws >> val;
        }
        if (!ss.bad() && ss.eof())
        {
            return ExprImpl::create(val);
        }
    }

    // parsing functions and symbols
    if (pos < str.size() && isalpha(str[pos]))
    {
        std::stringstream ss;
        while (pos < str.size() && (isalnum(str[pos]) || str[pos] == '_'))
        {
            ss << str[pos];
            pos++;
        }
        if (pos < str.size() && str[pos] == '(')
        {
            pos++;
            std::vector<ExprPtr> args;
            while (pos < str.size() && str[pos] != ')')
            {
                auto arg = parse(str, pos, ids);
                if (arg)
                {
                    args.push_back(arg);
                }
                if (pos < str.size() && str[pos] == ',')
                {
                    pos++;
                }
            }
            auto func = ss.str();
            if (funcs.count(func) && args.size() && pos < str.size() && str[pos] == ')')
            {
                pos++;
                return ExprImpl::create(funcs[func], args);
            }
        }
        else if (ss.rdbuf()->in_avail())
        {
            auto name = ss.str();
            auto var = ExprImpl::create(Symbol(name));
            if (ids.count(name))
            {
                var->m_symbol.set_id(ids[name]);
            }
            else
            {
                ids[name] = var->m_symbol.id();
            }
            return var;
        }
    }

    return nullptr;
}
ExprPtr ExprImpl::replace_symbol(ExprPtr expr, const std::map<std::string, ExprPtr> &symbols)
{
    if (expr->op() == ExprImpl::Op::Val)
    {
        return expr;
    }
    else if (expr->op() == ExprImpl::Op::Constant)
    {
        auto name = expr->symbol().name();
        if (symbols.count(name))
        {
            return symbols.at(name);
        }
        else
        {
            return expr;
        }
    }
    else
    {
        std::vector<ExprPtr> args;
        for (auto &arg : expr->args())
        {
            args.push_back(replace_symbol(arg, symbols));
        }
        return ExprImpl::create(expr->op(), args);
    }
}
const std::string &ExprImpl::op_to_string(Op op)
{
    static const std::vector<std::string> names = {
        "Func",
        "Branch",
        "Constant",
        "Val",
        "Read",
        "Write",
        "Negate",
        "Reciprocal",
        "Invert",
        "Ceil",
        "Floor",
        "Round",
        "RoundInt",
        "Itof",
        "Log2",
        "Sqrt",
        "Rsqrt",
        "Max",
        "Min",
        "Production",
        "Pow",
        "Add",
        "Sub",
        "Mul",
        "Mulh",
        "Div",
        "FloorDiv",
        "Mod",
        "LeftShift",
        "RightShift",
        "And",
        "Or",
        "Xor",
        "Less",
        "LessEqual",
        "Greater",
        "GreaterEqual",
        "Equal",
        "NotEqual",
    };
    static const std::string unkown = "unkown";
    if (op < Op::Func || op >= Op::_OP_NUM)
    {
        return unkown;
    }
    return names[(int)op];
}
std::string ExprImpl::to_string() const
{
    if (m_op == Op::Constant)
    {
        return m_symbol;
    }
    else if (m_op == Op::Val)
    {
        auto v = value();
        if (trunc(v) == v)
        {
            return std::to_string((long long)v);
        }
        return std::to_string(value());
    }
    else if (m_op == Op::Func)
    {
    }
    else if (m_op == Op::Branch)
    {
        return "(" + m_args[0]->to_string() + "?" + m_args[1]->to_string() + ":" + m_args[2]->to_string() + ")";
    }
    else
    {
        static std::vector<std::string> ops = {
            "-",
            "1/",
            "~",
            "ceil(",
            "floor(",
            "round(",
            "rint(",
            "itof(",
            "log2(",
            "sqrt(",
            "rsqrt(",
            "max(",
            "min(",
            "prod(",
            "**",
            "+",
            "-",
            "*",
            "mulh(",
            "/",
            "//",
            "%",
            "<<",
            ">>",
            "&",
            "|",
            "^",
            "<",
            "<=",
            ">",
            ">=",
            "==",
            "!=",
        };
        std::string s = ops[(int)m_op - (int)Op::Negate];

        if (m_op < Op::Ceil)
        {
            return "(" + s + m_args[0]->to_string() + ")";
        }
        else if (m_op < Op::Pow || m_op == Op::Mulh)
        {
            for (uint32_t i = 0; i < m_args.size(); ++i)
            {
                s += m_args[i]->to_string();
                if (i != m_args.size() - 1)
                {
                    s += ",";
                }
            }
            return s + ")";
        }
        else if (m_op == Op::Add)
        {
            if (m_args[1]->op() == Op::Val && m_args[1]->value() < 0)
            {
                return "(" + m_args[0]->to_string() + m_args[1]->to_string() + ")";
            }
        }
        else if (m_op >= Op::And && m_op <= Op::Xor)
        {
            if (m_args[1]->op() == Op::Val)
            {
                long long v = m_args[1]->value();
                return "(" + m_args[0]->to_string() + s + utils::int_to_hex(v) + ")";
            }
        }
        return "(" + m_args[0]->to_string() + s + m_args[1]->to_string() + ")";
    }
    return std::string();
}
uint32_t ExprImpl::depth() const
{
    if (m_op == Op::Constant || m_op == Op::Val)
    {
        return 0;
    }
    uint32_t d = 0;
    for (auto arg : m_args)
    {
        auto dd = arg->depth();
        if (dd > d)
        {
            d = dd;
        }
    }
    return d + 1;
}

const std::map<ExprImpl::Op, std::function<ExprPtr(ExprPtr)>> &ExprImpl::get_constant_folding()
{
    static std::map<ExprImpl::Op, std::function<ExprPtr(ExprPtr)>> methods =
        {
            {ExprImpl::Op::Branch, [](ExprPtr x) -> ExprPtr {
                 auto &args = x->args();
                 if (args[0]->m_op == Op::Val)
                 {
                     if (args[0]->value() != 0)
                     {
                         return args[1];
                     }
                     return args[2];
                 }
                 return x;
             }},
        };
    return methods;
}
ExprPtr ExprImpl::constant_folding(ExprPtr x)
{
    auto &methods = get_constant_folding();
    if (methods.count(x->op()))
    {
        return methods.at(x->op())(x);
    }

    bool is_all_constants = true;
    for (auto a : x->m_args)
    {
        if (a->m_op != Op::Val)
        {
            is_all_constants = false;
            break;
        }
    }
    if (is_all_constants && x->eval())
    {
        return ExprImpl::create(x->value());
    }
    return x;
}

// x= op1(op0(a,b),c)
// if op0,op1 is distributive, and b,c are constants transform to
// x= op0(op1(a,c),op1(b,c))
//  (a+const)*const -> a*const +const*const
template <ExprImpl::Op op0, ExprImpl::Op op1>
ExprPtr distributive(ExprPtr x)
{
    if (x->op() == op1 && x->args().size() == 2)
    {
        auto lhs = x->args()[0];
        auto rhs = x->args()[1];
        if (lhs->op() == op0 &&
            lhs->args().size() == 2 &&
            rhs->op() == ExprImpl::Op::Val &&
            lhs->args()[1]->op() == ExprImpl::Op::Val)
        {
            return ExprImpl::create(op0,
                                    ExprImpl::create(op1, lhs->args()[0], Expr(rhs->value())),
                                    ExprImpl::create(op1, Expr(lhs->args()[1]->value()), Expr(rhs->value())));
        }
    }
    return x;
}

/**
 * @brief 
 * x= (a+const1)-(b+const2) -> x=(a-b)+(const1-const2)
 * x= (a-const1)+(b-const2) -> x=(a+b)-(const1+const2)
 * x= (a-const1)-(b-const2) -> x=(a-b)-(const1-const2)
 */
template <ExprImpl::Op op0, ExprImpl::Op op1>
ExprPtr commutative(ExprPtr x)
{
    if (x->op() == op0 && x->args().size() == 2)
    {
        auto lhs = x->args()[0];
        auto rhs = x->args()[1];
        if (lhs->op() == op1 &&
            lhs->args().size() == 2 &&
            lhs->args()[1]->op() == ExprImpl::Op::Val &&
            rhs->op() == op1 &&
            rhs->args().size() == 2 &&
            rhs->args()[1]->op() == ExprImpl::Op::Val)
        {
            return ExprImpl::create(op1,
                                    ExprImpl::create(op0, lhs->args()[0], rhs->args()[0]),
                                    ExprImpl::create(op0, Expr(lhs->args()[1]->value()), Expr(rhs->args()[1]->value())));
        }
    }
    return x;
}
/**
 * @brief 
 * if all arg are not constant
 * x = op0(op1(a,b),op1(a,c)) => x=op1(op0(b,c),a)
 * a*b+a -> (b+1)*a
 * a*b+b -> (a+1)*b
 * a*b+a*c -> (b+c)*a
 * a*b+b*c -> (a+c)*b
 * a*b+c*a -> (b+c)*a
 * a*b+c*b -> (a+c)*b 
 * @param x 
 * @return template <ExprImpl::Op op0, ExprImpl::Op op1> 
 */
template <ExprImpl::Op op0, ExprImpl::Op op1, bool op1_commutative>
ExprPtr associative(ExprPtr x)
{
    if (x->op() == op0 && x->args().size() == 2)
    {
        auto lhs = x->args()[0];
        auto rhs = x->args()[1];
        if (lhs->op() == op1 &&
            lhs->args().size() == 2)
        {
            if (rhs->op() == ExprImpl::Op::Constant)
            {
                //a*b+a -> (b+1)*a
                if (lhs->args()[0]->op() == ExprImpl::Op::Constant &&
                    lhs->args()[0]->symbol() == rhs->symbol())
                {
                    return ExprImpl::create(op1, ExprImpl::create(op0, lhs->args()[1], Expr(1)), lhs->args()[0]);
                }
                //a*b+b-> (a+1)*b
                else if (lhs->args()[1]->op() == ExprImpl::Op::Constant &&
                         lhs->args()[1]->symbol() == rhs->symbol())
                {
                    return ExprImpl::create(op1, ExprImpl::create(op0, lhs->args()[0], Expr(1)), lhs->args()[1]);
                }
            }
            // a*b+c*d
            else if (rhs->op() == op1 &&
                     rhs->args().size() == 2)
            {
                // a/b+a/d -> a/(b+d)
                if (lhs->args()[0]->op() != ExprImpl::Op::Val && lhs->args()[0]->equal(rhs->args()[0]))
                {
                    return ExprImpl::create(op1, lhs->args()[0], ExprImpl::create(op0, lhs->args()[1], rhs->args()[1]));
                }
                // a*b-c*a ->(b-c)*a
                else if (op1_commutative && lhs->args()[0]->op() != ExprImpl::Op::Val && lhs->args()[0]->equal(rhs->args()[1]))
                {
                    return ExprImpl::create(op1, ExprImpl::create(op0, lhs->args()[1], rhs->args()[0]), lhs->args()[0]);
                }
                // a*b+b*c ->(a+c)*b
                else if (op1_commutative && lhs->args()[1]->op() != ExprImpl::Op::Val && lhs->args()[1]->equal(rhs->args()[0]))
                {
                    return ExprImpl::create(op1, ExprImpl::create(op0, lhs->args()[0], rhs->args()[1]), lhs->args()[1]);
                }
                // a/b-c/b ->(a-c)*b
                else if (lhs->args()[1]->op() != ExprImpl::Op::Val && lhs->args()[1]->equal(rhs->args()[1]))
                {
                    return ExprImpl::create(op1, ExprImpl::create(op0, lhs->args()[0], rhs->args()[0]), lhs->args()[1]);
                }
            }
        }
    }
    return x;
}

ExprPtr ExprImpl::canonicalize()
{
    if (m_op == Op::Constant || m_op == Op::Val)
    {
        return shared_from_this();
    }
    for (auto arg : m_args)
    {
        arg->canonicalize();
    }
    if (m_op >= Op::Max && m_op <= Op::Production)
    {
        std::vector<ExprPtr> vals, symbols, rest;
        for (auto arg : m_args)
        {
            if (arg->op() == Op::Val)
            {
                vals.push_back(arg);
            }
            else if (arg->op() == Op::Constant)
            {
                symbols.push_back(arg);
            }
            else
            {
                rest.push_back(arg);
            }
        }
        std::sort(rest.begin(), rest.end(), [](ExprPtr a, ExprPtr b) -> bool {
            return a->depth() > b->depth();
        });
        std::sort(symbols.begin(), symbols.end(), [](ExprPtr a, ExprPtr b) -> bool {
            return a->symbol() < b->symbol();
        });
        rest.insert(rest.end(), symbols.begin(), symbols.end());
        rest.insert(rest.end(), vals.begin(), vals.end());
        m_args = std::move(rest);
    }
    if (m_op == Op::Add || m_op == Op::Mul || (m_op >= Op::And && m_op <= Op::Xor))
    {
        auto lhs = m_args[0];
        auto rhs = m_args[1];
        if (lhs->op() == Op::Val && rhs->op() != Op::Val)
        {
            m_args = {rhs, lhs};
        }
        else if (lhs->op() == Op::Constant && rhs->op() == Op::Constant && lhs->symbol() > rhs->symbol())
        {
            m_args = {rhs, lhs};
        }
        else if (lhs->depth() < rhs->depth())
        {
            m_args = {rhs, lhs};
        }
    }
    // (a+s)+b -> (a+b)+s
    // (a+v)+b -> (a+b)+v
    // (a+s)+s -> (a+s)+s
    // (s+s)+s -> (s+s)+s
    // (a+v)+s -> (a+s)+v
    // (s+v)+s -> (s+s)+v
    if ((m_op >= Op::Add && m_op <= Op::FloorDiv) || (m_op >= Op::And && m_op <= Op::Xor))
    {
        auto lhs = m_args[0];
        auto rhs = m_args[1];
        if (lhs->op() == m_op)
        {
            auto llhs = lhs->args()[0];
            auto lrhs = lhs->args()[1];
            if (lrhs->depth() < rhs->depth())
            {
                lhs = create(lhs->op(), llhs, rhs)->canonicalize();
                m_args = {lhs, lrhs};
            }
            else if (lrhs->op() == Op::Constant && rhs->op() == Op::Constant && lrhs->symbol() > rhs->symbol())
            {
                lhs = create(lhs->op(), llhs, rhs)->canonicalize();
                m_args = {lhs, lrhs};
            }
            else if (llhs->op() != Op::Val && lrhs->op() == Op::Val && rhs->op() == Op::Constant)
            {
                lhs = create(lhs->op(), llhs, rhs)->canonicalize();
                m_args = {lhs, lrhs};
            }
        }
    }
    // a-b-c -> a-c-b
    // a/b/c -> a/c/b
    // a//b//c -> a//c//b
    if (m_op == Op::Sub || m_op == Op::Div || m_op == Op::FloorDiv)
    {
        auto lhs = m_args[0];
        auto rhs = m_args[1];
        if (lhs->op() == m_op)
        {
            auto llhs = lhs->args()[0];
            auto lrhs = lhs->args()[1];
            if (llhs->op() == Op::Val && lrhs->op() != Op::Val && rhs->op() == Op::Val)
            {
                lhs = create(lhs->op(), llhs, rhs)->canonicalize();
                m_args = {lhs, lrhs};
            }
        }
    }
    // move minus after plus
    // (a-b)+c -> (a+c)-b
    if (m_op == Op::Add)
    {
        auto lhs = m_args[0];
        auto rhs = m_args[1];
        if (lhs->op() == Op::Sub)
        {
            auto llhs = lhs->args()[0];
            auto lrhs = lhs->args()[1];
            if (!(llhs->op() == Op::Val && lrhs->op() == Op::Val))
            {
                lhs = create(Op::Add, llhs, rhs)->canonicalize();
                m_args = {lhs, lrhs};
                m_op = Op::Sub;
            }
        }
    }
    // move div after mul
    // (a/b)*c -> (a*c)/b
    if (m_op == Op::Mul)
    {
        auto lhs = m_args[0];
        auto rhs = m_args[1];
        if (lhs->op() == Op::Div)
        {
            auto llhs = lhs->args()[0];
            auto lrhs = lhs->args()[1];
            if (!(llhs->op() == Op::Val && lrhs->op() == Op::Val))
            {
                lhs = create(Op::Mul, llhs, rhs)->canonicalize();
                m_args = {lhs, lrhs};
                m_op = Op::Div;
            }
        }
    }
    // (a-b)*(a+c) -> (a+c)*(a-b)
    if (m_op == Op::Mul)
    {
        auto lhs = m_args[0];
        auto rhs = m_args[1];
        if (lhs->op() == Op::Sub && rhs->op() == Op::Add && lhs->depth() == rhs->depth())
        {
            m_args = {rhs, lhs};
        }
    }
    return shared_from_this();
}
ExprPtr ExprImpl::simplify()
{
    if (m_op == Op::Constant || m_op == Op::Val)
    {
        return shared_from_this();
    }
    canonicalize();
    std::vector<ExprPtr> args = m_args;
    static std::vector<std::function<ExprPtr(ExprPtr)>> basic_ops =
        {
            distributive<Op::Add, Op::Mul>,
            distributive<Op::Sub, Op::Mul>,
            distributive<Op::Max, Op::Add>,
            distributive<Op::Max, Op::Sub>,
            distributive<Op::Min, Op::Add>,
            distributive<Op::Min, Op::Sub>,
            commutative<Op::Add, Op::Add>,
            commutative<Op::Add, Op::Sub>,
            commutative<Op::Sub, Op::Add>,
            commutative<Op::Sub, Op::Sub>,
            commutative<Op::Sub, Op::Sub>,
            commutative<Op::Mul, Op::Mul>,
            commutative<Op::Mul, Op::Div>,
            commutative<Op::Div, Op::Mul>,
            commutative<Op::Div, Op::Div>,
            associative<Op::Add, Op::Mul, true>,
            associative<Op::Sub, Op::Mul, true>,
            // associative<Op::Add, Op::Div, false>,
            // associative<Op::Sub, Op::Div, false>,
            // associative<Op::Add, Op::FloorDiv, false>,
            // associative<Op::Sub, Op::FloorDiv, false>,
        };

    ExprPtr r = constant_folding(shared_from_this());
    for (auto law : basic_ops)
    {
        r = law(r);
    }
    if (r != shared_from_this())
    {
        return r;
    }

    switch (m_op)
    {
    case Op::Negate:
    case Op::Reciprocal:
    case Op::Invert:
    {
        if (args[0]->m_op == m_op)
        {
            return args[0]->args()[0];
        }
        break;
    }
    case Op::Ceil:
    case Op::Floor:
    case Op::Round:
    case Op::RoundInt:
    {
        auto arg = args[0];
        if (arg->m_op == m_op)
        {
            return arg;
        }
        break;
    }
    case Op::Itof:
    {
        break;
    }
    case Op::Log2:
    case Op::Sqrt:
    case Op::Rsqrt:
    {
        break;
    }
    case Op::Max:
    case Op::Min:
    case Op::Production:
    {
        std::vector<ExprPtr> vals, rest;
        for (auto arg : args)
        {
            if (arg->m_op == Op::Val)
            {
                vals.push_back(arg);
            }
            else
            {
                rest.push_back(arg);
            }
        }
        if (vals.size() > 1)
        {
            auto v = vals[0]->value();
            for (uint32_t i = 1; i < vals.size(); i++)
            {
                if (m_op == Op::Max)
                {
                    v = std::max(vals[i]->value(), v);
                }
                else if (m_op == Op::Min)
                {
                    v = std::min(vals[i]->value(), v);
                }
                else if (m_op == Op::Production)
                {
                    v = v * vals[i]->value();
                }
            }
            if (vals.size() == args.size())
            {
                return Expr(v);
            }
            else
            {
                rest.push_back(Expr(v));
                return create(m_op, rest)->simplify();
            }
        }
        if (args.size() == 1)
        {
            return args[0];
        }
        // max(a,a) -> a
        // min(a,a) -> a
        if (m_op == Op::Max || m_op == Op::Min)
        {
            std::vector<ExprPtr> filtered;
            for (auto arg : args)
            {
                bool existed = false;
                for (auto expr : filtered)
                {
                    if (expr->equal(arg))
                    {
                        existed = true;
                        break;
                    }
                }
                if (!existed)
                {
                    filtered.push_back(arg);
                }
            }
            if (filtered.size() != args.size())
            {
                return create(m_op, filtered)->simplify();
            }
        }
        break;
    }
    case Op::Pow:
    {
        auto lhs = args[0];
        auto rhs = args[1];
        if (lhs->m_op == Op::Val)
        {
            if (lhs->value() == 0)
            {
                return Expr(0);
            }
            if (lhs->value() == 1)
            {
                return Expr(1);
            }
        }
        if (rhs->m_op == Op::Val)
        {
            if (rhs->value() == 0)
            {
                return Expr(1);
            }
            if (rhs->value() == 1)
            {
                return lhs;
            }
            if (rhs->value() == -1)
            {
                return (Expr(1) / lhs)->simplify();
            }
        }
        break;
    }
    case Op::Add:
    {
        auto lhs = args[0];
        auto rhs = args[1];
        if (lhs->equal(rhs))
        {
            return (lhs * Expr(2))->simplify();
        }
        if (rhs->op() == Op::Val)
        {
            if (rhs->value() == 0)
            {
                return lhs;
            }
            if (lhs->op() == Op::Add)
            {
                auto llhs = lhs->args()[0];
                auto lrhs = lhs->args()[1];
                if (lrhs->op() == Op::Val)
                {
                    return (llhs + Expr(rhs->value() + lrhs->value()))->simplify();
                }
            }
            // a/b+c -> (a+b*c)/b
            if (lhs->op() == Op::Div)
            {
                auto llhs = lhs->args()[0];
                auto lrhs = lhs->args()[1];
                if (lrhs->op() == Op::Val)
                {
                    return ((llhs + Expr(lrhs->value() * rhs->value())) / lrhs)->simplify();
                }
            }
            // a//b+c -> (a+b*c)//b
            if (lhs->op() == Op::FloorDiv)
            {
                auto llhs = lhs->args()[0];
                auto lrhs = lhs->args()[1];
                if (lrhs->op() == Op::Val)
                {
                    Symbol::Value v = (int64_t)rhs->value();
                    if (rhs->value() == v)
                    {
                        return (llhs + Expr(lrhs->value() * rhs->value())).floordiv(lrhs)->simplify();
                    }
                }
            }
        }
        break;
    }
    case Op::Sub:
    {
        auto lhs = args[0];
        auto rhs = args[1];
        if (lhs->equal(rhs))
        {
            return Expr(0);
        }
        if (lhs->m_op == Op::Val)
        {
            if (rhs->m_op == Op::Val)
            {
                return Expr(lhs->value() - rhs->value());
            }
            if (rhs->op() == Op::Add || rhs->op() == Op::Sub)
            {
                auto rlhs = rhs->args()[0];
                auto rrhs = rhs->args()[1];
                if (rhs->op() == Op::Add && rrhs->op() == Op::Val)
                {
                    return (Expr(lhs->value() - rrhs->value()) - rlhs)->simplify();
                }
                if (rhs->op() == Op::Sub)
                {
                    if (rlhs->op() == Op::Val)
                    {
                        return (Expr(lhs->value() - rlhs->value()) + rrhs)->simplify();
                    }
                    if (rrhs->op() == Op::Val)
                    {
                        return (Expr(lhs->value() + rrhs->value()) - rlhs)->simplify();
                    }
                }
            }
            // a-b/c -> (a*c-b)/c
            if (rhs->op() == Op::Div)
            {
                auto rlhs = rhs->args()[0];
                auto rrhs = rhs->args()[1];
                if (rrhs->op() == Op::Val)
                {
                    return ((Expr(lhs->value() * rrhs->value()) - rlhs) / rrhs)->simplify();
                }
            }
            // a-b//c -> (a*c-b)//c
            if (rhs->op() == Op::FloorDiv)
            {
                auto rlhs = rhs->args()[0];
                auto rrhs = rhs->args()[1];
                if (rrhs->op() == Op::Val)
                {
                    Symbol::Value v = (int64_t)lhs->value();
                    if (lhs->value() == v)
                    {
                        return (Expr(lhs->value() * rrhs->value()) - rlhs).floordiv(rrhs)->simplify();
                    }
                }
            }
        }
        if (rhs->op() == Op::Val)
        {
            if (rhs->value() == 0)
            {
                return lhs;
            }
            if (lhs->op() == Op::Add || lhs->op() == Op::Sub)
            {
                auto llhs = lhs->args()[0];
                auto lrhs = lhs->args()[1];
                if (lrhs->op() == Op::Val)
                {
                    if (lhs->op() == Op::Add)
                    {
                        return (llhs + Expr(lrhs->value() - rhs->value()))->simplify();
                    }
                    if (lhs->op() == Op::Sub)
                    {
                        return (llhs - Expr(lrhs->value() + rhs->value()))->simplify();
                    }
                }
            }
            // a/b-c -> (a-b*c)/b
            if (lhs->op() == Op::Div)
            {
                auto llhs = lhs->args()[0];
                auto lrhs = lhs->args()[1];
                if (lrhs->op() == Op::Val)
                {
                    return ((llhs - Expr(lrhs->value() * rhs->value())) / lrhs)->simplify();
                }
            }
            // a//b-c -> (a-b*c)//b
            if (lhs->op() == Op::FloorDiv)
            {
                auto llhs = lhs->args()[0];
                auto lrhs = lhs->args()[1];
                if (lrhs->op() == Op::Val)
                {
                    Symbol::Value v = (int64_t)rhs->value();
                    if (rhs->value() == v)
                    {
                        return (llhs - Expr(lrhs->value() * rhs->value())).floordiv(lrhs)->simplify();
                    }
                }
            }
        }
        // (a+b)-a -> b
        if (lhs->op() == Op::Add)
        {
            auto llhs = lhs->args()[0];
            auto lrhs = lhs->args()[1];
            if (llhs->equal(rhs))
            {
                return lrhs;
            }
        }
        // a-(a-b) -> b
        if (rhs->op() == Op::Sub)
        {
            auto rlhs = rhs->args()[0];
            auto rrhs = rhs->args()[1];
            if (rlhs->equal(lhs))
            {
                return rrhs;
            }
        }
        break;
    }
    case Op::Mul:
    {
        auto lhs = args[0];
        auto rhs = args[1];
        if (rhs->m_op == Op::Val)
        {
            if (lhs->m_op == Op::Val)
            {
                return Expr(lhs->value() * rhs->value());
            }
            if (rhs->value() == 0)
            {
                return Expr(0);
            }
            if (rhs->value() == 1)
            {
                return lhs;
            }
            if (lhs->m_op == Op::Mul)
            {
                auto llhs = lhs->args()[0];
                auto lrhs = lhs->args()[1];
                if (lrhs->m_op == Op::Val)
                {
                    return (llhs * Expr(rhs->value() * lrhs->value()))->simplify();
                }
            }
            // a//b*b-> a&(xx if b is pow of 2)
            if (lhs->m_op == Op::FloorDiv)
            {
                auto llhs = lhs->args()[0];
                auto lrhs = lhs->args()[1];
                if (lrhs->m_op == Op::Val && lrhs->value() == rhs->value() && utils::is_pow_of2(rhs->value()))
                {
                    long long n = rhs->value();
                    for (uint32_t i = 0; i < 64; ++i)
                    {
                        if (n & (1 << i))
                        {
                            return (llhs & (~(uint32_t)((1 << i) - 1)))->simplify();
                        }
                    }
                }
            }
        }
        // a**b*a**c -> a**(b+c)
        // a**c*b**c -> (a*b)**c
        if (lhs->m_op == Op::Pow && rhs->m_op == Op::Pow)
        {
            auto llhs = lhs->args()[0];
            auto lrhs = lhs->args()[1];
            auto rlhs = rhs->args()[0];
            auto rrhs = rhs->args()[1];
            if (llhs->equal(rlhs))
            {
                return create(Op::Pow, llhs, lrhs + rrhs)->simplify();
            }
            if (lrhs->equal(rrhs))
            {
                return create(Op::Pow, llhs * rlhs, lrhs)->simplify();
            }
        }
        break;
    }
    case Op::Mulh:
    {
        break;
    }
    case Op::Div:
    {
        auto lhs = args[0];
        auto rhs = args[1];
        if (lhs->equal(rhs))
        {
            return Expr(1);
        }
        if (lhs->m_op == Op::Val)
        {
            if (lhs->value() == 0)
            {
                return Expr(0);
            }
            // const /(a*const)
            if (rhs->m_op == Op::Mul || rhs->m_op == Op::Div)
            {
                auto rlhs = rhs->args()[0];
                auto rrhs = rhs->args()[1];
                if (rhs->m_op == Op::Mul && rrhs->m_op == Op::Val && rrhs->value() != 0)
                {
                    return (Expr(lhs->value() / rrhs->value()) / rlhs)->simplify();
                }
                if (rhs->m_op == Op::Div)
                {
                    if (rlhs->m_op == Op::Val && rlhs->value() != 0)
                    {
                        return (Expr(lhs->value() / rlhs->value()) * rrhs)->simplify();
                    }
                    if (rrhs->m_op == Op::Val && rrhs->value() != 0)
                    {
                        return (Expr(lhs->value() * rrhs->value()) / rlhs)->simplify();
                    }
                }
            }
        }
        if (rhs->m_op == Op::Val)
        {
            if (rhs->value() == 1)
            {
                return lhs;
            }
            if (lhs->m_op == Op::Mul || lhs->m_op == Op::Div)
            {
                auto llhs = lhs->args()[0];
                auto lrhs = lhs->args()[1];
                if (lrhs->m_op == Op::Val)
                {
                    if (lhs->m_op == Op::Mul && rhs->value() != 0)
                    {
                        return (llhs * Expr(lrhs->value() / rhs->value()))->simplify();
                    }
                    if (lhs->m_op == Op::Div && lrhs->value() != 0)
                    {
                        return (llhs / Expr(lrhs->value() * rhs->value()))->simplify();
                    }
                }
            }
        }
        // a**b/a**c -> a**(b-c)
        // a**c/b**c -> (a/b)**c
        if (lhs->m_op == Op::Pow && rhs->m_op == Op::Pow)
        {
            auto llhs = lhs->args()[0];
            auto lrhs = lhs->args()[1];
            auto rlhs = rhs->args()[0];
            auto rrhs = rhs->args()[1];
            if (llhs->equal(rlhs))
            {
                return create(Op::Pow, llhs, lrhs - rrhs)->simplify();
            }
            if (lrhs->equal(rrhs))
            {
                return create(Op::Pow, llhs / rlhs, lrhs)->simplify();
            }
        }
        break;
    }
    case Op::FloorDiv:
    {
        auto lhs = args[0];
        auto rhs = args[1];
        if (lhs->equal(rhs))
        {
            return Expr(1);
        }
        if (lhs->m_op == Op::Val)
        {
            if (lhs->value() == 0)
            {
                return Expr(0);
            }
        }
        if (rhs->m_op == Op::Val)
        {
            if (rhs->value() == 1)
            {
                return lhs;
            }
            if (lhs->m_op == Op::FloorDiv)
            {
                Expr llhs = lhs->args()[0];
                Expr lrhs = lhs->args()[1];
                // a//const//const
                if (lrhs->m_op == Op::Val)
                {
                    return llhs.floordiv(Expr(lrhs->value() * rhs->value()))->simplify();
                }
            }
        }
        break;
    }
    case Op::Mod:
    {
        auto lhs = args[0];
        auto rhs = args[1];
        if (lhs->equal(rhs))
        {
            return Expr(0);
        }
        if (lhs->m_op == Op::Val)
        {
            if (lhs->value() == 0)
            {
                return Expr(0);
            }
        }
        break;
    }
    case Op::LeftShift:
    case Op::RightShift:
    {
        auto lhs = args[0];
        auto rhs = args[1];
        // 0<<a -> 0
        // 0>>a -> 0
        if (lhs->m_op == Op::Val && lhs->value() == 0)
        {
            return Expr(0);
        }
        // a<<0 -> a
        // a>>0 -> a
        if (rhs->m_op == Op::Val && rhs->value() == 0)
        {
            return lhs;
        }
        break;
    }
    case Op::And:
    case Op::Or:
    case Op::Xor:
    {
        auto lhs = args[0];
        auto rhs = args[1];
        if (m_op == Op::Xor && lhs->equal(rhs))
        {
            return Expr(0);
        }
        if (rhs->m_op == Op::Val)
        {
            // a&0 -> 0
            if (m_op == Op::And && rhs->value() == 0)
            {
                return Expr(0);
            }
            // a|0 -> a
            if (m_op == Op::Or && rhs->value() == 0)
            {
                return lhs;
            }
            // (a&const)&const
            if (lhs->m_op == m_op)
            {
                auto llhs = lhs->args()[0];
                auto lrhs = lhs->args()[1];
                if (lrhs->op() == Op::Val)
                {
                    Symbol::Value v = 0;
                    if (m_op == Op::And)
                    {
                        v = (int64_t)lrhs->value() & (int64_t)rhs->value();
                    }
                    else if (m_op == Op::Or)
                    {
                        v = (int64_t)lrhs->value() | (int64_t)rhs->value();
                    }
                    else
                    {
                        v = (int64_t)lrhs->value() ^ (int64_t)rhs->value();
                    }
                    return create(m_op, llhs, Expr(v))->simplify();
                }
            }
        }
        break;
    }
    case Op::Less:
    case Op::LessEqual:
    case Op::Greater:
    case Op::GreaterEqual:
    case Op::Equal:
    case Op::NotEqual:
    {
        auto lhs = args[0];
        auto rhs = args[1];
        if (lhs->equal(rhs))
        {
            if (m_op == Op::LessEqual || m_op == Op::GreaterEqual || m_op == Op::Equal)
            {
                return Expr(1);
            }
            else
            {
                return Expr(0);
            }
        }
        break;
    }
    default:
        break;
    }
    return shared_from_this();
}
std::vector<Symbol> ExprImpl::variables() const
{
    std::set<Symbol> res;
    if (m_op == Op::Constant)
    {
        res.insert(m_symbol);
    }
    else if (m_op != Op::Val)
    {
        for (auto arg : m_args)
        {
            for (auto &s : arg->variables())
            {
                res.insert(s);
            }
        }
    }
    return {res.begin(), res.end()};
}
bool ExprImpl::eval(const std::map<Symbol, Symbol::Value> &vars)
{
    bool ret = true;
    if (m_op == Op::Constant)
    {
        auto iter = vars.find(m_symbol);
        if (iter != vars.end())
        {
            m_symbol.m_value = iter->second;
        }
        else
        {
            ret = false;
        }
    }
    else if (m_op != Op::Val)
    {
        auto args = m_args;
        for (auto arg : args)
        {
            ret &= arg->eval(vars);
        }
        switch (m_op)
        {
        case Op::Branch:
        {
            if (args[0]->value() != 0)
            {
                m_symbol.m_value = args[1]->value();
            }
            else
            {
                m_symbol.m_value = args[2]->value();
            }
            break;
        }
        case Op::Negate:
        {
            m_symbol.m_value = -args[0]->value();
            break;
        }
        case Op::Reciprocal:
        {
            auto v = args[0]->value();
            if (v != 0)
            {
                m_symbol.m_value = 1 / v;
            }
            else
            {
                ret = false;
            }
            break;
        }
        case Op::Invert:
        {
            m_symbol.m_value = ~(int64_t)args[0]->value();
            break;
        }
        case Op::Ceil:
        {
            m_symbol.m_value = std::ceil(args[0]->value());
            break;
        }
        case Op::Floor:
        {
            m_symbol.m_value = std::floor(args[0]->value());
            break;
        }
        case Op::Round:
        {
            m_symbol.m_value = std::round(args[0]->value());
            break;
        }
        case Op::RoundInt:
        {
            m_symbol.m_value = std::rint(args[0]->value());
            break;
        }
        case Op::Itof:
        {
            m_symbol.m_value = args[0]->value();
            break;
        }
        case Op::Log2:
        {
            m_symbol.m_value = std::log2(args[0]->value());
            break;
        }
        case Op::Sqrt:
        {
            m_symbol.m_value = std::sqrt(args[0]->value());
            break;
        }
        case Op::Rsqrt:
        {
            Symbol::Value v = std::sqrt(args[0]->value());
            if (v != 0)
            {
                m_symbol.m_value = 1 / v;
            }
            else
            {
                ret = false;
            }
            break;
        }
        case Op::Max:
        {
            auto v = args[0]->value();
            for (uint32_t i = 1; i < args.size(); ++i)
            {
                v = std::max(v, args[i]->value());
            }
            m_symbol.m_value = v;
            break;
        }
        case Op::Min:
        {
            auto v = args[0]->value();
            for (uint32_t i = 1; i < args.size(); ++i)
            {
                v = std::min(v, args[i]->value());
            }
            m_symbol.m_value = v;
            break;
        }
        case Op::Production:
        {
            auto v = 1;
            for (auto &&arg : args)
            {
                v = v * arg->value();
            }
            m_symbol.m_value = v;
            break;
        }
        case Op::Pow:
        {
            auto v = args[1]->value();
            if (args[1]->op() == Op::Val && v == (int64_t)v)
            {
                m_symbol.m_value = std::pow(args[0]->value(), v);
            }
            else
            {
                ret = false;
            }
            break;
        }
        case Op::Add:
        {
            m_symbol.m_value = args[0]->value() + args[1]->value();
            break;
        }
        case Op::Sub:
        {
            m_symbol.m_value = args[0]->value() - args[1]->value();
            break;
        }
        case Op::Mul:
        {
            m_symbol.m_value = args[0]->value() * args[1]->value();
            break;
        }
        case Op::Mulh:
        {
            m_symbol.m_value = (((int64_t)args[0]->value() * (int64_t)args[1]->value()) >> 32) & 0xffffffff;
            break;
        }
        case Op::Div:
        {
            auto v = args[1]->value();
            if (v != 0)
            {
                m_symbol.m_value = args[0]->value() / v;
            }
            else
            {
                ret = false;
            }
            break;
        }
        case Op::FloorDiv:
        {
            auto v = args[1]->value();
            if (v != 0)
            {
                m_symbol.m_value = std::floor(args[0]->value() / v);
            }
            else
            {
                ret = false;
            }
            break;
        }
        case Op::Mod:
        {
            auto v = args[1]->value();
            if (v != 0)
            {
                m_symbol.m_value = std::fmod(args[0]->value(), v);
            }
            else
            {
                ret = false;
            }
            break;
        }
        case Op::LeftShift:
        {
            m_symbol.m_value = (int64_t)args[0]->value() << (int64_t)args[1]->value();
            break;
        }
        case Op::RightShift:
        {
            m_symbol.m_value = (int64_t)args[0]->value() >> (int64_t)args[1]->value();
            break;
        }
        case Op::And:
        {
            m_symbol.m_value = (int64_t)args[0]->value() & (int64_t)args[1]->value();
            break;
        }
        case Op::Or:
        {
            m_symbol.m_value = (int64_t)args[0]->value() | (int64_t)args[1]->value();
            break;
        }
        case Op::Xor:
        {
            m_symbol.m_value = (int64_t)args[0]->value() ^ (int64_t)args[1]->value();
            break;
        }
        case Op::Less:
        {
            m_symbol.m_value = args[0]->value() < args[1]->value() ? 1 : 0;
            break;
        }
        case Op::LessEqual:
        {
            m_symbol.m_value = args[0]->value() <= args[1]->value() ? 1 : 0;
            break;
        }
        case Op::Greater:
        {
            m_symbol.m_value = args[0]->value() > args[1]->value() ? 1 : 0;
            break;
        }
        case Op::GreaterEqual:
        {
            m_symbol.m_value = args[0]->value() >= args[1]->value() ? 1 : 0;
            break;
        }
        case Op::Equal:
        {
            m_symbol.m_value = args[0]->value() == args[1]->value() ? 1 : 0;
            break;
        }
        case Op::NotEqual:
        {
            m_symbol.m_value = args[0]->value() != args[1]->value() ? 1 : 0;
            break;
        }
        default:
            ret = false;
            break;
        }
    }
    return ret;
}
bool ExprImpl::equal(ExprPtr other) const
{
    if (other.get() == this || symbol().id() == other->symbol().id())
    {
        return true;
    }
    if (op() == other->op())
    {
        if (op() == ExprImpl::Op::Constant)
        {
            return symbol() == other->symbol();
        }
        else if (op() == ExprImpl::Op::Val)
        {
            return value() == other->value();
        }
        else if (op() == ExprImpl::Op::Branch || op() > ExprImpl::Op::Val)
        {
            if (args().size() == other->args().size())
            {
                for (uint32_t i = 0; i < args().size(); ++i)
                {
                    if (args()[i]->equal(other->args()[i]))
                    {
                        continue;
                    }
                    return false;
                }
                return true;
            }
        }
    }
    return false;
}
bool compare(ExprPtr a, const std::vector<Symbol> &arga,
             ExprPtr b, const std::vector<Symbol> &argb,
             bool relaxed_mode)
{
    if (!relaxed_mode)
    {
        return a->equal(b);
    }
    if (a == b || a->symbol().id() == b->symbol().id())
    {
        return true;
    }
    if (a->op() == b->op())
    {
        if (a->op() == ExprImpl::Op::Constant)
        {
            if (arga.size() != argb.size())
            {
                return false;
            }
            for (uint32_t i = 0; i < arga.size(); ++i)
            {
                if (a->symbol() == arga[i] && b->symbol() == argb[i])
                {
                    return true;
                }
            }
        }
        else if (a->op() == ExprImpl::Op::Val)
        {
            return a->value() == b->value();
        }
        else if (a->op() == ExprImpl::Op::Branch || a->op() > ExprImpl::Op::Val)
        {
            if (a->args().size() == b->args().size())
            {
                for (uint32_t i = 0; i < a->args().size(); ++i)
                {
                    if (compare(a->args()[i], arga, b->args()[i], argb, relaxed_mode))
                    {
                        continue;
                    }
                    return false;
                }
                return true;
            }
        }
    }
    return false;
}
static void expr_to_nodes(ExprPtr expr, std::map<ExprPtr, Node> &ptrs, Dtype d)
{
    auto args = expr->args();
    for (auto arg : args)
    {
        if (!ptrs.count(arg))
        {
            expr_to_nodes(arg, ptrs, d);
        }
    }
    switch (expr->op())
    {
    case ExprImpl::Op::Branch:
    {
        Node mov = NodeImpl::create(Op::mov, d, ptrs[args[2]]);
        Node phi = NodeImpl::create(Op::phi, d, mov);
        If(ptrs[args[0]] != Node(0, d))
        {
            Node mov = NodeImpl::create(Op::mov, d, ptrs[args[1]]);
            phi->add_input(mov);
        }
        EndIf;
        ptrs[expr] = phi;
        break;
    }
    case ExprImpl::Op::Constant:
    {
        ptrs[expr] = NodeImpl::create(Op::alloc, d);
        break;
    }
    case ExprImpl::Op::Val:
    {
        ptrs[expr] = Node((uint32_t)expr->value(), d);
        break;
    }
    case ExprImpl::Op::Negate:
    {
        ptrs[expr] = Node::sub(Node(0, d), ptrs[args[0]], d);
        break;
    }
    case ExprImpl::Op::Reciprocal:
    {
        ptrs[expr] = Node::div(Node(1, d), ptrs[args[0]], d);
        break;
    }
    case ExprImpl::Op::Invert:
    {
        ptrs[expr] = NodeImpl::create(Op::bxor, d, ptrs[args[0]], Node(-1, d));
        break;
    }
    case ExprImpl::Op::Ceil:
    {
        if (args[0]->op() == ExprImpl::Op::Reciprocal || args[0]->op() == ExprImpl::Op::Div)
        {
            Node ins = ptrs[args[0]];
            // construct float division
            auto inps = ins->inputs();
            for (uint32_t i = 0; i < inps.size(); ++i)
            {
                Node mov = NodeImpl::create(Op::mov, Dtype::f32, inps[i]);
                inps[i] = NodeImpl::create(Op::itof, Dtype::f32, mov);
            }
            Node div = Node::div(inps[0], inps[1], Dtype::f32);
            // compute ceil(x)
            Node mov = NodeImpl::create(Op::mov, Dtype::f32, ins);
            Node itof = NodeImpl::create(Op::itof, Dtype::f32, mov);
            mov = NodeImpl::create(Op::mov, d, ins);
            Node phi = NodeImpl::create(Op::phi, d, mov);
            If(div != itof)
            {
                If(div > Node(0, Dtype::f32))
                {
                    Node add = Node::add(ins, Node(1, d), d);
                    phi->add_input(add);
                }
                EndIf;
            }
            EndIf;
            ptrs[expr] = phi;
        }
        else
        {
            ptrs[expr] = ptrs[args[0]];
        }
        break;
    }
    case ExprImpl::Op::Floor:
    {
        // (a * float).floor()
        if (args[0]->op() == ExprImpl::Op::Mul && !args[0]->args()[1]->is_integral_constant())
        {
            Node ins = ptrs[args[0]];
            // construct float multiplication
            auto inp0 = ins->inputs()[0];
            auto inp1 = ins->inputs()[1];
            Node mov = NodeImpl::create(Op::mov, Dtype::f32, inp0);
            inp0 = NodeImpl::create(Op::itof, Dtype::f32, mov);
            inp1 = Node((float)args[0]->args()[1]->value(), Dtype::f32);
            Node mul = Node::mul(inp0, inp1, Dtype::f32);
            // compute floor(x)
            // TODO: support negative input
            Node ftoi = NodeImpl::create(Op::ftoi, Dtype::f32, mul);
            ptrs[expr] = NodeImpl::create(Op::mov, d, ftoi);
        }
        else
        {
            ptrs[expr] = ptrs[args[0]];
        }
        break;
    }
    case ExprImpl::Op::Round:
    {
        if (args[0]->op() == ExprImpl::Op::Reciprocal || args[0]->op() == ExprImpl::Op::Div)
        {
            Node ins = ptrs[args[0]];
            // construct float division
            auto inps = ins->inputs();
            for (uint32_t i = 0; i < inps.size(); ++i)
            {
                Node mov = NodeImpl::create(Op::mov, Dtype::f32, inps[i]);
                inps[i] = NodeImpl::create(Op::itof, Dtype::f32, mov);
            }
            Node div = Node::div(inps[0], inps[1], Dtype::f32);
            // compute round(x)
            Node mov = NodeImpl::create(Op::mov, Dtype::f32, ins);
            Node itof = NodeImpl::create(Op::itof, Dtype::f32, mov);
            Node one(1, d), half(0.5f, Dtype::f32);
            mov = NodeImpl::create(Op::mov, d, ins);
            Node phi = NodeImpl::create(Op::phi, d, mov);
            If(div >= Node(0, Dtype::f32))
            {
                Node sub = Node::sub(div, itof, Dtype::f32);
                If(sub >= half)
                {
                    Node pos = Node::add(ins, one, d);
                    phi->add_input(pos);
                }
                EndIf;
            }
            Else
            {
                Node sub = Node::sub(itof, div, Dtype::f32);
                If(sub >= half)
                {
                    Node neg = Node::sub(ins, one, d);
                    phi->add_input(neg);
                }
                EndIf;
            }
            EndIf;
            ptrs[expr] = phi;
        }
        else
        {
            ptrs[expr] = ptrs[args[0]];
        }
        break;
    }
    case ExprImpl::Op::RoundInt:
    {
        // (a / b).rint()
        if (args[0]->op() == ExprImpl::Op::Reciprocal || args[0]->op() == ExprImpl::Op::Div)
        {
            Node ins = ptrs[args[0]];
            // construct float division
            auto inps = ins->inputs();
            for (uint32_t i = 0; i < inps.size(); ++i)
            {
                Node mov = NodeImpl::create(Op::mov, Dtype::f32, inps[i]);
                inps[i] = NodeImpl::create(Op::itof, Dtype::f32, mov);
            }
            Node div = Node::div(inps[0], inps[1], Dtype::f32);
            // compute rint(x)
            Node rint = NodeImpl::create(Op::rint, Dtype::f32, div);
            Node ftoi = NodeImpl::create(Op::ftoi, Dtype::f32, rint);
            ptrs[expr] = NodeImpl::create(Op::mov, d, ftoi);
        }
        // (a * float).rint()
        else if (args[0]->op() == ExprImpl::Op::Mul && !args[0]->args()[1]->is_integral_constant())
        {
            Node ins = ptrs[args[0]];
            // construct float multiplication
            auto inp0 = ins->inputs()[0];
            auto inp1 = ins->inputs()[1];
            Node mov = NodeImpl::create(Op::mov, Dtype::f32, inp0);
            inp0 = NodeImpl::create(Op::itof, Dtype::f32, mov);
            inp1 = Node((float)args[0]->args()[1]->value(), Dtype::f32);
            Node mul = Node::mul(inp0, inp1, Dtype::f32);
            // compute rint(x)
            Node rint = NodeImpl::create(Op::rint, Dtype::f32, mul);
            Node ftoi = NodeImpl::create(Op::ftoi, Dtype::f32, rint);
            ptrs[expr] = NodeImpl::create(Op::mov, d, ftoi);
        }
        else
        {
            ptrs[expr] = ptrs[args[0]];
        }
        break;
    }
    case ExprImpl::Op::Itof:
    {
        Node ins = ptrs[args[0]];
        Node mov = NodeImpl::create(Op::mov, Dtype::f32, ins);
        ins = NodeImpl::create(Op::itof, Dtype::f32, mov);
        ptrs[expr] = NodeImpl::create(Op::mov, d, ins);
        break;
    }
    case ExprImpl::Op::Log2:
    {
        Node ins = ptrs[args[0]];
        Node mov = NodeImpl::create(Op::mov, Dtype::f32, ins);
        ins = NodeImpl::create(Op::itof, Dtype::f32, mov);
        // compute log2(x)
        Node log2 = NodeImpl::create(Op::log2, Dtype::f32, ins);
        ptrs[expr] = NodeImpl::create(Op::mov, d, log2);
        break;
    }
    case ExprImpl::Op::Sqrt:
    case ExprImpl::Op::Rsqrt:
    {
        auto arg = args[0];
        Node ins = ptrs[arg];
        // x.itof().sqrt/rsqrt() -> float result
        // x.sqrt/rsqrt() -> int result
        Node inp = NodeImpl::create(Op::mov, Dtype::f32, ins);
        if (arg->op() != ExprImpl::Op::Itof)
        {
            inp = NodeImpl::create(Op::itof, Dtype::f32, inp);
        }
        // compute sqrt/rsqrt(x)
        Node res = nullptr;
        if (expr->op() == ExprImpl::Op::Sqrt)
        {
            res = NodeImpl::create(Op::sqrt, Dtype::f32, inp);
        }
        else
        {
            res = NodeImpl::create(Op::rsqrt, Dtype::f32, inp);
        }
        if (arg->op() != ExprImpl::Op::Itof)
        {
            res = NodeImpl::create(Op::ftoi, Dtype::f32, res);
        }
        ptrs[expr] = NodeImpl::create(Op::mov, d, res);
        break;
    }
    case ExprImpl::Op::Max:
    case ExprImpl::Op::Min:
    {
        auto ins = ptrs[args[0]];
        for (uint32_t i = 1; i < args.size(); ++i)
        {
            if (expr->op() == ExprImpl::Op::Max)
            {
                ins = Node::max(ins, ptrs[args[i]]);
            }
            else
            {
                ins = Node::min(ins, ptrs[args[i]]);
            }
        }
        ptrs[expr] = ins;
        break;
    }
    case ExprImpl::Op::Production:
    {
        auto ins = ptrs[args[0]];
        for (uint32_t i = 1; i < args.size(); ++i)
        {
            ins = Node::mul(ins, ptrs[args[i]], d);
        }
        ptrs[expr] = ins;
        break;
    }
    case ExprImpl::Op::Pow:
    {
        int p = args[1]->value();
        auto ins = Node(1, d);
        for (int i = 0; i < std::abs(p); i++)
        {
            ins = Node::mul(ins, ptrs[args[0]], d);
        }
        if (p < 0)
        {
            ins = Node::div(Node(1, d), ins, d);
        }
        ptrs[expr] = ins;
        break;
    }
    case ExprImpl::Op::Add:
    {
        ptrs[expr] = Node::add(ptrs[args[0]], ptrs[args[1]], d);
        break;
    }
    case ExprImpl::Op::Sub:
    {
        ptrs[expr] = Node::sub(ptrs[args[0]], ptrs[args[1]], d);
        break;
    }
    case ExprImpl::Op::Mul:
    {
        ptrs[expr] = Node::mul(ptrs[args[0]], ptrs[args[1]], d);
        break;
    }
    case ExprImpl::Op::Mulh:
    {
        ptrs[expr] = Node::mulh(ptrs[args[0]], ptrs[args[1]], d);
        break;
    }
    case ExprImpl::Op::Div:
    {
        ptrs[expr] = Node::div(ptrs[args[0]], ptrs[args[1]], d);
        break;
    }
    case ExprImpl::Op::FloorDiv:
    {
        ptrs[expr] = Node::div(ptrs[args[0]], ptrs[args[1]], d);
        break;
    }
    case ExprImpl::Op::Mod:
    {
        ptrs[expr] = Node::mod(ptrs[args[0]], ptrs[args[1]], d);
        break;
    }
    case ExprImpl::Op::LeftShift:
    {
        ptrs[expr] = NodeImpl::create(Op::lshit, d, ptrs[args[0]], ptrs[args[1]]);
        break;
    }
    case ExprImpl::Op::RightShift:
    {
        auto arg0 = args[0];
        auto arg1 = args[1];
        if (arg1->op() == ExprImpl::Op::Val && (int64_t)arg1->value() == 32)
        {
            // a >> 32 -> 0
            ptrs[expr] = Node(0, d);
        }
        else
        {
            ptrs[expr] = NodeImpl::create(Op::rshit, d, ptrs[arg0], ptrs[arg1]);
        }
        break;
    }
    case ExprImpl::Op::And:
    {
        auto arg0 = args[0];
        auto arg1 = args[1];
        if (arg1->op() == ExprImpl::Op::Val && (int64_t)arg1->value() == 0xffffffff)
        {
            ptrs[expr] = ptrs[arg0];
        }
        else
        {
            ptrs[expr] = NodeImpl::create(Op::band, d, ptrs[arg0], ptrs[arg1]);
        }
        break;
    }
    case ExprImpl::Op::Or:
    {
        ptrs[expr] = NodeImpl::create(Op::bor, d, ptrs[args[0]], ptrs[args[1]]);
        break;
    }
    case ExprImpl::Op::Xor:
    {
        ptrs[expr] = NodeImpl::create(Op::bxor, d, ptrs[args[0]], ptrs[args[1]]);
        break;
    }
    case ExprImpl::Op::Less:
    {
        ptrs[expr] = ptrs[args[0]] < ptrs[args[1]];
        break;
    }
    case ExprImpl::Op::LessEqual:
    {
        ptrs[expr] = ptrs[args[0]] <= ptrs[args[1]];
        break;
    }
    case ExprImpl::Op::Greater:
    {
        ptrs[expr] = ptrs[args[0]] > ptrs[args[1]];
        break;
    }
    case ExprImpl::Op::GreaterEqual:
    {
        ptrs[expr] = ptrs[args[0]] >= ptrs[args[1]];
        break;
    }
    case ExprImpl::Op::Equal:
    {
        ptrs[expr] = ptrs[args[0]] == ptrs[args[1]];
        break;
    }
    case ExprImpl::Op::NotEqual:
    {
        ptrs[expr] = ptrs[args[0]] != ptrs[args[1]];
        break;
    }
    default:
        break;
    }
}
Function expr_to_func(Expr expr, const std::string &func_name)
{
    Dtype d = Dtype::i32;
    std::map<ExprPtr, Node> ptrs;
    Def(func_name, d, func)
    {
        expr_to_nodes(expr, ptrs, d);
        NodeImpl::create(Op::ret, ptrs[expr]);
    }
    EndDef;
    // adjust order of alloc nodes to match symbols
    auto vars = expr->variables();
    std::vector<NodePtr> body(vars.size(), nullptr);
    for (auto &kv : ptrs)
    {
        if (kv.first->op() == ExprImpl::Op::Constant && kv.second->op() == Op::alloc)
        {
            auto it = std::find_if(vars.begin(), vars.end(), [&kv](Symbol &s) -> bool {
                return s == kv.first->symbol();
            });
            if (it != vars.end())
            {
                int idx = it - vars.begin();
                if (body[idx]) // merge duplicated alloc nodes
                {
                    auto consumers = kv.second->consumers();
                    for (auto c : consumers)
                    {
                        if (!c.expired())
                        {
                            auto n = c.lock();
                            n->replace_input(kv.second, body[idx]);
                        }
                    }
                }
                else
                {
                    body[idx] = kv.second;
                }
            }
        }
    }
    for (auto ins : func->body())
    {
        if (ins->op() != Op::alloc)
        {
            body.push_back(ins);
        }
    }
    func->body() = std::move(body);
    return func;
}
static uint32_t get_variable_id()
{
    static uint32_t id = 0;
    return id++;
}
static void expr_to_cl_code(ExprPtr expr,
                            const std::map<int, std::string> &func_args,
                            std::map<ExprPtr, std::string> &func_vars,
                            std::ostringstream &oss)
{
    static const std::map<ExprImpl::Op, std::string> op_map = {
        {ExprImpl::Op::Ceil, "ceil"},
        {ExprImpl::Op::Round, "round"},
        {ExprImpl::Op::RoundInt, "rint"},
        {ExprImpl::Op::Log2, "log2"},
        {ExprImpl::Op::Sqrt, "sqrt"},
        {ExprImpl::Op::Rsqrt, "rsqrt"},
        {ExprImpl::Op::Max, "max"},
        {ExprImpl::Op::Min, "min"},
        {ExprImpl::Op::Pow, "pow"},
        {ExprImpl::Op::Mulh, "mul_hi"},
        {ExprImpl::Op::Add, "+"},
        {ExprImpl::Op::Sub, "-"},
        {ExprImpl::Op::Mul, "*"},
        {ExprImpl::Op::Div, "/"},
        {ExprImpl::Op::FloorDiv, "/"},
        {ExprImpl::Op::Mod, "%"},
        {ExprImpl::Op::LeftShift, "<<"},
        {ExprImpl::Op::RightShift, ">>"},
        {ExprImpl::Op::And, "&"},
        {ExprImpl::Op::Or, "|"},
        {ExprImpl::Op::Xor, "^"},
        {ExprImpl::Op::Less, "<"},
        {ExprImpl::Op::LessEqual, "<="},
        {ExprImpl::Op::Greater, ">"},
        {ExprImpl::Op::GreaterEqual, ">="},
        {ExprImpl::Op::Equal, "=="},
        {ExprImpl::Op::NotEqual, "!="},
    };

    for (auto &kv : func_vars)
    {
        if (expr->equal(kv.first))
        {
            func_vars[expr] = kv.second;
            return;
        }
    }
    auto op = expr->op();
    auto args = expr->args();
    for (auto arg : args)
    {
        if (!func_vars.count(arg))
        {
            expr_to_cl_code(arg, func_args, func_vars, oss);
        }
    }
    auto var = "var" + std::to_string(get_variable_id());
    switch (op)
    {
    case ExprImpl::Op::Branch:
    {
        oss << "uint " << var << " = " << func_vars[args[0]] << " != 0 ? "
            << func_vars[args[1]] << " : " << func_vars[args[2]] << ";\n";
        func_vars[expr] = var;
        break;
    }
    case ExprImpl::Op::Constant:
    {
        func_vars[expr] = func_args.at(expr->symbol().id());
        break;
    }
    case ExprImpl::Op::Val:
    {
        func_vars[expr] = "((uint)" + std::to_string((uint32_t)expr->value()) + ")";
        break;
    }
    case ExprImpl::Op::Negate:
    {
        oss << "uint " << var << " = -" << func_vars[args[0]] << ";\n";
        func_vars[expr] = var;
        break;
    }
    case ExprImpl::Op::Reciprocal:
    {
        oss << "uint " << var << " = 1 / " << func_vars[args[0]] << ";\n";
        func_vars[expr] = var;
        break;
    }
    case ExprImpl::Op::Invert:
    {
        oss << "uint " << var << " = " << func_vars[args[0]] << " ^ 0xffffffff;\n";
        func_vars[expr] = var;
        break;
    }
    case ExprImpl::Op::Ceil:
    case ExprImpl::Op::Round:
    case ExprImpl::Op::RoundInt:
    {
        // (a / b).ceil/round/rint()
        if (args[0]->op() == ExprImpl::Op::Reciprocal || args[0]->op() == ExprImpl::Op::Div)
        {
            auto fn = op_map.at(op);
            auto inp0 = args[0]->args()[0];
            auto inp1 = args[0]->args()[1];
            oss << "uint " << var << " = " << fn << "((float)" << func_vars[inp0] << " / (float)" << func_vars[inp1] << ");\n";
            func_vars[expr] = var;
        }
        // (a * float).ceil/round/rint()
        else if (args[0]->op() == ExprImpl::Op::Mul && !args[0]->args()[1]->is_integral_constant())
        {
            auto fn = op_map.at(op);
            auto inp0 = args[0]->args()[0];
            float inp1 = args[0]->args()[1]->value();
            oss << "uint " << var << " = " << fn << "((float)" << func_vars[inp0] << " * (float)" << inp1 << ");\n";
            func_vars[expr] = var;
        }
        else
        {
            func_vars[expr] = func_vars[args[0]];
        }
        break;
    }
    case ExprImpl::Op::Itof:
    {
        auto tmp = "var" + std::to_string(get_variable_id());
        oss << "float " << tmp << " = (float)" << func_vars[args[0]] << ";\n";
        oss << "uint " << var << " = *((uint *)&" << tmp << ");\n";
        func_vars[expr] = var;
        break;
    }
    case ExprImpl::Op::Log2:
    {
        auto fn = op_map.at(op);
        oss << "uint " << var << " = " << fn << "((float)" << func_vars[args[0]] << ");\n";
        func_vars[expr] = var;
        break;
    }
    case ExprImpl::Op::Sqrt:
    case ExprImpl::Op::Rsqrt:
    {
        auto fn = op_map.at(op);
        // x.itof().sqrt/rsqrt() -> float result
        if (args[0]->op() == ExprImpl::Op::Itof)
        {
            auto tmp = "var" + std::to_string(get_variable_id());
            oss << "float " << tmp << " = " << fn << "((float)" << func_vars[args[0]->args()[0]] << ");\n";
            oss << "uint " << var << " = *((uint *)&" << tmp << ");\n";
        }
        // x.sqrt/rsqrt() -> int result
        else
        {
            oss << "uint " << var << " = " << fn << "((float)" << func_vars[args[0]] << ");\n";
        }
        func_vars[expr] = var;
        break;
    }
    case ExprImpl::Op::Floor:
    {
        // (a * float).floor()
        if (args[0]->op() == ExprImpl::Op::Mul && !args[0]->args()[1]->is_integral_constant())
        {
            auto inp0 = args[0]->args()[0];
            float inp1 = args[0]->args()[1]->value();
            oss << "uint " << var << " = floor((float)" << func_vars[inp0] << " * (float)" << inp1 << ");\n";
            func_vars[expr] = var;
        }
        else
        {
            func_vars[expr] = func_vars[args[0]];
        }
        break;
    }
    case ExprImpl::Op::Max:
    case ExprImpl::Op::Min:
    {
        oss << "uint " << var << " = " << func_vars[args[0]] << ";\n";
        auto fn = op_map.at(op);
        for (uint32_t i = 1; i < args.size(); ++i)
        {
            oss << var << " = " << fn << "(" << func_vars[args[i]] << ", " << var << ");\n";
        }
        func_vars[expr] = var;
        break;
    }
    case ExprImpl::Op::Production:
    {
        oss << "uint " << var << " = " << func_vars[args[0]];
        for (uint32_t i = 1; i < args.size(); ++i)
        {
            oss << " * " << func_vars[args[i]];
        }
        oss << ";\n";
        func_vars[expr] = var;
        break;
    }
    case ExprImpl::Op::Pow:
    {
        float p = args[1]->value();
        oss << "uint " << var << " = pow((float)" << func_vars[args[0]] << ", (float)" << p << ");\n";
        func_vars[expr] = var;
        break;
    }
    case ExprImpl::Op::Mulh:
    {
        oss << "uint " << var << " = mul_hi(" << func_vars[args[0]] << ", " << func_vars[args[1]] << ");\n";
        func_vars[expr] = var;
        break;
    }
    case ExprImpl::Op::Add:
    case ExprImpl::Op::Sub:
    case ExprImpl::Op::Mul:
    case ExprImpl::Op::Div:
    case ExprImpl::Op::FloorDiv:
    case ExprImpl::Op::Mod:
    case ExprImpl::Op::LeftShift:
    case ExprImpl::Op::RightShift:
    case ExprImpl::Op::And:
    case ExprImpl::Op::Or:
    case ExprImpl::Op::Xor:
    case ExprImpl::Op::Less:
    case ExprImpl::Op::LessEqual:
    case ExprImpl::Op::Greater:
    case ExprImpl::Op::GreaterEqual:
    case ExprImpl::Op::Equal:
    case ExprImpl::Op::NotEqual:
    {
        if (op == ExprImpl::Op::And &&
            args[1]->op() == ExprImpl::Op::Val && (int64_t)args[1]->value() == 0xffffffff)
        {
            func_vars[expr] = func_vars[args[0]];
        }
        else
        {
            auto fn = op_map.at(op);
            oss << "uint " << var << " = " << func_vars[args[0]] << " " << fn << " " << func_vars[args[1]] << ";\n";
            func_vars[expr] = var;
        }
        break;
    }
    default:
        break;
    }
}
std::vector<std::string> expr_to_cl_code(Expr expr,
                                         const std::map<int, std::string> &func_args,
                                         std::map<ExprPtr, std::string> &func_vars,
                                         std::string &res)
{
    std::ostringstream oss;
    expr_to_cl_code(expr, func_args, func_vars, oss);
    res = func_vars.at(expr);
    return utils::split_string(oss.str(), "\n");
}
std::vector<std::string> expr_to_cl_code(Expr expr, std::string &res)
{
    std::map<int, std::string> func_args;
    std::map<ExprPtr, std::string> func_vars;
    auto vars = expr->variables();
    for (auto &&var : vars)
    {
        func_args[var.id()] = var.name();
    }
    return expr_to_cl_code(expr, func_args, func_vars, res);
}
std::string expr_to_cl_func(Expr expr, const std::string &func_name)
{
    std::ostringstream oss;
    auto args = expr->variables();

    oss << "// " << expr->to_string() << "\n"
        << "uint " << func_name << "(";
    for (size_t i = 0; i < args.size(); ++i)
    {
        oss << "uint " << args[i].name();
        if (i != args.size() - 1)
        {
            oss << ", ";
        }
    }
    oss << ")\n"
        << "{\n";

    auto indent = std::string(4, ' ');
    std::string res;
    auto codes = expr_to_cl_code(expr, res);
    for (auto &&c : codes)
    {
        oss << indent << c << "\n";
    }

    oss << indent << "return " << res << ";\n"
        << "}\n";
    return oss.str();
}
} // namespace tongue
