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
 * @Date: 2023-04-03 17:08:47 
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-08-08 11:45:20
 */
#pragma once
#include "ins.h"
#include "op.h"
#include "reg.h"
#include <algorithm>
#include <bitset>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

/*
the simple pass for this backend
IR->ISel->RA->Emit

stack layout

use r0-r7 for arg register, caller managed the arg8-argn

============
upper frame1
============   <- upper frame pointer
upper frame0
-.-.-.-.-.-.
arg 8
------------
arg 9
------------
......
arg n
============
ret reg:r30
------------
saved base pointer
------------   <- stack pointer (SP)
local 0
------------
local 1
------------
......
local n
------------   <- stack pointer (SP)

first stage:
implement simple scalar instruction
second stage:
implement vadd example via SRAM and TPC instruction

do we need a compiletime stack for stack optimization?
*/

#define If(c) tongue::sbe::if_((c),[&](){
#define Else \
    }        \
    , [&]() {
#define EndIf \
    });

#define For(i, c, u) tongue::sbe::for_((i),(c),(u),[&](){
#define EndFor \
    });

#define Def(n, d, f)                                      \
    tongue::sbe::Function f = nullptr;                    \
    {                                                     \
        auto self = tongue::sbe::FunctionImpl::def(n, d); \
        f = self.m_func;
#define EndDef \
    }

namespace tongue
{
namespace sbe
{
    class NodeImpl;
    class Node;
    using NodePtr = std::shared_ptr<NodeImpl>;
    using NodeRef = std::weak_ptr<NodeImpl>;
    class NodeImpl : public std::enable_shared_from_this<NodeImpl>
    {
    public:
        NodeImpl() = delete;
        virtual ~NodeImpl() = default;

        static std::vector<NodePtr> &collections()
        {
            static std::vector<NodePtr> m_collections;
            return m_collections;
        }

        template <typename... Args>
        static NodePtr create(Op op, Dtype d, Args... args)
        {
            auto ret = NodePtr(new NodeImpl(op, d));
            ret->add_inputs(std::forward<Args>(args)...);
            NodeImpl::collections().push_back(ret);
            return ret;
        }
        template <typename... Args>
        static NodePtr create(Op op, Args... args)
        {
            return create(op, Dtype::none, std::forward<Args>(args)...);
        }
        template <typename T, typename std::enable_if<std::is_integral<T>::value &&
                                                      std::is_unsigned<T>::value>::type * = nullptr>
        static NodePtr create(T v, Dtype d = Dtype::u32)
        {
            return constant(v, d);
        }
        template <typename T, typename std::enable_if<std::is_integral<T>::value &&
                                                      std::is_signed<T>::value>::type * = nullptr>
        static NodePtr create(T v, Dtype d = Dtype::i32)
        {
            return constant(v, d);
        }
        template <typename T, typename std::enable_if<std::is_floating_point<T>::value>::type * = nullptr>
        static NodePtr create(T v, Dtype d = Dtype::f32)
        {
            return constant(v, d);
        }

        template <typename T, typename std::enable_if<std::is_unsigned<T>::value>::type * = nullptr>
        static NodePtr constant(T v, Dtype d)
        {
            auto ret = NodeImpl::create(Op::constant, d);
            ret->m_value = v;
            return ret;
        }
        template <typename T, typename std::enable_if<std::is_signed<T>::value>::type * = nullptr>
        static NodePtr constant(T v, Dtype d)
        {
            uint32_t c = 0;
            if (d == Dtype::f32)
            {
                float t = v;
                c = *(uint32_t *)(&t);
            }
            else
            {
                int t = v;
                c = *(uint32_t *)(&t);
            }
            return constant(c, d);
        }

        virtual NodePtr clone() const
        {
            auto ret = create(m_op, m_dtype);
            ret->m_reg = m_reg;
            ret->m_name = m_name;
            ret->m_value = m_value;
            for (auto &&inp : m_inputs)
            {
                ret->add_input(inp);
            }
            ret->m_imms = m_imms;
            return ret;
        }

        void add_input(NodePtr inp)
        {
            m_inputs.push_back(inp);
            inp->m_consumers.push_back(shared_from_this());
        }
        template <typename... Args>
        void add_inputs(Args... args)
        {
            std::vector<NodePtr> inps = {args...};
            for (auto &&inp : inps)
            {
                add_input(inp);
            }
        }

        void remove_input(NodePtr inp)
        {
            auto it = std::find(m_inputs.begin(), m_inputs.end(), inp);
            if (it != m_inputs.end())
            {
                inp->remove_consumer(shared_from_this());
                m_inputs.erase(it);
            }
        }
        void remove_inputs()
        {
            for (auto &&inp : m_inputs)
            {
                inp->remove_consumer(shared_from_this());
            }
            m_inputs.clear();
        }

        void replace_input(NodePtr replaced, NodePtr by)
        {
            auto it = std::find(m_inputs.begin(), m_inputs.end(), replaced);
            if (it != m_inputs.end())
            {
                int idx = it - m_inputs.begin();
                m_inputs[idx] = by;
                replaced->remove_consumer(shared_from_this());
                by->m_consumers.push_back(shared_from_this());
            }
        }

        virtual std::string to_string() const
        {
            std::ostringstream oss;
            if (m_op == Op::label)
            {
                return to_ssa();
            }
            else if (m_op == Op::constant)
            {
                oss << " %" << m_id << " " << m_dtype << " = #" << m_value;
            }
            else if (m_op == Op::ret)
            {
                oss << "ret";
                if (m_inputs.size())
                {
                    oss << " %" << m_inputs[0]->m_id << " " << m_inputs[0]->m_dtype;
                }
            }
            else
            {
                oss << " %" << m_id << " " << m_dtype << " = " << m_op << " ";
                for (uint32_t i = 0; i < m_inputs.size(); ++i)
                {
                    if (i != 0)
                    {
                        oss << ",";
                    }
                    oss << m_inputs[i]->to_ssa();
                }
            }
            return oss.str();
        }

        static NodePtr push(NodePtr ins)
        {
            Dtype d = Dtype::u32;
            if (ins->dtype() == Dtype::f32)
            {
                d = Dtype::f32;
            }
            return create(Op::push, d, ins);
        }
        static NodePtr push(const Reg &reg, Dtype d = Dtype::u32)
        {
            if (d != Dtype::f32)
            {
                d = Dtype::u32;
            }
            NodePtr n = create(Op::alloc, d);
            n->m_reg = reg;
            return create(Op::push, d, n);
        }

        static NodePtr pop(NodePtr ins)
        {
            Dtype d = Dtype::u32;
            if (ins->dtype() == Dtype::f32)
            {
                d = Dtype::f32;
            }
            NodePtr n = create(Op::pop, d);
            n->m_reg = ins->m_reg;
            return n;
        }
        static NodePtr pop(const Reg &reg, Dtype d = Dtype::u32)
        {
            if (d != Dtype::f32)
            {
                d = Dtype::u32;
            }
            NodePtr n = create(Op::pop, d);
            n->m_reg = reg;
            return n;
        }

        uint32_t id() const { return m_id; }
        void set_id(uint32_t id) { m_id = id; }
        Op op() const { return m_op; }
        void set_op(Op op) { m_op = op; }
        Dtype dtype() const { return m_dtype; }
        void set_dtype(Dtype dtype) { m_dtype = dtype; }
        const Reg &reg() const { return m_reg; }
        Reg &reg() { return m_reg; }
        void set_reg(const Reg &reg) { m_reg = reg; }
        const std::string &name() const { return m_name; }
        void set_name(const std::string &name) { m_name = name; }
        uint32_t value() const { return m_value; }
        void set_value(uint32_t value) { m_value = value; }
        const std::vector<NodePtr> &inputs() const { return m_inputs; }
        const std::vector<NodeRef> &consumers() const { return m_consumers; }
        const std::vector<uint32_t> &imms() const { return m_imms; }
        std::vector<uint32_t> &imms() { return m_imms; }
        const InsPackages &inss() const { return m_inss; }
        InsPackages &inss() { return m_inss; }

    protected:
        NodeImpl(Op op, Dtype d)
            : m_op(op), m_dtype(d), m_reg(Reg(0, Operand::NONE, 32))
        {
            static uint32_t gid = 0;
            m_id = gid++;
        }

        std::string to_ssa() const
        {
            if (m_op == Op::label)
            {
                if (m_name.size())
                {
                    return m_name;
                }
                return ".LBL_" + std::to_string(m_id);
            }
            return "%" + std::to_string(m_id);
        }

        void remove_consumer(NodePtr c)
        {
            for (uint32_t i = 0; i < m_consumers.size(); i++)
            {
                if (m_consumers[i].lock() == c)
                {
                    m_consumers.erase(m_consumers.begin() + i);
                    break;
                }
            }
        }

        uint32_t m_id;
        Op m_op;
        Dtype m_dtype;
        Reg m_reg;
        std::string m_name;
        uint32_t m_value;
        std::vector<NodePtr> m_inputs;
        std::vector<NodeRef> m_consumers;
        std::vector<uint32_t> m_imms;
        InsPackages m_inss;
        friend class Node;
    };

    class Node
    {
    public:
        Node() : m_ptr(nullptr) {}
        Node(std::nullptr_t p) : m_ptr(p) {}
        Node(NodePtr p) : m_ptr(p) {}
        template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
        Node(T v, Dtype d) : m_ptr(NodeImpl::create(v, d)) {}
        template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
        Node(T v) : m_ptr(NodeImpl::create(v)) {}

        NodeImpl &operator*()
        {
            return *m_ptr;
        }

        const NodeImpl &operator*() const
        {
            return *m_ptr;
        }

        NodeImpl *operator->()
        {
            return m_ptr.get();
        }

        const NodeImpl *operator->() const
        {
            return m_ptr.get();
        }

        operator NodePtr() const
        {
            return m_ptr;
        }

        operator bool() const
        {
            return m_ptr != nullptr;
        }

        static bool check_dtype(Node a, Node b)
        {
            bool ret = false;
            if ((INT_DTYPE.count(a->dtype()) && INT_DTYPE.count(b->dtype())) ||
                (UINT_DTYPE.count(a->dtype()) && UINT_DTYPE.count(b->dtype())) ||
                (a->dtype() == Dtype::f32 && b->dtype() == Dtype::f32))
            {
                ret = true;
            }
            return ret;
        }

        static Node add(Node a, Node b, Dtype d)
        {
            return NodeImpl::create(Op::add, d, a, b);
        }
        static Node iadd(Node a, Node b)
        {
            Node phi = NodeImpl::create(Op::phi, a->m_dtype, a);
            phi->add_input(add(phi, b, a->m_dtype));
            return phi;
        }
        static Node sub(Node a, Node b, Dtype d)
        {
            return NodeImpl::create(Op::sub, d, a, b);
        }
        static Node isub(Node a, Node b)
        {
            Node phi = NodeImpl::create(Op::phi, a->m_dtype, a);
            phi->add_input(sub(phi, b, a->m_dtype));
            return phi;
        }
        static Node mul(Node a, Node b, Dtype d)
        {
            return NodeImpl::create(Op::mul, d, a, b);
        }
        static Node imul(Node a, Node b)
        {
            Node phi = NodeImpl::create(Op::phi, a->m_dtype, a);
            phi->add_input(mul(phi, b, a->m_dtype));
            return phi;
        }
        static Node mulh(Node a, Node b, Dtype d)
        {
            return NodeImpl::create(Op::mulh, d, a, b);
        }
        static Node div(Node a, Node b, Dtype d)
        {
            return NodeImpl::create(Op::div, d, a, b);
        }
        static Node idiv(Node a, Node b)
        {
            Node phi = NodeImpl::create(Op::phi, a->m_dtype, a);
            phi->add_input(div(phi, b, a->m_dtype));
            return phi;
        }
        static Node mod(Node a, Node b, Dtype d)
        {
            return NodeImpl::create(Op::mod, d, a, b);
        }

        static Node max(Node a, Node b)
        {
            aipu_assert(Node::check_dtype(a, b));
            Dtype d = std::max(a->dtype(), b->dtype());
            return NodeImpl::create(Op::max, d, a, b);
        }
        static Node min(Node a, Node b)
        {
            aipu_assert(Node::check_dtype(a, b));
            Dtype d = std::max(a->dtype(), b->dtype());
            return NodeImpl::create(Op::min, d, a, b);
        }

    protected:
        NodePtr m_ptr;
    };

    inline Node operator+(Node a, Node b)
    {
        aipu_assert(Node::check_dtype(a, b));
        Dtype d = std::max(a->dtype(), b->dtype());
        return Node::add(a, b, d);
    }
    inline Node operator+=(Node a, Node b)
    {
        return Node::iadd(a, b);
    }
    inline Node operator-(Node a, Node b)
    {
        aipu_assert(Node::check_dtype(a, b));
        Dtype d = std::max(a->dtype(), b->dtype());
        return Node::sub(a, b, d);
    }
    inline Node operator-=(Node a, Node b)
    {
        return Node::isub(a, b);
    }
    inline Node operator*(Node a, Node b)
    {
        aipu_assert(Node::check_dtype(a, b));
        Dtype d = std::max(a->dtype(), b->dtype());
        return Node::mul(a, b, d);
    }
    inline Node operator*=(Node a, Node b)
    {
        return Node::imul(a, b);
    }
    inline Node operator/(Node a, Node b)
    {
        aipu_assert(Node::check_dtype(a, b));
        Dtype d = std::max(a->dtype(), b->dtype());
        return Node::div(a, b, d);
    }
    inline Node operator/=(Node a, Node b)
    {
        return Node::idiv(a, b);
    }
    inline Node operator%(Node a, Node b)
    {
        aipu_assert(Node::check_dtype(a, b));
        Dtype d = std::max(a->dtype(), b->dtype());
        return Node::mod(a, b, d);
    }

    inline Node operator<(Node a, Node b)
    {
        aipu_assert(Node::check_dtype(a, b));
        return NodeImpl::create(Op::slt, Dtype::i32, a, b);
    }
    inline Node operator<=(Node a, Node b)
    {
        aipu_assert(Node::check_dtype(a, b));
        return NodeImpl::create(Op::sle, Dtype::i32, a, b);
    }
    inline Node operator>(Node a, Node b)
    {
        aipu_assert(Node::check_dtype(a, b));
        return NodeImpl::create(Op::sgt, Dtype::i32, a, b);
    }
    inline Node operator>=(Node a, Node b)
    {
        aipu_assert(Node::check_dtype(a, b));
        return NodeImpl::create(Op::sge, Dtype::i32, a, b);
    }
    inline Node operator==(Node a, Node b)
    {
        aipu_assert(Node::check_dtype(a, b));
        return NodeImpl::create(Op::seq, Dtype::i32, a, b);
    }
    inline Node operator!=(Node a, Node b)
    {
        aipu_assert(Node::check_dtype(a, b));
        return NodeImpl::create(Op::sne, Dtype::i32, a, b);
    }

    class FunctionImpl;
    class Function;
    using FunctionPtr = std::shared_ptr<FunctionImpl>;
    using FunctionRef = std::weak_ptr<FunctionImpl>;
    class FunctionImpl : public NodeImpl
    {
    public:
        struct def
        {
            def(FunctionPtr func)
            {
                m_func = func;
                m_backup = NodeImpl::collections();
                NodeImpl::collections().clear();
            }
            def(const std::string &name, Dtype d)
            {
                m_func = FunctionImpl::create(name, d);
                m_backup = NodeImpl::collections();
                NodeImpl::collections().clear();
            }
            ~def()
            {
                m_func->m_body = NodeImpl::collections();
                NodeImpl::collections() = std::move(m_backup);
            }

            template <typename... Args>
            NodePtr operator()(Args... args) const
            {
                return NodeImpl::create(Op::call, m_func->m_dtype, m_func, std::forward<Args>(args)...);
            }

            FunctionPtr m_func;
            std::vector<NodePtr> m_backup;
        };

        operator NodePtr()
        {
            return std::static_pointer_cast<NodeImpl>(shared_from_this());
        }

        static FunctionPtr create(const std::string &name, Dtype d)
        {
            auto ret = FunctionPtr(new FunctionImpl(name, d));
            NodeImpl::collections().push_back(ret);
            return ret;
        }

        template <typename... Args>
        NodePtr operator()(Args... args)
        {
            return NodeImpl::create(Op::call, m_dtype, shared_from_this(), std::forward<Args>(args)...);
        }

        virtual NodePtr clone() const override
        {
            auto ret = create(m_name, m_dtype);
            std::map<int, NodePtr> nodes;
            for (auto &&ins : m_body)
            {
                auto n = ins->clone();
                nodes[ins->id()] = n;
                ret->m_body.push_back(n);
            }
            for (auto &&ins : ret->m_body)
            {
                auto inps = ins->inputs();
                ins->remove_inputs();
                for (auto &&inp : inps)
                {
                    if (!nodes.count(inp->id()))
                    {
                        if (inp->op() == ret->op() && inp->name() == ret->name())
                        {
                            ins->add_input(ret);
                        }
                        else
                        {
                            ins->add_input(inp);
                        }
                    }
                    else
                    {
                        ins->add_input(nodes[inp->id()]);
                    }
                }
            }
            return ret;
        }

        std::vector<NodePtr> codes() const
        {
            std::vector<NodePtr> ret;
            auto label = NodeImpl::create(Op::label);
            label->set_name(m_name);
            ret.push_back(label);
            std::map<int, NodePtr> used_r_regs, used_f_regs;
            uint32_t num_spill_vars = 0;
            for (auto &&ins : m_body)
            {
                int idx = ins->reg().index();
                if (ins->reg().spilled())
                {
                    if (ins->reg().offset() < 0)
                    {
                        num_spill_vars += 1;
                    }
                }
                else if (ins->reg().is_freg())
                {
                    used_f_regs[idx] = ins;
                }
                else if (ins->reg().is_rreg())
                {
                    used_r_regs[idx] = ins;
                }
            }
            NodePtr sp = nullptr;
            if (num_spill_vars)
            {
                // push local vars [var0, ..., varn]
                sp = NodeImpl::create(Op::alloc, Dtype::u32);
                sp->set_reg(RReg(SP));
                auto sub = NodeImpl::create(Op::sub, Dtype::u32, sp);
                sub->set_reg(sp->reg());
                sub->imms().push_back(4 * num_spill_vars);
                ret.push_back(sp);
                ret.push_back(sub);
            }
            std::vector<NodePtr> pop;
            std::set<int> regs = {R0, FP, SP, RA, R31};
            for (auto &&p : used_r_regs)
            {
                if (!regs.count(p.first) && p.first > R7)
                {
                    ret.push_back(NodeImpl::push(p.second));
                    pop.push_back(NodeImpl::pop(p.second));
                }
            }
            for (auto &&p : used_f_regs)
            {
                ret.push_back(NodeImpl::push(p.second));
                pop.push_back(NodeImpl::pop(p.second));
            }
            std::reverse(pop.begin(), pop.end());
            NodePtr add = nullptr;
            if (num_spill_vars)
            {
                // pop local vars [varn, ..., var0]
                add = NodeImpl::create(Op::add, Dtype::u32, sp);
                add->set_reg(sp->reg());
                add->imms().push_back(4 * num_spill_vars);
            }
            bool no_return = true;
            for (auto &&ins : m_body)
            {
                if (ins->op() == Op::ret)
                {
                    ret.insert(ret.end(), pop.begin(), pop.end());
                    if (num_spill_vars)
                    {
                        ret.push_back(add);
                    }
                    no_return = false;
                }
                ret.push_back(ins);
            }
            if (no_return)
            {
                ret.insert(ret.end(), pop.begin(), pop.end());
                if (num_spill_vars)
                {
                    ret.push_back(add);
                }
                ret.push_back(NodeImpl::create(Op::ret));
            }
            return ret;
        }

        virtual std::string to_string() const override
        {
            std::ostringstream oss;
            oss << "def " << m_name << ":\n";
            for (auto &&ins : m_body)
            {
                oss << ins->to_string() << "\n";
            }
            return oss.str();
        }

        std::vector<NodePtr> &body() { return m_body; }
        const std::vector<NodePtr> &body() const { return m_body; }

    protected:
        FunctionImpl(const std::string &name, Dtype d)
            : NodeImpl(Op::fun, d)
        {
            m_name = name;
        }

        std::vector<NodePtr> m_body;
        friend class Function;
    };

    class Function : public Node
    {
    public:
        Function() : Node(nullptr) {}
        Function(std::nullptr_t p) : Node(p) {}
        Function(FunctionPtr p) : Node(p->operator NodePtr()) {}

        operator FunctionPtr() const
        {
            return std::dynamic_pointer_cast<FunctionImpl>(m_ptr);
        }

        template <typename... Args>
        Node operator()(Args... args) const
        {
            return NodeImpl::create(Op::call, m_ptr->dtype(), m_ptr, std::forward<Args>(args)...);
        }

        FunctionImpl &operator*()
        {
            return *std::dynamic_pointer_cast<FunctionImpl>(m_ptr);
        }

        const FunctionImpl &operator*() const
        {
            return *std::dynamic_pointer_cast<FunctionImpl>(m_ptr);
        }

        FunctionImpl *operator->()
        {
            return std::dynamic_pointer_cast<FunctionImpl>(m_ptr).get();
        }

        const FunctionImpl *operator->() const
        {
            return std::dynamic_pointer_cast<FunctionImpl>(m_ptr).get();
        }
    };

    class LoopWrapper
    {
    public:
        LoopWrapper() = default;
        LoopWrapper(Node init, std::function<Node(Node)> cond, std::function<Node(Node)> update)
            : m_init(init), m_cond(cond), m_update(update) {}

        void init_loop()
        {
            auto nodes = NodeImpl::collections();
            NodeImpl::collections().clear();
            auto it = std::find(nodes.begin(), nodes.end(), m_init.operator NodePtr());
            if (it != nodes.end())
            {
                nodes.erase(it);
            }
            m_backup = std::move(nodes);
            NodeImpl::create(Op::label); // start_loop
            auto init_c = m_init->clone();
            m_loop = NodeImpl::create(Op::label);
            m_index = NodeImpl::create(Op::phi, init_c->dtype(), init_c);
            m_jump_end = NodeImpl::create(Op::cbz, m_cond(m_index));
        }

        void exit_loop()
        {
            auto body = NodeImpl::collections();
            m_backup.insert(m_backup.end(), body.begin(), body.end());
            NodeImpl::collections() = std::move(m_backup);
            m_index->add_input(m_update(m_index));
            // should keep live of global variable
            std::set<int> local_vars, global_vars;
            for (auto &&ins : body)
            {
                local_vars.insert(ins->id());
            }
            std::vector<Node> keep_live_vars;
            for (auto &&ins : body)
            {
                if (ins->op() != Op::phi)
                {
                    for (auto &&inp : ins->inputs())
                    {
                        if (!local_vars.count(inp->id()) && !global_vars.count(inp->id()))
                        {
                            global_vars.insert(inp->id());
                            keep_live_vars.push_back(inp);
                        }
                    }
                }
            }
            auto keep_live = NodeImpl::create(Op::keep_live);
            for (auto &&inp : keep_live_vars)
            {
                keep_live->add_input(inp);
            }
            NodeImpl::create(Op::jmp, m_loop);
            auto end_loop = NodeImpl::create(Op::label);
            m_jump_end->add_input(end_loop);
        }

        Node index() const { return m_index; }

    protected:
        std::vector<NodePtr> m_backup;
        Node m_init;
        std::function<Node(Node)> m_cond;
        std::function<Node(Node)> m_update;
        Node m_loop;
        Node m_index;
        Node m_jump_end;
    };

    /*
    generate a code like:
    loop_init:
    init
    cond
    cbz cond,end_loop
    loop:
    cond=Phi(idxer)
    body
    update
    cond
    cbnz cond,loop
    end_loop:
    */
    inline void for_(Node init, std::function<Node(Node)> cond, std::function<Node(Node)> update, std::function<void()> for_stat)
    {
        auto loop = LoopWrapper(init, cond, update);
        loop.init_loop();
        for_stat();
        loop.exit_loop();
    }

    inline void if_(Node cond, std::function<void()> if_stat, std::function<void()> else_stat = nullptr)
    {
        auto jmp = NodeImpl::create(Op::cbz, cond);
        if_stat();
        if (else_stat)
        {
            auto jmp_end = NodeImpl::create(Op::jmp);
            auto start_else = NodeImpl::create(Op::label);
            jmp->add_input(start_else);
            else_stat();
            auto end = NodeImpl::create(Op::label);
            jmp_end->add_input(end);
        }
        else
        {
            auto end = NodeImpl::create(Op::label);
            jmp->add_input(end);
        }
    }

    // the reg allocator will allocate register for a function
    // so the input is a function: list of instruction
    class RegAllocator
    {
    public:
        RegAllocator(uint32_t r_regs, uint32_t f_regs, bool use_free_args = false)
            : m_r_regs(r_regs), m_f_regs(f_regs), m_use_free_args(use_free_args) {}
        RegAllocator() = delete;

        void allocate(Function func)
        {
            reset_allocator();
            // special for phi and alloc node
            std::vector<NodePtr> phis, args, body;
            for (auto &&ins : func->body())
            {
                if (ins->op() == Op::phi)
                {
                    phis.push_back(ins);
                }
                if (ins->op() == Op::alloc)
                {
                    args.push_back(ins);
                }
                else
                {
                    body.push_back(ins);
                }
            }
            body.insert(body.begin(), args.begin(), args.end());
            std::map<int, NodePtr> shared_regs;
            for (auto &&phi : phis)
            {
                shared_regs[phi->id()] = phi;
                for (auto &&inp : phi->inputs())
                {
                    shared_regs[inp->id()] = phi;
                }
            }
            // liveness analysis: compute liveness_end of each node
            for (uint32_t i = 0; i < body.size(); i++)
            {
                NodePtr n = body[i];
                if (n->op() == Op::alloc) // always keep alloc nodes alive
                {
                    m_liveness[n->id()] = INT32_MAX;
                }
                else
                {
                    for (uint32_t j = 0; j < n->inputs().size(); j++)
                    {
                        m_liveness[n->inputs()[j]->id()] = i;
                    }
                    if (!m_liveness.count(n->id()))
                    {
                        m_liveness[n->id()] = i;
                    }
                }
            }
            for (auto &&phi : phis)
            {
                int lmax = INT32_MIN;
                for (auto &&inp : phi->inputs())
                {
                    lmax = std::max(lmax, m_liveness[inp->id()]);
                }
                m_liveness[phi->id()] = std::max(lmax, m_liveness[phi->id()]);
                for (auto &&inp : phi->inputs())
                {
                    m_liveness[inp->id()] = m_liveness[phi->id()];
                }
            }
            for (uint32_t i = 0; i < body.size(); i++)
            {
                NodePtr n = body[i], phi = nullptr;
                expire_old_intervals(i);
                if (shared_regs.count(n->id()))
                {
                    phi = shared_regs[n->id()];
                }
                bool need_allocate = true;
                if (phi && !phi->reg().is_none())
                {
                    need_allocate = false;
                }
                if (need_allocate)
                {
                    bool need_spill = false;
                    if (n->dtype() == Dtype::f32 && m_f_status.count() == m_f_regs)
                    {
                        need_spill = true;
                    }
                    else if (m_r_status.count() == m_r_regs)
                    {
                        need_spill = true;
                    }
                    if (!need_spill && !allocate_node_to_register(n, args.size()))
                    {
                        need_spill = true;
                    }
                    if (need_spill)
                    {
                        spill_at_interval(n);
                    }
                }
                std::sort(m_active_nodes.begin(), m_active_nodes.end(), [this](NodePtr a, NodePtr b) {
                    return m_liveness[a->id()] > m_liveness[b->id()];
                });
                if (phi)
                {
                    for (auto &&inp : phi->inputs())
                    {
                        inp->set_reg(n->reg());
                    }
                    phi->set_reg(n->reg());
                }
            }
        }

        // allocate temp register for spilled node
        void allocate_temp_register(Node node) const
        {
            std::set<int> r_regs, f_regs;
            auto nodes = node->inputs();
            nodes.push_back(node);
            std::vector<NodePtr> spilled_nodes;
            for (auto &&n : nodes)
            {
                if (n->reg().spilled())
                {
                    spilled_nodes.push_back(n);
                }
                else if (n->reg().is_freg())
                {
                    f_regs.insert(n->reg().index());
                }
                else if (n->reg().is_rreg())
                {
                    r_regs.insert(n->reg().index());
                }
            }
            if (spilled_nodes.size())
            {
                for (auto &&n : spilled_nodes)
                {
                    if (n->reg().is_freg())
                    {
                        for (uint32_t idx = 0; idx < m_f_regs; idx++)
                        {
                            if (!f_regs.count(idx))
                            {
                                n->reg().set_index(idx);
                                f_regs.insert(idx);
                                break;
                            }
                        }
                    }
                    else if (n->reg().is_rreg())
                    {
                        for (uint32_t idx = 0; idx < m_r_regs; idx++)
                        {
                            if (!r_regs.count(idx))
                            {
                                n->reg().set_index(idx);
                                r_regs.insert(idx);
                                break;
                            }
                        }
                    }
                }
            }
        }

        bool spilled() const { return m_spilled; }

    protected:
        void reset_allocator()
        {
            m_spill_args = 0;
            m_spill_vars = 0;
            m_spilled = false;
            m_r_status.reset();
            m_f_status.reset();
            m_liveness.clear();
            m_active_nodes.clear();
        }

        void expire_old_intervals(int liveness)
        {
            std::vector<NodePtr> to_delete_nodes;
            for (auto &&n : m_active_nodes)
            {
                if (m_liveness[n->id()] < liveness)
                {
                    to_delete_nodes.push_back(n);
                }
            }
            for (auto &&n : to_delete_nodes)
            {
                if (n->reg().is_freg())
                {
                    m_f_status.reset(n->reg().index());
                }
                else
                {
                    m_r_status.reset(n->reg().index());
                }
                auto it = std::find(m_active_nodes.begin(), m_active_nodes.end(), n);
                if (it != m_active_nodes.end())
                {
                    m_active_nodes.erase(it);
                }
            }
        }

        bool allocate_node_to_register(Node node, int num_args)
        {
            if (NO_DEPS_OP.count(node->op()))
            {
                return true;
            }
            if (node->dtype() == Dtype::f32)
            {
                for (uint32_t x = 0; x < m_f_regs; x++)
                {
                    if (!m_f_status.test(x))
                    {
                        m_f_status.set(x);
                        m_active_nodes.push_back(node);
                        node->set_reg(FReg(x));
                        return true;
                    }
                }
            }
            else
            {
                uint32_t prefered = 0;
                if (m_use_free_args)
                {
                    prefered = 0xff >> (8 - std::min(num_args, 8));
                    if (node->op() != Op::alloc)
                    {
                        prefered = ~prefered;
                    }
                }
                else
                {
                    prefered = 0xffffff00;
                    if (node->op() == Op::alloc)
                    {
                        prefered = 0xff;
                    }
                }
                std::bitset<32> p(prefered);
                // iterate all registers, with prefer registers (p_stat=1)
                for (uint32_t x = 0; x < m_r_regs; x++)
                {
                    if (!m_r_status.test(x) && p.test(x))
                    {
                        m_r_status.set(x);
                        m_active_nodes.push_back(node);
                        node->set_reg(RReg(x));
                        return true;
                    }
                }
            }
            return false;
        }

        void spill_at_interval(Node node)
        {
            auto get_spilled_register = [this](Node n) -> Reg {
                Reg ret = RReg(R0);
                if (n->dtype() == Dtype::f32)
                {
                    ret = FReg(0);
                }
                int offset = 0;
                if (n->op() == Op::alloc)
                {
                    offset = 4 * (2 + m_spill_args);
                    m_spill_args += 1;
                }
                else
                {
                    m_spill_vars += 1;
                    offset = -4 * m_spill_vars;
                }
                ret.set_offset(offset);
                m_spilled = true;
                return ret;
            };
            // spill first non-alloc node if available
            NodePtr spill = nullptr;
            for (auto &&n : m_active_nodes)
            {
                if (n->op() != Op::alloc)
                {
                    spill = n;
                    break;
                }
            }
            if (spill && m_liveness[spill->id()] > m_liveness[node->id()])
            {
                auto it = std::find(m_active_nodes.begin(), m_active_nodes.end(), spill);
                if (it != m_active_nodes.end())
                {
                    m_active_nodes.erase(it);
                }
                m_active_nodes.push_back(node);
                node->set_reg(spill->reg());
                spill->set_reg(get_spilled_register(spill));
            }
            else
            {
                node->set_reg(get_spilled_register(node));
            }
        }

        uint32_t m_r_regs{32};
        uint32_t m_f_regs{32};
        uint32_t m_spill_args{0};
        uint32_t m_spill_vars{0};
        bool m_spilled{false};
        bool m_use_free_args{false};
        std::bitset<32> m_r_status;
        std::bitset<32> m_f_status;
        std::map<int, int> m_liveness;
        std::vector<NodePtr> m_active_nodes;
    };

    class ASMGenerator
    {
    public:
        InsPackages generate(std::vector<Node> &nodes, const RegAllocator &ra)
        {
            InsPackages codes;
            for (uint32_t i = 0; i < nodes.size(); ++i)
            {
                std::vector<Node> block;
                for (uint32_t j = i; j < nodes.size(); ++j)
                {
                    Node n = nodes[j];
                    auto op = n->op();
                    if (op != Op::push && op != Op::pop)
                    {
                        break;
                    }
                    bool spilled = false;
                    if (op == Op::push)
                    {
                        spilled = n->inputs()[0]->reg().spilled();
                    }
                    else
                    {
                        spilled = n->reg().spilled();
                    }
                    if (!spilled)
                    {
                        if (block.size())
                        {
                            if (block.back()->op() != op)
                            {
                                break;
                            }
                            block.push_back(n);
                        }
                        else
                        {
                            block.push_back(n);
                        }
                    }
                }
                Node n = nodes[i];
                if (block.size())
                {
                    generate(block, codes);
                }
                else
                {
                    generate(n, ra);
                    for (auto &&pkg : n->inss().pkgs)
                    {
                        codes.add(pkg);
                    }
                }
                if (block.size())
                {
                    i += block.size() - 1;
                }
            }
            return codes;
        }

    protected:
        void generate(Node n, const RegAllocator &ra)
        {
            ra.allocate_temp_register(n);
            switch (n->op())
            {
            case Op::constant:
            case Op::mov:
            {
                if (n->op() == Op::constant)
                {
                    mov(n, n->reg(), n->value());
                }
                else
                {
                    mov(n, n->reg(), n->inputs()[0]->reg());
                }
                break;
            }
            case Op::add:
            case Op::sub:
            case Op::mul:
            case Op::mulh:
            case Op::div:
            case Op::mod:
            {
                if (n->op() == Op::add && n->imms().size() > 0)
                {
                    aipu_assert(n->imms().size() < 3);
                    if (n->imms().size() == 1)
                    {
                        arithmetic(n, Op::add, n->reg(), n->inputs()[0]->reg(), n->imms()[0]);
                    }
                    else
                    {
                        mov(n, n->reg(), n->imms()[0] + n->imms()[1]);
                    }
                }
                else if (n->op() == Op::sub && n->imms().size() > 0)
                {
                    aipu_assert(n->imms().size() == 1);
                    arithmetic(n, Op::sub, n->reg(), n->inputs()[0]->reg(), n->imms()[0]);
                }
                else
                {
                    auto inp0 = n->inputs()[0];
                    auto inp1 = n->inputs()[1];
                    arithmetic(n, n->op(), n->reg(), inp0->reg(), inp1->reg(), n->dtype(), inp0->dtype(), inp1->dtype());
                }
                break;
            }
            case Op::itof:
            case Op::ftoi:
            case Op::rint:
            case Op::log2:
            case Op::sqrt:
            case Op::rsqrt:
            {
                arithmetic(n, n->op(), n->reg(), n->inputs()[0]->reg());
                break;
            }
            case Op::pop:
            case Op::push:
            {
                if (n->op() == Op::pop)
                {
                    memory(n, Op::pop, n->reg());
                }
                else
                {
                    memory(n, Op::push, n->inputs()[0]->reg());
                }
                break;
            }
            case Op::load:
            case Op::store:
            {
                if (n->op() == Op::load)
                {
                    memory(n, Op::load, n->reg(), n->inputs()[0]->reg());
                }
                else
                {
                    memory(n, Op::store, n->inputs()[0]->reg(), n->inputs()[1]->reg());
                }
                break;
            }
            case Op::label:
            {
                n->inss().add(n->to_string());
                break;
            }
            case Op::ret:
            case Op::call:
            {
                branch(n, n->op());
                break;
            }
            case Op::cbz:
            case Op::cbnz:
            case Op::jmp:
            {
                if (n->op() == Op::jmp)
                {
                    branch(n, Op::cbz, RReg(R31), n->inputs()[0]->to_string());
                }
                else
                {
                    branch(n, n->op(), n->inputs()[0]->reg(), n->inputs()[1]->to_string());
                }
                break;
            }
            case Op::lshit:
            case Op::rshit:
            case Op::band:
            case Op::bor:
            case Op::bxor:
            {
                bitwise(n, n->op(), n->reg(), n->inputs()[0]->reg(), n->inputs()[1]->reg());
                break;
            }
            case Op::slt:
            case Op::sle:
            case Op::sgt:
            case Op::sge:
            case Op::seq:
            case Op::sne:
            {
                auto inp0 = n->inputs()[0];
                auto inp1 = n->inputs()[1];
                compare(n, n->op(), n->reg(), inp0->reg(), inp1->reg(), n->dtype(), inp0->dtype(), inp1->dtype());
                break;
            }
            case Op::max:
            case Op::min:
            {
                auto inp0 = n->inputs()[0];
                auto inp1 = n->inputs()[1];
                compare(n, n->op(), n->reg(), inp0->reg(), inp1->reg(), inp0->dtype(), inp1->dtype());
                break;
            }
            default:
                break;
            }
        }
        void generate(std::vector<Node> &block, InsPackages &codes)
        {
            aipu_assert(block.size() > 0);
            auto block_type = block[0]->op();
            if (block_type == Op::push)
            {
                codes.add(SUB(SP, SP, 4 * block.size()));
                for (uint32_t i = 0; i < block.size(); ++i)
                {
                    codes.add(ST(block[i]->inputs()[0]->reg(), SP, 4 * (block.size() - 1 - i)));
                }
            }
            if (block_type == Op::pop)
            {
                for (uint32_t i = 0; i < block.size() / 2; ++i)
                {
                    if (block.size() % 2 == 0 && i == block.size() / 2 - 1)
                    {
                        codes.add({
                            LD(block[2 * i]->reg(), SP, 4 * (2 * i)),
                            LD(block[2 * i + 1]->reg(), SP, 4 * (2 * i + 1)),
                            ADD(SP, SP, 4 * block.size()),
                        });
                    }
                    else
                    {
                        codes.add({
                            LD(block[2 * i]->reg(), SP, 4 * (2 * i)),
                            LD(block[2 * i + 1]->reg(), SP, 4 * (2 * i + 1)),
                        });
                    }
                }
                if (block.size() % 2 != 0)
                {
                    codes.add({
                        LD(block[block.size() - 1]->reg(), SP, 4 * (block.size() - 1)),
                        ADD(SP, SP, 4 * block.size()),
                    });
                }
            }
        }
        template <typename... Args>
        void load_spilled_regs(Node n, Reg rd, Args... args)
        {
            load_spilled_reg(n, rd, false);
            for (auto &&reg : std::vector<Reg>({args...}))
            {
                load_spilled_reg(n, reg, true);
            }
        }
        void load_spilled_reg(Node n, Reg rd, bool load_value)
        {
            aipu_assert(!rd.is_none());
            if (!rd.spilled())
            {
                return;
            }
            n->inss().add(SUB(SP, SP, 4)).add(ST(rd, SP, 0));
            if (load_value)
            {
                n->inss().add(LD(rd, FP, rd.offset()));
            }
        }
        template <typename... Args>
        void store_spilled_regs(Node n, Reg rd, Args... args)
        {
            std::vector<Reg> regs = {args...};
            for (int i = regs.size() - 1; i >= 0; --i)
            {
                store_spilled_reg(n, regs[i], false);
            }
            store_spilled_reg(n, rd, true);
        }
        void store_spilled_reg(Node n, Reg rd, bool store_value)
        {
            aipu_assert(!rd.is_none());
            if (!rd.spilled())
            {
                return;
            }
            if (store_value)
            {
                n->inss().add(ST(rd, FP, rd.offset()));
            }
            n->inss().add(LD(rd, SP, 0)).add(ADD(SP, SP, 4));
        }
        void mov(Node n, Reg rd, uint32_t imm)
        {
            aipu_assert(rd.is_rreg());
            load_spilled_regs(n, rd);
            uint16_t half = imm & 0xffff;
            int16_t low = *(int16_t *)(&half);
            half = (imm >> 16) & 0xffff;
            int16_t high = *(int16_t *)(&half);
            n->inss().add(MOV(rd, low));
            if (low < 0 || high != 0)
            {
                n->inss().add(MOVH(rd, high));
            }
            store_spilled_regs(n, rd);
        }
        void mov(Node n, Reg rd, Reg rn)
        {
            aipu_assert(!rd.is_none());
            aipu_assert(!rn.is_none());
            load_spilled_regs(n, rd, rn);
            if (rd.is_rreg() && rn.is_rreg())
            {
                n->inss().add(ADD(rd, rn, 0));
            }
            else
            {
                n->inss().add(MOV(rd, rn));
            }
            store_spilled_regs(n, rd, rn);
        }
        void arithmetic(Node n, Op op, Reg rd, Reg rn, uint32_t imm)
        {
            aipu_assert(rd.is_rreg());
            aipu_assert(rn.is_rreg());
            aipu_assert(!(imm & 0xfffffc00));
            load_spilled_regs(n, rd, rn);
            if (op == Op::add)
            {
                n->inss().add(ADD(rd, rn, imm));
            }
            else if (op == Op::sub)
            {
                n->inss().add(SUB(rd, rn, imm));
            }
            store_spilled_regs(n, rd, rn);
        }
        void arithmetic(Node n, Op op, Reg rd, Reg rn, Reg rm, Dtype rd_d, Dtype rn_d, Dtype rm_d)
        {
            aipu_assert(!rd.is_none());
            aipu_assert(!rn.is_none());
            aipu_assert(!rm.is_none());
            if (rd.is_freg())
            {
                aipu_assert(rn.is_freg());
                aipu_assert(rm.is_freg());
            }
            load_spilled_regs(n, rd, rn, rm);
            n->inss().add(Arithmetic(op, rd, rn, rm, rd_d, rn_d, rm_d));
            store_spilled_regs(n, rd, rn, rm);
        }
        void arithmetic(Node n, Op op, Reg rd, Reg rn)
        {
            aipu_assert(rd.is_freg());
            aipu_assert(rn.is_freg());
            load_spilled_regs(n, rd, rn);
            n->inss().add(Arithmetic(op, rd, rn));
            store_spilled_regs(n, rd, rn);
        }
        void memory(Node n, Op op, Reg rd, Reg rn, int offset = 0)
        {
            aipu_assert(!rd.is_none());
            aipu_assert(rn.is_rreg());
            if (op == Op::load)
            {
                load_spilled_regs(n, rd, rn);
            }
            else if (op == Op::store)
            {
                load_spilled_reg(n, rd, true);
                load_spilled_reg(n, rn, true);
            }
            if (rn.index() == static_cast<uint32_t>(SP))
            {
                if (rd.spilled())
                {
                    offset += 4;
                }
                if (rn.spilled())
                {
                    offset += 4;
                }
            }
            if (op == Op::load)
            {
                n->inss().add(LD(rd, rn, offset));
                store_spilled_regs(n, rd, rn);
            }
            else if (op == Op::store)
            {
                n->inss().add(ST(rd, rn, offset));
                store_spilled_reg(n, rn, false);
                store_spilled_reg(n, rd, false);
            }
        }
        void memory(Node n, Op op, Reg rd)
        {
            aipu_assert(!rd.is_none());
            if (rd.spilled())
            {
                if (op == Op::push)
                {
                    n->inss().add(SUB(SP, SP, 8)).add(ST(rd, SP, 0)).add(LD(rd, FP, rd.offset()));
                    n->inss().add(ST(rd, SP, 4)).add(LD(rd, SP, 0)).add(ADD(SP, SP, 4));
                }
                else if (op == Op::pop)
                {
                    n->inss().add(SUB(SP, SP, 4)).add(ST(rd, SP, 0)).add(LD(rd, SP, 4));
                    n->inss().add(ST(rd, FP, rd.offset())).add(LD(rd, SP, 0)).add(ADD(SP, SP, 8));
                }
            }
            else
            {
                if (op == Op::push)
                {
                    n->inss().add(SUB(SP, SP, 4)).add(ST(rd, SP, 0));
                }
                else if (op == Op::pop)
                {
                    n->inss().add(LD(rd, SP, 0)).add(ADD(SP, SP, 4));
                }
            }
        }
        void branch(Node n, Op op, Reg rd, const std::string &label)
        {
            aipu_assert(rd.is_rreg());
            if (rd.spilled())
            {
                n->inss().add(SUB(SP, SP, 4)).add(ST(rd, SP, 0)).add(LD(rd, FP, rd.offset()));
                if (op == Op::cbz)
                {
                    n->inss().add({
                        CBZ(rd, label),
                        LD(rd, SP, 0),
                        ADD(SP, SP, 4),
                    });
                }
                else if (op == Op::cbnz)
                {
                    n->inss().add({
                        CBNZ(rd, label),
                        LD(rd, SP, 0),
                        ADD(SP, SP, 4),
                    });
                }
            }
            else
            {
                if (op == Op::cbz)
                {
                    n->inss().add(CBZ(rd, label));
                }
                else if (op == Op::cbnz)
                {
                    n->inss().add(CBNZ(rd, label));
                }
            }
        }
        void branch(Node n, Op op)
        {
            if (op == Op::ret)
            {
                n->inss().add(BR(R30));
            }
            else if (op == Op::call)
            {
                aipu_assert(!n->reg().is_none());
                auto label = n->inputs()[0]->name();
                n->inss().add(BL(label));
            }
        }
        void bitwise(Node n, Op op, Reg rd, Reg rn, Reg rm)
        {
            aipu_assert(rd.is_rreg());
            aipu_assert(rn.is_rreg());
            aipu_assert(rm.is_rreg());
            load_spilled_regs(n, rd, rn, rm);
            std::string bit_op;
            switch (op)
            {
            case Op::lshit:
                bit_op = "lsl";
                break;
            case Op::rshit:
                bit_op = "lsr";
                break;
            case Op::band:
                bit_op = "and";
                break;
            case Op::bor:
                bit_op = "or";
                break;
            case Op::bxor:
                bit_op = "xor";
                break;
            default:
                break;
            }
            aipu_assert(bit_op.size());
            n->inss().add(Bitwise(op, rd, rn, rm));
            store_spilled_regs(n, rd, rn, rm);
        }
        void compare(Node n, Op op, Reg rd, Reg rn, Reg rm, Dtype rd_d, Dtype rn_d, Dtype rm_d)
        {
            aipu_assert(rd.is_rreg());
            aipu_assert(!rn.is_none());
            aipu_assert(!rm.is_none());
            load_spilled_regs(n, rd, rn, rm);
            std::string cmp_op;
            bool is_freg = rn.is_freg() || rm.is_freg();
            if (is_freg)
            {
                aipu_assert(rn.is_freg());
                aipu_assert(rm.is_freg());
            }
            switch (op)
            {
            case Op::slt:
                cmp_op = is_freg ? "cmp.qlt" : "cmp.lt";
                break;
            case Op::sle:
                cmp_op = is_freg ? "cmp.qle" : "cmp.le";
                break;
            case Op::sgt:
                cmp_op = is_freg ? "cmp.qgt" : "cmp.gt";
                break;
            case Op::sge:
                cmp_op = is_freg ? "cmp.qge" : "cmp.ge";
                break;
            case Op::seq:
                cmp_op = is_freg ? "cmp.qeq" : "cmp.eq";
                break;
            case Op::sne:
                cmp_op = is_freg ? "cmp.qne" : "cmp.ne";
                break;
            default:
                break;
            }
            aipu_assert(cmp_op.size());
            n->inss().add(Compare(op, rd, rn, rm, rd_d, rn_d, rm_d));
            store_spilled_regs(n, rd, rn, rm);
        }
        void compare(Node n, Op op, Reg rd, Reg rn, Reg rm, Dtype rn_d, Dtype rm_d)
        {
            aipu_assert(!rd.is_none());
            aipu_assert(!rn.is_none());
            aipu_assert(!rm.is_none());
            if (rd.is_freg() || rn.is_freg() || rm.is_freg())
            {
                aipu_assert(rd.is_freg());
                aipu_assert(rn.is_freg());
                aipu_assert(rm.is_freg());
            }
            load_spilled_regs(n, rd, rn, rm);
            n->inss().add(Compare(op, rd, rn, rm, rn_d, rm_d));
            store_spilled_regs(n, rd, rn, rm);
        }
    };

    class Emitter
    {
    public:
        std::string emit(Function func, bool &spilled, bool use_free_args = false)
        {
            auto codes = lower(func, spilled, use_free_args);
            return codes.disasm();
        }
        InsPackages lower(Function func, bool &spilled, bool use_free_args = false)
        {
            static std::set<Op> valid_ops = {
                Op::constant,
                Op::add,
                Op::sub,
                Op::mul,
                Op::mulh,
                Op::div,
                Op::mod,
                Op::ret,
                Op::label,
                Op::call,
                Op::push,
                Op::pop,
                Op::cbz,
                Op::cbnz,
                Op::load,
                Op::store,
                Op::lshit,
                Op::rshit,
                Op::band,
                Op::bor,
                Op::bxor,
                Op::slt,
                Op::sle,
                Op::sgt,
                Op::sge,
                Op::seq,
                Op::sne,
                Op::jmp,
                Op::mov,
                Op::itof,
                Op::ftoi,
                Op::max,
                Op::min,
                Op::rint,
                Op::log2,
                Op::sqrt,
                Op::rsqrt,
            };

            blocking(func);
            isel(func);
            std::map<int, std::vector<Node>> movs;
            build_phi(func, movs);
            opt_imm(func);
            RegAllocator ra(22, 32, use_free_args);
            ra.allocate(func);
            spilled = ra.spilled();
            resolve_phi(func, movs);
            fix_call(func);
            std::vector<Node> nodes;
            for (auto ins : func->codes())
            {
                if (!valid_ops.count(ins->op()))
                {
                    continue;
                }
                if (ins->op() == Op::mov && ins->reg() == ins->inputs()[0]->reg())
                {
                    continue;
                }
                nodes.push_back(ins);
            }

            ASMGenerator asmg;
            return asmg.generate(nodes, ra);
        }

    protected:
        // make a function into severial blocks
        // blocking must be before RA
        Function blocking(Function func)
        {
            std::vector<NodePtr> res;
            NodePtr cur_label = nullptr;
            for (auto &&ins : func->body())
            {
                if (ins->op() != Op::label && !cur_label)
                {
                    cur_label = NodeImpl::create(Op::label);
                    res.push_back(cur_label);
                }
                res.push_back(ins);
                if (BRANCH_OP.count(ins->op()))
                {
                    cur_label = NodeImpl::create(Op::label);
                    res.push_back(cur_label);
                }
            }
            // reduce the same label
            std::vector<std::vector<NodePtr>> repeated_labels;
            for (uint32_t i = 0; i < res.size() - 1; i++)
            {
                if (res[i]->op() == Op::label && res[i + 1]->op() == Op::label)
                {
                    repeated_labels.push_back({res[i], res[i + 1]});
                    i++;
                }
            }
            for (auto &&l : repeated_labels)
            {
                for (uint32_t i = 1; i < l.size(); i++)
                {
                    auto it = std::find(res.begin(), res.end(), l[i]);
                    if (it != res.end())
                    {
                        res.erase(it);
                    }
                    for (auto &&ins : res)
                    {
                        ins->replace_input(l[i], l[0]);
                    }
                }
            }
            func->body() = std::move(res);
            return func;
        }
        Node get_next_block_label(Function func, Node ins)
        {
            Node ret = nullptr;
            std::vector<NodePtr> nodes(func->body().rbegin(), func->body().rend());
            for (auto &&n : nodes)
            {
                if (n->op() == Op::label)
                {
                    ret = n;
                }
                if (n == ins.operator NodePtr())
                {
                    return ret;
                }
            }
            return ret;
        }
        Function isel(Function func)
        {
            std::vector<NodePtr> res;
            for (auto &&ins : func->body())
            {
                switch (ins->op())
                {
                case Op::constant:
                case Op::alloc:
                {
                    aipu_assert(ins->inputs().size() == 0);
                    aipu_assert(is_scalar(ins->dtype()));
                    res.push_back(ins);
                    if (ins->dtype() == Dtype::f32)
                    {
                        ins->set_dtype(Dtype::u32);
                        auto consumers = ins->consumers();
                        auto mov = NodeImpl::create(Op::mov, Dtype::f32, ins);
                        for (auto &&c : consumers)
                        {
                            auto n = c.lock();
                            n->replace_input(ins, mov);
                        }
                        res.push_back(mov);
                    }
                    break;
                }
                case Op::add:
                case Op::sub:
                case Op::mul:
                case Op::mulh:
                case Op::div:
                case Op::mod:
                {
                    aipu_assert(ins->inputs().size() == 2);
                    auto inp0 = ins->inputs()[0];
                    auto inp1 = ins->inputs()[1];
                    aipu_assert(is_scalar(inp0->dtype()));
                    aipu_assert(is_scalar(inp1->dtype()));
                    aipu_assert(is_scalar(ins->dtype()));
                    bool is_input_float = false;
                    if (inp0->dtype() == Dtype::f32 || inp1->dtype() == Dtype::f32)
                    {
                        if (inp0->dtype() != Dtype::f32)
                        {
                            auto mov = NodeImpl::create(Op::mov, Dtype::f32, inp0);
                            auto cvt = NodeImpl::create(Op::itof, Dtype::f32, mov);
                            ins->replace_input(inp0, cvt);
                            res.push_back(mov);
                            res.push_back(cvt);
                        }
                        if (inp1->dtype() != Dtype::f32)
                        {
                            auto mov = NodeImpl::create(Op::mov, Dtype::f32, inp1);
                            auto cvt = NodeImpl::create(Op::itof, Dtype::f32, mov);
                            ins->replace_input(inp1, cvt);
                            res.push_back(mov);
                            res.push_back(cvt);
                        }
                        is_input_float = true;
                    }
                    res.push_back(ins);
                    if (is_input_float && ins->dtype() != Dtype::f32)
                    {
                        auto consumers = ins->consumers();
                        auto cvt = NodeImpl::create(Op::ftoi, Dtype::f32, ins);
                        auto mov = NodeImpl::create(Op::mov, ins->dtype(), cvt);
                        ins->set_dtype(Dtype::f32);
                        for (auto &&c : consumers)
                        {
                            auto n = c.lock();
                            n->replace_input(ins, mov);
                        }
                        res.push_back(cvt);
                        res.push_back(mov);
                    }
                    else if (!is_input_float && ins->dtype() == Dtype::f32)
                    {
                        ins->set_dtype(Dtype::i32);
                        auto consumers = ins->consumers();
                        auto mov = NodeImpl::create(Op::mov, Dtype::f32, ins);
                        auto cvt = NodeImpl::create(Op::itof, Dtype::f32, mov);
                        for (auto &&c : consumers)
                        {
                            auto n = c.lock();
                            n->replace_input(ins, cvt);
                        }
                        res.push_back(mov);
                        res.push_back(cvt);
                    }
                    break;
                }
                case Op::ret:
                {
                    aipu_assert(ins->inputs().size() <= 1);
                    aipu_assert(ins->consumers().size() == 0);
                    aipu_assert(is_none(ins->dtype()));
                    if (ins->inputs().size() == 1)
                    {
                        auto inp0 = ins->inputs()[0];
                        aipu_assert(is_scalar(inp0->dtype()));
                        aipu_assert(is_scalar(func->dtype()));
                        if (inp0->dtype() == Dtype::f32 && func->dtype() != Dtype::f32)
                        {
                            auto cvt = NodeImpl::create(Op::ftoi, Dtype::f32, inp0);
                            auto mov = NodeImpl::create(Op::mov, func->dtype(), cvt);
                            ins->replace_input(inp0, mov);
                            res.push_back(cvt);
                            res.push_back(mov);
                        }
                        else if (inp0->dtype() != Dtype::f32 && func->dtype() == Dtype::f32)
                        {
                            auto mov = NodeImpl::create(Op::mov, Dtype::f32, inp0);
                            auto cvt = NodeImpl::create(Op::itof, Dtype::f32, mov);
                            ins->replace_input(inp0, cvt);
                            res.push_back(mov);
                            res.push_back(cvt);
                        }
                    }
                    else
                    {
                        aipu_assert(is_none(func->dtype()));
                    }
                    res.push_back(ins);
                    break;
                }
                case Op::phi:
                {
                    aipu_assert(is_scalar(ins->dtype()));
                    for (auto &&inp : ins->inputs())
                    {
                        aipu_assert(inp->dtype() == ins->dtype());
                    }
                    res.push_back(ins);
                    break;
                }
                case Op::label:
                {
                    aipu_assert(ins->inputs().size() == 0);
                    aipu_assert(is_none(ins->dtype()));
                    res.push_back(ins);
                    break;
                }
                case Op::call:
                {
                    aipu_assert(ins->inputs().size() >= 1);
                    auto inp0 = ins->inputs()[0];
                    aipu_assert(inp0->op() == Op::fun);
                    if (ins->consumers().size())
                    {
                        aipu_assert(is_scalar(ins->dtype()));
                        aipu_assert(is_scalar(inp0->dtype()));
                    }
                    if (is_none(inp0->dtype()))
                    {
                        aipu_assert(is_none(ins->dtype()));
                    }
                    res.push_back(ins);
                    if (inp0->dtype() == Dtype::f32 && is_integral(ins->dtype()))
                    {
                        auto consumers = ins->consumers();
                        auto cvt = NodeImpl::create(Op::ftoi, Dtype::f32, ins);
                        auto mov = NodeImpl::create(Op::mov, ins->dtype(), cvt);
                        ins->set_dtype(Dtype::f32);
                        for (auto &&c : consumers)
                        {
                            auto n = c.lock();
                            n->replace_input(ins, mov);
                        }
                        res.push_back(cvt);
                        res.push_back(mov);
                    }
                    else if (is_integral(inp0->dtype()) && ins->dtype() == Dtype::f32)
                    {
                        ins->set_dtype(Dtype::i32);
                        auto consumers = ins->consumers();
                        auto mov = NodeImpl::create(Op::mov, Dtype::f32, ins);
                        auto cvt = NodeImpl::create(Op::itof, Dtype::f32, mov);
                        for (auto &&c : consumers)
                        {
                            auto n = c.lock();
                            n->replace_input(ins, cvt);
                        }
                        res.push_back(mov);
                        res.push_back(cvt);
                    }
                    break;
                }
                case Op::fun:
                {
                    aipu_assert(ins->inputs().size() == 0);
                    aipu_assert(ins->name().size());
                    res.push_back(ins);
                    break;
                }
                case Op::pop:
                {
                    aipu_assert(ins->inputs().size() == 0);
                    aipu_assert(is_scalar(ins->dtype()));
                    res.push_back(ins);
                    break;
                }
                case Op::push:
                {
                    aipu_assert(ins->inputs().size() == 1);
                    auto inp0 = ins->inputs()[0];
                    aipu_assert(is_scalar(inp0->dtype()));
                    aipu_assert(is_scalar(ins->dtype()));
                    aipu_assert(ins->consumers().size() == 0);
                    res.push_back(ins);
                    break;
                }
                case Op::cbz:
                case Op::cbnz:
                {
                    aipu_assert(ins->inputs().size() == 2);
                    auto inp0 = ins->inputs()[0];
                    auto inp1 = ins->inputs()[1];
                    aipu_assert(is_integral(inp0->dtype()));
                    aipu_assert(inp1->op() == Op::label);
                    aipu_assert(is_none(ins->dtype()));
                    aipu_assert(ins->consumers().size() == 0);
                    res.push_back(ins);
                    break;
                }
                case Op::load:
                {
                    aipu_assert(ins->inputs().size() == 1);
                    auto inp0 = ins->inputs()[0];
                    aipu_assert(is_integral(inp0->dtype()));
                    aipu_assert(is_scalar(ins->dtype()));
                    res.push_back(ins);
                    break;
                }
                case Op::store:
                {
                    aipu_assert(ins->inputs().size() == 2);
                    auto inp0 = ins->inputs()[0];
                    auto inp1 = ins->inputs()[1];
                    aipu_assert(is_scalar(inp0->dtype()));
                    aipu_assert(is_integral(inp1->dtype()));
                    aipu_assert(is_scalar(ins->dtype()));
                    aipu_assert(ins->consumers().size() == 0);
                    res.push_back(ins);
                    break;
                }
                case Op::lshit:
                case Op::rshit:
                case Op::band:
                case Op::bor:
                case Op::bxor:
                {
                    aipu_assert(ins->inputs().size() == 2);
                    auto inp0 = ins->inputs()[0];
                    auto inp1 = ins->inputs()[1];
                    aipu_assert(is_integral(inp0->dtype()));
                    aipu_assert(is_integral(inp1->dtype()));
                    aipu_assert(is_scalar(ins->dtype()));
                    res.push_back(ins);
                    if (ins->dtype() == Dtype::f32)
                    {
                        ins->set_dtype(Dtype::i32);
                        auto consumers = ins->consumers();
                        auto mov = NodeImpl::create(Op::mov, Dtype::f32, ins);
                        auto cvt = NodeImpl::create(Op::itof, Dtype::f32, mov);
                        for (auto &&c : consumers)
                        {
                            auto n = c.lock();
                            n->replace_input(ins, cvt);
                        }
                        res.push_back(mov);
                        res.push_back(cvt);
                    }
                    break;
                }
                case Op::slt:
                case Op::sle:
                case Op::sgt:
                case Op::sge:
                case Op::seq:
                case Op::sne:
                {
                    aipu_assert(ins->inputs().size() == 2);
                    auto inp0 = ins->inputs()[0];
                    auto inp1 = ins->inputs()[1];
                    aipu_assert(is_scalar(inp0->dtype()));
                    aipu_assert(is_scalar(inp1->dtype()));
                    aipu_assert(is_scalar(ins->dtype()));
                    if (inp0->dtype() == Dtype::f32 || inp1->dtype() == Dtype::f32)
                    {
                        aipu_assert(inp0->dtype() == Dtype::f32);
                        aipu_assert(inp1->dtype() == Dtype::f32);
                    }
                    res.push_back(ins);
                    if (ins->dtype() == Dtype::f32)
                    {
                        ins->set_dtype(Dtype::i32);
                        auto consumers = ins->consumers();
                        auto mov = NodeImpl::create(Op::mov, Dtype::f32, ins);
                        auto cvt = NodeImpl::create(Op::itof, Dtype::f32, mov);
                        for (auto &&c : consumers)
                        {
                            auto n = c.lock();
                            n->replace_input(ins, cvt);
                        }
                        res.push_back(mov);
                        res.push_back(cvt);
                    }
                    break;
                }
                case Op::jmp:
                {
                    aipu_assert(ins->inputs().size() == 1);
                    auto inp0 = ins->inputs()[0];
                    aipu_assert(inp0->op() == Op::label);
                    aipu_assert(ins->consumers().size() == 0);
                    aipu_assert(is_none(ins->dtype()));
                    res.push_back(ins);
                    break;
                }
                case Op::mov:
                {
                    aipu_assert(ins->inputs().size() == 1);
                    auto inp0 = ins->inputs()[0];
                    aipu_assert(is_scalar(inp0->dtype()));
                    aipu_assert(is_scalar(ins->dtype()));
                    res.push_back(ins);
                    break;
                }
                case Op::itof:
                case Op::ftoi:
                {
                    aipu_assert(ins->inputs().size() == 1);
                    auto inp0 = ins->inputs()[0];
                    aipu_assert(inp0->dtype() == Dtype::f32);
                    aipu_assert(ins->dtype() == Dtype::f32);
                    res.push_back(ins);
                    break;
                }
                case Op::max:
                case Op::min:
                {
                    aipu_assert(ins->inputs().size() == 2);
                    auto inp0 = ins->inputs()[0];
                    auto inp1 = ins->inputs()[1];
                    aipu_assert(is_scalar(inp0->dtype()));
                    aipu_assert(is_scalar(inp1->dtype()));
                    aipu_assert(is_scalar(ins->dtype()));
                    bool is_input_float = false;
                    if (inp0->dtype() == Dtype::f32 || inp1->dtype() == Dtype::f32)
                    {
                        aipu_assert(inp0->dtype() == Dtype::f32);
                        aipu_assert(inp1->dtype() == Dtype::f32);
                        is_input_float = true;
                    }
                    res.push_back(ins);
                    if (is_input_float && ins->dtype() != Dtype::f32)
                    {
                        auto consumers = ins->consumers();
                        auto cvt = NodeImpl::create(Op::ftoi, Dtype::f32, ins);
                        auto mov = NodeImpl::create(Op::mov, ins->dtype(), cvt);
                        ins->set_dtype(Dtype::f32);
                        for (auto &&c : consumers)
                        {
                            auto n = c.lock();
                            n->replace_input(ins, mov);
                        }
                        res.push_back(cvt);
                        res.push_back(mov);
                    }
                    else if (!is_input_float && ins->dtype() == Dtype::f32)
                    {
                        ins->set_dtype(Dtype::i32);
                        auto consumers = ins->consumers();
                        auto mov = NodeImpl::create(Op::mov, Dtype::f32, ins);
                        auto cvt = NodeImpl::create(Op::itof, Dtype::f32, mov);
                        for (auto &&c : consumers)
                        {
                            auto n = c.lock();
                            n->replace_input(ins, cvt);
                        }
                        res.push_back(mov);
                        res.push_back(cvt);
                    }
                    break;
                }
                case Op::rint:
                {
                    aipu_assert(ins->inputs().size() == 1);
                    auto inp0 = ins->inputs()[0];
                    aipu_assert(inp0->dtype() == Dtype::f32);
                    aipu_assert(ins->dtype() == Dtype::f32);
                    res.push_back(ins);
                    break;
                }
                case Op::log2:
                {
                    aipu_assert(ins->inputs().size() == 1);
                    auto inp0 = ins->inputs()[0];
                    aipu_assert(inp0->dtype() == Dtype::f32);
                    aipu_assert(ins->dtype() == Dtype::f32);
                    res.push_back(ins);
                    break;
                }
                case Op::sqrt:
                case Op::rsqrt:
                {
                    aipu_assert(ins->inputs().size() == 1);
                    auto inp0 = ins->inputs()[0];
                    aipu_assert(inp0->dtype() == Dtype::f32);
                    aipu_assert(ins->dtype() == Dtype::f32);
                    res.push_back(ins);
                    break;
                }
                case Op::keep_live:
                {
                    aipu_assert(ins->inputs().size());
                    aipu_assert(ins->consumers().size() == 0);
                    aipu_assert(is_none(ins->dtype()));
                    res.push_back(ins);
                    break;
                }
                default:
                    ERROR() << "unsupported instruction: " << ins->to_string();
                    break;
                }
            }
            func->body() = std::move(res);
            return func;
        }
        // build phi should be before RA, it will insert mov instruction
        Function build_phi(Function func, std::map<int, std::vector<Node>> &movs)
        {
            std::vector<NodePtr> phis;
            for (auto &&ins : func->body())
            {
                if (ins->op() == Op::phi)
                {
                    phis.push_back(ins);
                }
            }
            for (auto &&phi : phis)
            {
                for (auto &&inp : phi->inputs())
                {
                    NodePtr label = get_next_block_label(func, inp);
                    if (label)
                    {
                        auto mov = NodeImpl::create(Op::mov, inp->dtype(), inp);
                        movs[phi->id()].push_back(mov);
                        auto it = std::find(func->body().begin(), func->body().end(), label);
                        int idx = it - func->body().begin();
                        while (idx > 0 && BRANCH_OP.count(func->body()[idx - 1]->op()))
                        {
                            idx -= 1;
                        }
                        func->body().insert(func->body().begin() + idx, mov);
                    }
                }
            }
            return func;
        }
        Function opt_imm(Function func)
        {
            std::vector<NodePtr> res, constants;
            for (auto &&ins : func->body())
            {
                if (ins->op() == Op::constant)
                {
                    constants.push_back(ins);
                }
                res.push_back(ins);
            }
            for (auto &&ins : constants)
            {
                auto check_func = [ins](NodeRef ref) -> bool {
                    if (!ref.expired())
                    {
                        auto n = ref.lock();
                        switch (n->op())
                        {
                        case Op::add:
                        {
                            if (n->dtype() != Dtype::f32 && ins->value() <= 1023)
                            {
                                return true;
                            }
                            break;
                        }
                        case Op::sub:
                        {
                            if (n->inputs()[0]->op() != Op::constant &&
                                n->inputs()[1]->op() == Op::constant &&
                                n->dtype() != Dtype::f32 && ins->value() <= 1023)
                            {
                                return true;
                            }
                            break;
                        }
                        default:
                            break;
                        }
                        return false;
                    }
                    return true;
                };
                auto consumers = ins->consumers();
                if (std::all_of(consumers.begin(), consumers.end(), check_func))
                {
                    for (auto &&c : consumers)
                    {
                        if (!c.expired())
                        {
                            NodePtr n = c.lock();
                            n->remove_input(ins);
                            n->imms().push_back(ins->value());
                        }
                    }
                    auto it = std::find(res.begin(), res.end(), ins);
                    if (it != res.end())
                    {
                        res.erase(it);
                    }
                }
            }
            func->body() = std::move(res);
            return func;
        }
        // resolve phi should be after RA, it makes problem more simple
        Function resolve_phi(Function func, std::map<int, std::vector<Node>> &movs)
        {
            std::vector<NodePtr> phis;
            for (auto &&ins : func->body())
            {
                if (ins->op() == Op::phi)
                {
                    phis.push_back(ins);
                }
            }
            for (auto &&phi : phis)
            {
                for (auto &&mov : movs[phi->id()])
                {
                    mov->set_reg(phi->reg());
                }
                auto it = std::find(func->body().begin(), func->body().end(), phi);
                if (it != func->body().end())
                {
                    func->body().erase(it);
                }
            }
            return func;
        }
        // instruction selection
        // gen eprologue for call then call and mov ret reg to dst reg
        Function fix_call(Function func)
        {
            std::vector<NodePtr> res;
            std::map<int, NodePtr> used_regs;
            for (auto &&ins : func->body())
            {
                if (ins->reg().is_rreg() && !ins->reg().spilled())
                {
                    used_regs[ins->reg().index()] = ins;
                }
                if (ins->op() == Op::call)
                {
                    std::vector<NodePtr> pop;
                    uint32_t num_spill_args = 0;
                    // set args [arg0, ..., argn]
                    if (ins->inputs().size() > 1)
                    {
                        // set args to [r0, ..., r7]
                        for (uint32_t i = 1; i < ins->inputs().size(); i++)
                        {
                            if (i >= 9)
                            {
                                break;
                            }
                            uint32_t idx = i - 1;
                            if (used_regs.count(idx))
                            {
                                auto reg = used_regs[idx]->reg();
                                res.push_back(NodeImpl::push(reg));
                                pop.push_back(NodeImpl::pop(reg));
                            }
                            auto inp = ins->inputs()[i];
                            if (inp->reg() != RReg(idx))
                            {
                                auto mov = NodeImpl::create(Op::mov, Dtype::u32, inp);
                                mov->reg() = RReg(idx);
                                res.push_back(mov);
                            }
                        }
                        // set [argn, ..., arg8] to stack
                        if (ins->inputs().size() > 9)
                        {
                            num_spill_args = ins->inputs().size() - 9;
                            for (int i = ins->inputs().size() - 1; i >= 9; i--)
                            {
                                auto inp = ins->inputs()[i];
                                res.push_back(NodeImpl::push(inp->reg()));
                            }
                        }
                    }
                    // push [ra, fp]
                    res.push_back(NodeImpl::push(RReg(RA)));
                    res.push_back(NodeImpl::push(RReg(FP)));
                    // move sp to fp
                    auto sp = NodeImpl::create(Op::alloc, Dtype::u32);
                    sp->set_reg(RReg(SP));
                    auto mov = NodeImpl::create(Op::mov, Dtype::u32, sp);
                    mov->set_reg(RReg(FP));
                    res.push_back(sp);
                    res.push_back(mov);
                    res.push_back(ins);
                    // pop [fp, ra]
                    res.push_back(NodeImpl::pop(RReg(FP)));
                    res.push_back(NodeImpl::pop(RReg(RA)));
                    // set ret of call to r0
                    if (ins->reg() != RReg(R0))
                    {
                        auto consumers = ins->consumers();
                        mov = NodeImpl::create(Op::mov, ins->dtype(), ins);
                        mov->set_reg(ins->reg());
                        ins->set_dtype(Dtype::u32);
                        ins->set_reg(RReg(R0));
                        for (auto &&c : consumers)
                        {
                            auto n = c.lock();
                            n->replace_input(ins, mov);
                        }
                        res.push_back(mov);
                    }
                    // pop [arg8, ..., argn]
                    if (num_spill_args)
                    {
                        auto add = NodeImpl::create(Op::add, Dtype::u32, sp);
                        add->set_reg(sp->reg());
                        add->imms().push_back(4 * num_spill_args);
                        res.push_back(add);
                    }
                    // pop [r7, ..., r0]
                    std::reverse(pop.begin(), pop.end());
                    res.insert(res.end(), pop.begin(), pop.end());
                }
                else
                {
                    res.push_back(ins);
                }
            }
            std::vector<NodePtr> body;
            for (auto &&ins : res)
            {
                if (ins->op() == Op::ret)
                {
                    if (ins->inputs().size() == 1 && ins->inputs()[0]->reg() != RReg(R0))
                    {
                        auto inp0 = ins->inputs()[0];
                        auto r0 = NodeImpl::create(Op::mov, Dtype::u32, inp0);
                        r0->set_reg(RReg(R0));
                        body.push_back(r0);
                        ins->remove_input(inp0);
                        ins->add_input(r0);
                    }
                    ins->set_reg(RReg(R0));
                    body.push_back(ins);
                }
                else
                {
                    body.push_back(ins);
                }
            }
            func->body() = std::move(body);
            return func;
        }
    };

    inline std::string emit(Function func, bool &spilled, bool use_free_args = false)
    {
        Emitter emitter;
        func = std::dynamic_pointer_cast<FunctionImpl>(func->clone());
        return emitter.emit(func, spilled, use_free_args);
    }

    inline InsPackages lower(Function func, bool &spilled, bool use_free_args = false)
    {
        Emitter emitter;
        func = std::dynamic_pointer_cast<FunctionImpl>(func->clone());
        return emitter.lower(func, spilled, use_free_args);
    }
} // namespace sbe
} // namespace tongue
