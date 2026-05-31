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
 * @Date: 2023-03-24 10:38:34 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-04-03 16:53:44
 */
#pragma once

#include "tongue/symbolic.h"
#include <memory>
#include <sstream>

namespace tongue
{

namespace dyno
{
    using Operation = ExprImpl::Op;
    const std::string &to_string(Operation op);

    class Constraint
    {
    };
    // variable is a loop index
    class Var
    {
    private:
        struct VarImpl
        {
            std::string name;
            uint32_t value;
            Symbol sym;
        };
        std::shared_ptr<VarImpl> m_ptr;

    public:
        Var() = default;
        Var(const std::string &name)
        {
            m_ptr = std::shared_ptr<VarImpl>(new VarImpl({name, 0}));
        }
        const std::string &name() const { return m_ptr->name; }
        uint32_t value() const { return m_ptr->value; }
        template <typename T>
        T value() const { return *(T *)(&(m_ptr->value)); }
    };

    /**
     * @brief ComputeNode is for the body of a loop
     * a loop body may have several compute node, a compute node is designed as a DAG node.
     * for direct mapping, a statement in loop body is a compute node
     * It contains :
     *  1. loop index( i.g. i,j,k,m,n...)
     *  2. local or global variable as inputs
     *  3. one outputs for the compute node target
     *  4. contrains for the compute node valide condition
     */
    class Node
    {
    protected:
        struct NodeImpl
        {
            Operation op;
            uint32_t id;
            std::vector<Node> inputs;
            std::vector<Node> outputs;
            std::vector<Var> vars;
            std::vector<Constraint> constraints;
            const std::string to_string()
            {
                std::stringstream oss;
                oss << "%" << id << "[";
                for (auto i = 0u; i < vars.size(); ++i)
                {
                    oss << vars[i].name();
                    if (i != vars.size() - 1)
                    {
                        oss << ",";
                    }
                }
                oss << "]";
                return oss.str();
            }
            const std::string dump()
            {
                std::stringstream oss;
                oss << to_string() << "<-" << dyno::to_string(op)
                    << "(";
                for (auto i = 0u; i < inputs.size(); ++i)
                {
                    oss << inputs[i].to_string();
                    if (i != inputs.size() - 1)
                    {
                        oss << ",";
                    }
                }
                oss << ")";
                return oss.str();
            }
        };

        static uint32_t gen_id()
        {
            static uint32_t id = 0;
            return id++;
        }

    private:
        std::shared_ptr<NodeImpl> m_ptr;

    public:
        Node() = default;
        Node(const Node &other) : m_ptr(other.m_ptr) {}
        Node &operator=(const Node &other)
        {
            m_ptr = other.m_ptr;
            return *this;
        }
        Node(Operation op, const std::vector<Node> &inputs, const std::vector<Var> &vars, const std::vector<Constraint> &constr = std::vector<Constraint>())
        {
            m_ptr = std::make_shared<NodeImpl>();
            m_ptr->op = op;
            m_ptr->id = gen_id();
            m_ptr->inputs = inputs;
            m_ptr->vars = vars;
            m_ptr->constraints = constr;
        }
        const std::string to_string()
        {
            return m_ptr->to_string();
        }
        const std::string dump()
        {
            return m_ptr->dump();
        }
    };

    /**
     * @brief a loop order describes a loop
     *  var is the loop index
     *  size is the loop number
     *  tail is the size after vectorized 
     * 
     */
    struct LoopOrder
    {
        Var var;
        int64_t size;
        int64_t tail;
    };
    /**
     * @brief Loop describes a loop, it contains one more info: depth than LoopOrder.
     *  depth is which loop depth.
     * 
     */
    struct Loop : LoopOrder
    {
        int depth;
    };

    struct LoopNode
    {
        struct LoopNodeImpl
        {
            std::weak_ptr<LoopNodeImpl> parent;
            std::vector<LoopNode> children;
            bool is_loop = false;
            Loop loop;
            Node node;
            int depth = -1;
            std::map<std::string, std::string> attributes;
        };

        std::shared_ptr<LoopNodeImpl> m_ptr;
        LoopNode()
        {
            m_ptr = nullptr;
        }
        LoopNode(LoopNode parent, Node n)
        {
            m_ptr = std::make_shared<LoopNodeImpl>();
            if (parent)
            {
                m_ptr->parent = parent.m_ptr;
            }
            m_ptr->node = n;
        }
        LoopNode(LoopNode parent, Loop loop)
        {
            m_ptr = std::make_shared<LoopNodeImpl>();
            if (parent)
            {
                m_ptr->parent = parent.m_ptr;
            }
            m_ptr->loop = loop;
            m_ptr->is_loop = true;
        }
        LoopNodeImpl *operator->()
        {
            return m_ptr.get();
        }
        const LoopNodeImpl *operator->() const
        {
            return m_ptr.get();
        }
        bool operator==(std::nullptr_t nptr)
        {
            return m_ptr == nptr;
        }
        operator bool()
        {
            return m_ptr != nullptr;
        }
        void add_child(LoopNode child)
        {
            m_ptr->children.push_back(child);
        }
        bool merge(LoopNode &other)
        {
            return false;
        }
        const std::string dump();
    };

    struct LoopTree
    {
        const std::string dump() const
        {
            return "";
        }
        void walk();
        void add_node(LoopNode parent, LoopNode node)
        {
            nodes.push_back(node);
            if (parent == nullptr)
            {
                node->depth = 0;
            }
            else
            {
                node->depth = parent->depth + 1;
                parent.add_child(node);
            }
        }
        std::vector<LoopNode> nodes;
    };

    std::ostream &operator<<(std::ostream &oss, LoopNode &ln);
    std::ostream &operator<<(std::ostream &oss, const Loop &ln);
    std::ostream &operator<<(std::ostream &oss, const Node &ln);
} // namespace dyno
} // namespace tongue