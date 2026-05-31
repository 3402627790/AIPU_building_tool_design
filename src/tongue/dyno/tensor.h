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

#include "ir.h"
#include "tongue/symbolic.h"
#include <memory>

namespace tongue
{

namespace dyno
{

    /**
    * @brief Tensor for dyno
    * 
    * Tensor is the basic(or unique) element of dyno IR
    * each operator only applies to a tensor
    * 
    */
    using Operation = ExprImpl::Op;

    class Tensor
    {
        struct TensorImpl
        {
            using TensorPtr = std::shared_ptr<TensorImpl>;
            // uint64_t
            Operation op = Operation::Constant;
            std::vector<Symbol> shape;
            std::vector<int64_t> sizes;
            std::vector<Constraint> constraints;
            std::vector<Tensor> deps;
            void add_dim(Symbol s)
            {
                shape.push_back(s);
            }
            template <typename... IList>
            void add_dim(Symbol di, IList... new_shape)
            {
                add_dim(di);
                add_dim(new_shape...);
            }
            std::string to_string()
            {
                std::string ret = "Tensor:" + ExprImpl::op_to_string(op) + "[";
                for (auto i = 0u; i < shape.size(); ++i)
                {
                    ret += shape[i].name();
                    if (i != shape.size() - 1)
                    {
                        ret += ",";
                    }
                    else
                    {
                        ret += "]";
                    }
                }
                return ret;
            }
        };
        using TensorPtr = std::shared_ptr<TensorImpl>;
        TensorPtr m_ptr;
        Tensor reduce(Operation op, const Symbol &var)
        {
            std::vector<Symbol> new_shape;
            for (auto i : m_ptr->shape)
            {
                if (i == var)
                {
                    continue;
                }
                new_shape.push_back(i);
            }
            return Tensor(op, new_shape, {*this});
        }

    public:
        Tensor(const Symbol &s)
        {
            m_ptr = std::make_shared<TensorImpl>();
            m_ptr->shape.push_back(s);
        }
        template <typename... IList>
        Tensor(const Symbol &di, IList... new_shape) : Tensor(di)
        {
            m_ptr->add_dim(new_shape...);
        }
        Tensor(Operation op, const std::vector<Symbol> &new_shape, const std::vector<Tensor> &deps)
        {
            m_ptr = std::make_shared<TensorImpl>();
            m_ptr->op = op;
            m_ptr->shape = new_shape;
            m_ptr->deps = deps;
        }
        Tensor sum(const Symbol &axis)
        {
            return reduce(Operation::Add, axis);
        }
        Tensor max(const Symbol &axis)
        {
            return reduce(Operation::Max, axis);
        }
        std::string to_string()
        {
            return m_ptr->to_string();
        }
    };

} // namespace dyno
} // namespace tongue