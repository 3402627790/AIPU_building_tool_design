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
 * @Date: 2022-05-13 14:36:30 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-08-21 16:11:08
 */

#include <core/placeholder.h>

namespace aipubt
{
Placeholder::Placeholder() : Tensor(nullptr, 0)
{
}

Placeholder::Placeholder(MemoryObjectPtr t) : Placeholder()
{
    m_offset->base_offset = t;
}

Placeholder::Placeholder(uint32_t v) : Placeholder()
{
    m_offset->relative_offset = v;
}

Placeholder::Placeholder(MemoryObjectPtr t, uint32_t v) : Placeholder()
{
    m_offset->relative_offset = v;
    m_offset->base_offset = t;
}

Placeholder::Placeholder(TensorPtr t) : Placeholder()
{
    m_offset->base_offset = t->memory_offset();
}

Placeholder::Placeholder(PlaceholderPtr t) : Placeholder()
{
    m_offset->base_offset = t->memory_offset();
}

Placeholder::Placeholder(TensorPtr t, uint32_t v) : Placeholder()
{
    m_offset->base_offset = t->memory_offset();
    m_offset->relative_offset = v;
}

Placeholder &Placeholder::operator=(uint32_t v)
{
    m_offset->relative_offset = v;
    m_offset->base_offset = nullptr; //clear the base
    return *this;
}

Placeholder &Placeholder::operator=(MemoryObjectPtr v)
{
    m_offset->relative_offset = 0;
    m_offset->base_offset = v;
    return *this;
}

Placeholder &Placeholder::operator=(TensorPtr t)
{
    m_offset->relative_offset = 0;
    m_offset->base_offset = t->memory_offset();
    return *this;
}

Placeholder &Placeholder::operator=(const Placeholder &t)
{
    m_offset->relative_offset = 0;
    m_offset->base_offset = ((Placeholder)t).memory_offset();
    return *this;
}

Placeholder &Placeholder::operator+=(uint32_t v)
{
    m_offset->relative_offset += v;
    return *this;
}

Placeholder Placeholder::operator+(uint32_t v)
{
    Placeholder o(this->memory_offset());
    o += v;
    return o;
}

PlaceholderPtr operator+(PlaceholderPtr p, uint32_t v)
{
    auto r = std::make_shared<Placeholder>(p->memory_offset());
    if (p->attrs().count(Constants::expr_offset))
    {
        Expr eo = p->get_attr(Constants::expr_offset);
        r->set_attr(Constants::expr_offset, eo + v);
    }
    else
    {
        r->memory_offset()->relative_offset += v;
    }
    return r;
}

PlaceholderPtr operator+(PlaceholderPtr p, Expr o)
{
    auto r = std::make_shared<Placeholder>(p->memory_offset());
    if (p->attrs().count(Constants::expr_offset))
    {
        Expr eo = p->get_attr(Constants::expr_offset);
        r->set_attr(Constants::expr_offset, eo + o);
    }
    else
    {
        r->set_attr(Constants::expr_offset, o);
    }
    return r;
}

PlaceholderPtr operator-(PlaceholderPtr p, uint32_t v)
{
    auto r = std::make_shared<Placeholder>(p->memory_offset());
    if (p->attrs().count(Constants::expr_offset))
    {
        Expr eo = p->get_attr(Constants::expr_offset);
        r->set_attr(Constants::expr_offset, eo - v);
    }
    else
    {
        r->memory_offset()->relative_offset -= v;
    }
    return r;
}

PlaceholderPtr operator-(PlaceholderPtr p, Expr o)
{
    auto r = std::make_shared<Placeholder>(p->memory_offset());
    if (p->attrs().count(Constants::expr_offset))
    {
        Expr eo = p->get_attr(Constants::expr_offset);
        r->set_attr(Constants::expr_offset, eo - o);
    }
    else
    {
        r->set_attr(Constants::expr_offset, -o);
    }
    return r;
}

// struct placeholder
// {
//     placeholder()
//     {
//         auto a = std::make_shared<Placeholder>(3);
//         PlaceholderPtr b = std::make_shared<Placeholder>();
//         INFO() << "=======================";
//         INFO() << a->offset();
//         INFO() << b->offset() << " " << b.get();
//         *a = 18;
//         *b = *a;
//         INFO() << b->offset() << " " << b.get();
//         *b = 20;
//         INFO() << b->offset() << " " << b.get();
//         *b = *a + 20;
//         INFO() << b->offset() << " " << b.get();
//         *b += 30;
//         INFO() << b->offset() << " " << b.get();
//         INFO() << "=======================";
//     }
// };

// static placeholder test;

} //namespace aipubt