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
 * @Author: Alex Zheng
 * @Date: 2022-07-25 11:44:32
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2022-07-25 11:44:32
 */
#include "core/location.h"
#include "core/config.h"
#include "core/constants.h"
namespace aipubt
{

std::string &Location::name()
{
    return m_name;
}

std::string Location::name() const
{
    return m_name;
}

void Location::set_name(std::string name)
{
    m_name = name;
}

LocationList &Location::roots()
{
    return m_roots;
}

LocationList Location::roots() const
{
    return m_roots;
}

void Location::set_roots(LocationList roots)
{
    m_roots = roots;
    return;
}

void Location::add_roots(LocationList roots)
{
    m_roots.insert(m_roots.end(), roots.begin(), roots.end());
    return;
}

void Location::add_roots(TensorPtrList tensors)
{
    for (auto tensor : tensors)
    {
        if ((tensor != nullptr) && (tensor->node() != nullptr) && tensor->node()->loc() != nullptr)
            m_roots.push_back(tensor->node()->loc());
    }
    return;
}

void Location::add_roots(NodePtrList nodes)
{
    for (auto node : nodes)
    {
        if ((node != nullptr) && node->loc() != nullptr)
            m_roots.push_back(node->loc());
    }
    return;
}

void Location::set_type(OpType type)
{
    m_type = type;
}

OpType &Location::type()
{
    return m_type;
}

void Location::set_original(bool is_original)
{
    m_is_original = is_original;
}

bool Location::is_original()
{
    return m_is_original;
}

bool Location::through_pass(const std::string &pass_name)
{
    if (m_name == pass_name)
    {
        return true;
    }
    for (auto root : m_roots)
    {
        if (root != nullptr && root->through_pass(pass_name))
        {
            return true;
        }
    }
    return false;
}

std::string Location::to_string(LocationPtr loc)
{
    std::string ret;
    size_t count = 1;
    ret += loc->name();
    if (loc->roots().size() != 0)
    {
        ret += "::[";
        for (auto root : loc->roots())
        {
            if (root != nullptr)
            {
                ret += to_string(root);
            }
            if (count++ < loc->roots().size())
                ret += ", ";
        }
        ret += "]";
    }
    return ret;
}

std::string Location::to_string()
{
    std::string ret;
    size_t count = 1;
    ret += "[" + m_name + "]";
    if (m_roots.size() != 0)
    {
        ret += "::[";
        for (auto root : m_roots)
        {
            if (root != nullptr)
            {
                ret += to_string(root);
                if (count++ < m_roots.size())
                    ret += ", ";
            }
        }
        ret += "]";
    }
    return ret;
}

void Location::collect_names(LocationPtr loc, std::vector<std::string> &names)
{
    names.emplace_back(loc->name());
    for (auto root : loc->roots())
    {
        if (root != nullptr)
        {
            collect_names(root, names);
        }
    }
    return;
}

std::vector<std::string> Location::all()
{
    std::vector<std::string> names;
    names.emplace_back(m_name);
    for (auto root : m_roots)
    {
        if (root != nullptr)
        {
            collect_names(root, names);
        }
    }
    return names;
}

std::string Location::serialize()
{
    std::string ret;
    bool show_location = BuilderConfig::get<bool>(Constants::show_location);
    if (show_location)
    {
        ret += "location=" + to_string() + "\n";
    }
    return ret;
}

}; // namespace aipubt