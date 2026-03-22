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
 * @Date: 2020-01-29 14:27:16
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-04-01 17:39:38
 */
#ifndef CORE_PASS_MANAGER
#define CORE_PASS_MANAGER

#include "core/basepass.h"
#include "utils/utils.h"
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace aipubt
{
class PassManager;
class PassTuple;
using PassManagerPtr = std::shared_ptr<PassManager>;
using PassTuplePtr = std::shared_ptr<PassTuple>;

std::set<std::string> &get_disable_passes();
std::set<std::string> &get_serialize_before_pass();
std::set<std::string> &get_serialize_after_pass();
std::set<std::string> &get_svg_before_pass();
std::set<std::string> &get_svg_after_pass();
std::set<std::string> &get_pass_before_tiling();
class PassManager : public PassBase<PassManager>
{
public:
    template <typename T>
    static PassManagerPtr create(const std::string &name)
    {
        auto ptr = new T();
        ptr->m_name = name;
        return PassManagerPtr(ptr);
    }
    static PassManagerPtr create(const std::string &name)
    {
        return create<PassManager>(name);
    }
    bool operator()(GraphPtr g) const override;
    inline bool is_recursively()
    {
        return m_recursively;
    }
    inline void set_recursive(bool r)
    {
        m_recursively = r;
    }
    bool show_debug_info() const
    {
        return m_show_debug_info;
    }
    void set_show_debug_info(bool show)
    {
        m_show_debug_info = show;
    }
    PassManager &add(PassFunc p)
    {
        return add(Pass::create(p));
    }
    PassManager &add(const std::string &name, PassFunc p)
    {
        return add(Pass::create(name, p));
    }
    PassManager &add(PassPtr p);
    template <typename... Args>
    PassManager &add(PassPtr p, Args... args)
    {
        add(p);
        return add(args...);
    }
    template <typename... Args>
    PassManager &add(const std::string &name, PassFunc p, Args... args)
    {
        add(name, p);
        return add(args...);
    }
    template <typename... Args>
    PassManager &add(PassFunc p, Args... args)
    {
        add(p);
        return add(args...);
    }

    const std::vector<PassPtr> &passes() const;

protected:
    std::vector<PassPtr> m_passes;
    bool m_recursively = false;
    bool m_show_debug_info = false;
};

class PassTuple : public PassManager
{
    template <typename... Args>
    friend PassTuplePtr make_pass_tuple(Args... args);
    template <typename... Args>
    friend PassTuplePtr make_pass_tuple();

public:
protected:
    template <typename... Args>
    PassTuple(Args... args)
    {
        add(args...);
        m_name = "PassTuple(" + utils::join_string(",", m_passes) + ")";
    }
    PassManager &add() { return *this; }
    using PassManager::add;
};

template <typename... Args>
PassTuplePtr make_pass_tuple()
{
    PassTuplePtr ptr(new PassTuple(Args::create()...));
    return ptr;
}
template <typename... Args>
PassTuplePtr make_pass_tuple(Args... args)
{
    PassTuplePtr ptr(new PassTuple(args...));
    return ptr;
}
} // namespace aipubt
#endif //! CORE_PASS_MANAGER
