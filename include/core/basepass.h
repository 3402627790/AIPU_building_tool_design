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
 * @Last Modified time: 2022-05-13 15:51:55
 */

#ifndef CORE_BASEPASS_H
#define CORE_BASEPASS_H
#include <core/any.h>
#include <core/graph.h>
#include <core/logger.h>
#include <cstdint>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <vector>
namespace aipubt
{
class Graph;
using PassFunc = std::function<bool(GraphPtr g)>;
typedef bool (*Passfunc)(GraphPtr);
typedef void (*voidPassFunc)(GraphPtr);
using VoidPassFunc = std::function<void(GraphPtr g)>;
class Pass;
using PassPtr = std::shared_ptr<Pass>;

enum PassType
{
    PassType_Common = 0,
    PassType_Static = 1,  // for static shape passes
    PassType_Dynamic = 2, // for dynamic shape passes
};

class Pass : public PassFunc
{
protected:
public:
    virtual bool operator()(GraphPtr g) const = 0;
    virtual const std::string &name() const = 0;
    virtual PassType type() const = 0;
    virtual bool run_on_graph(GraphPtr g) const
    {
        return this->operator()(g);
    };
    static PassPtr create(const std::string &name, PassFunc func);
    static PassPtr create(PassFunc func);
};

std::ostream &operator<<(std::ostream &os, PassPtr ptr);

/// This class provides a CRTP wrapper around a base pass class to define
/// several necessary utility methods. This should only be used for passes that
/// are not suitably represented using the declarative pass specification(i.e.
/// tablegen backend).
template <typename PassT>
class PassBase : public Pass
{
public:
    static PassPtr create()
    {
        PassPtr ptr(new PassT());
        return ptr;
    }
    virtual bool operator()(GraphPtr) const
    {
        return false;
    };
    virtual const std::string &name() const
    {
        return m_name;
    }
    virtual PassType type() const
    {
        return PassType_Common;
    }

protected:
    virtual ~PassBase() = default;
    const std::vector<PassPtr> &dependencies() const;
    void add_dep(PassPtr dep);
    void remove_dep(const std::string &dep);
    void resolve_dep();
    std::string m_name = std::string(get_name_wo_ns<PassT>().data, get_name_wo_ns<PassT>().size);
    std::vector<PassPtr> m_dep; //dependencies
};

//all Func Pass starts with a "FPass_"
template <typename PassT, voidPassFunc func, PassType pass_type = PassType_Common>
class VFuncPass : public PassBase<PassT>
{
public:
    bool operator()(GraphPtr g) const override
    {
        func(g);
        return true;
    }
    PassType type() const override
    {
        return pass_type;
    }
    VFuncPass()
    {
        //all Func Pass starts with a "FPass_"
        this->m_name = std::string(get_name_wo_ns<PassT>().data + 6, get_name_wo_ns<PassT>().size - 6);
    }
};

template <typename PassT, Passfunc func, PassType pass_type = PassType_Common>
class FuncPass : public PassBase<PassT>
{
public:
    bool operator()(GraphPtr g) const override
    {
        return func(g);
    }
    PassType type() const override
    {
        return pass_type;
    }
    FuncPass()
    {
        //all Func Pass starts with a "FPass_"
        this->m_name = std::string(get_name_wo_ns<PassT>().data + 6, get_name_wo_ns<PassT>().size - 6);
    }
};

} //namespace aipubt
#endif //!CORE_BASEPASS_H