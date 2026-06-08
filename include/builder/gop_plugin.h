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
 * @Date: 2020-01-29 14:23:45
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2021-07-26 15:12:14
 */
#pragma once

#include "builder/bldtypes.h"
#include "builder/readonly_param.h"
#include "core/optype.h"
#include "core/plugin.h"
#include "dtypes/op_data_types.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

#define GOPPLUGIN_TYPE 0x31
namespace aipubt
{
struct ReadOnlyParam;
class BuilderOpPlugin;
enum OpType;
class Any;
using NodePtrList = std::vector<NodePtr>;

class GopPlugin;
class GopImpl;
using GopPluginPtr = std::shared_ptr<GopPlugin>;
using GopImplPtr = std::shared_ptr<GopImpl>;
typedef std::pair<
    std::vector<std::pair<std::string, int>>,         // nodes
    std::vector<std::pair<std::string, std::string>>> // edges
    GraphMatcherPattern;

struct GPattern
{
    std::vector<std::pair<std::string, int>> nodes;
    std::vector<std::pair<std::string, std::string>> edges;

    GPattern() {}
    GPattern(OpType optype)
    {
        nodes.push_back(std::make_pair("0", optype));
    }
    GPattern(OpType op1, OpType op2)
    {
        nodes.push_back(std::make_pair("0", op1));
        nodes.push_back(std::make_pair("1", op2));
        edges.push_back(std::make_pair("0", "1"));
    }
    GPattern(const std::vector<std::pair<std::string, int>> &n, const std::vector<std::pair<std::string, std::string>> &e) : nodes(n), edges(e) {}
    GPattern(const GraphMatcherPattern &pattern) : nodes(pattern.first), edges(pattern.second) {}
    ~GPattern() {}
};

struct Dim3
{
    tongue::Expr x = 1, y = 1, z = 1;
    Dim3() {}
    Dim3(tongue::Expr x) : x(x) {}
    Dim3(tongue::Expr x, tongue::Expr y) : x(x), y(y) {}
    Dim3(tongue::Expr x, tongue::Expr y, tongue::Expr z) : x(x), y(y), z(z) {}
};

struct KernelInfo
{
    std::string kernel_name;
    Dim3 grid;
    Dim3 block;
    size_t sharemem_size;
    std::vector<ReadOnlyParam> args;
};

using KernelInfos = std::vector<KernelInfo>;

class GopImpl
{
public:
    virtual ~GopImpl() = default;

    virtual KernelInfos kernel_infos() = 0;

    // optional: use for turning performance
    virtual void update_prefer_cfg(const std::string &key, const Any &value)
    {
    }
    // optional: implement score
    virtual uint32_t score() const
    {
        return 0;
    }

    // optional: kernel cps bin file name
    virtual const std::string kernel_cpsbin()
    {
        return "";
    }

    // optional: implement flops
    virtual uint64_t flops() const
    {
        return 0;
    }
};

class GopPlugin
{
public:
    virtual ~GopPlugin() = default;
    virtual const std::vector<GPattern> patterns() = 0;
    virtual const std::vector<std::vector<Op_Dtype>> dtype_cap() = 0;
    virtual bool set_target(const std::string &target) = 0;
    virtual GopImplPtr get_impl(const NodePtrList &nodes, const GPattern &pattern = {}) = 0;
};

} // namespace aipubt