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
 * @Date: 2020-01-29 14:27:12
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-04-11 11:31:48
 */
#include "core/pass_manager.h"
#include "core/config.h"
#include "core/core.h"
#include "utils/utils.h"
#include <cstdint>
#include <mutex>
#include <string>

namespace aipubt
{
std::vector<GraphPtr> get_all_graphs(GraphPtr g);
// std::set<std::string> &get_disable_passes()
// implemented in lowering.cpp

std::set<std::string> &get_serialize_before_pass()
{
    static std::set<std::string> passes;
    return passes;
}

std::set<std::string> &get_serialize_after_pass()
{
    static std::set<std::string> passes;
    return passes;
}

std::set<std::string> &get_svg_before_pass()
{
    static std::set<std::string> passes;
    return passes;
}

std::set<std::string> &get_svg_after_pass()
{
    static std::set<std::string> passes;
    return passes;
}

std::set<std::string> &get_pass_before_tiling()
{
    static std::set<std::string> passes;
    return passes;
}
inline PassManagerPtr is_pass_manager(PassPtr p)
{
    if (auto pm = std::dynamic_pointer_cast<PassManager>(p))
    {
        return pm;
    }
    return nullptr;
}

static inline bool pass_run_recursively(GraphPtr g, PassPtr p, bool show_debug_info = false)
{
    bool ret = false;
    for (auto n : g->nodes())
    {
        if (SubGraphNode::is_subgraph_node(n) &&
            !n->attrs().count(Constants::plugin_attrs_name))
        {
            auto sgn = n->get_shared_ptr<SubGraphNode>();
            ret |= pass_run_recursively(sgn->subgraph(), p, show_debug_info);
        }
    }
    if (show_debug_info)
    {
        if (auto pm = is_pass_manager(p))
        {
            pm->set_show_debug_info(true);
            ret = (*p)(g) || ret;
        }
        else
        {
            if ((*p)(g))
            {
                DEBUG() << "Pass " << p->name() << " affected the graph(" << g->name()
                        << ") which has " << g->nodes().size() << " nodes";
                ret = true;
            }
        }
    }
    else
    {
        ret = (*p)(g) || ret;
    }

    return ret;
}

PassManager &PassManager::add(PassPtr p)
{
    m_passes.push_back(std::move(p));
    return *this;
}

const std::vector<PassPtr> &PassManager::passes() const
{
    return m_passes;
}

bool PassManager::operator()(GraphPtr g) const
{
    static unsigned int count = 0;
    static bool enable_pass_perf = false;
    static char *enable_pass_perf_ = getenv(Constants::PASS_PERF);
    static auto serialize_before_pass = get_serialize_before_pass();
    static auto serialize_after_pass = get_serialize_after_pass();
    static auto png_before_pass = get_svg_before_pass();
    static auto png_after_pass = get_svg_after_pass();
    bool enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);
    bool is_dynamic = g->is_dynamic();
    if (enable_pass_perf_)
    {
        enable_pass_perf = true;
    }
    auto ret = false;
    for (auto &&p : m_passes)
    {
        auto &name = p->name();

        if (enable_ds)
        {
            if (get_disable_passes().count(name) ||
                (is_dynamic && p->type() == PassType_Static) ||
                (!is_dynamic && p->type() == PassType_Dynamic))
            {
                DEBUG() << "The pass " << name << " is disabled for graph " << g->name() << ".";
                continue;
            }
        }
        else if (get_disable_passes().count(name) || p->type() == PassType_Dynamic)
        {
            DEBUG() << "The pass " << name << " is disabled.";
            continue;
        }

        count++;
        auto fname = g->name() + "_" + name + "_" + std::to_string(count);

        if (contains(serialize_before_pass, name))
        {
            g->serialize("before_pass" + fname + ".def", "before_pass" + fname + ".cbin");
        }
        if (contains(png_before_pass, name))
        {
            g->to_svg("before_pass" + fname + ".svg");
        }
        uint64_t t0 = 0;
        if (enable_pass_perf)
        {
            t0 = utils::get_time_in_us();
        }
        if (m_recursively)
        {
            ret = pass_run_recursively(g, p, m_show_debug_info) || ret;
        }
        else if (m_show_debug_info)
        {
            if (auto pm = is_pass_manager(p))
            {
                pm->set_show_debug_info(true);
                ret = (*p)(g) || ret;
            }
            else
            {
                if ((*p)(g))
                {
                    DEBUG() << "Pass " << p->name() << " affected the graph(" << g->name()
                            << ") which has " << g->nodes().size() << " nodes";
                    ret = true;
                }
            }
        }
        else
        {
            ret = (*p)(g) || ret;
        }

        if (enable_pass_perf)
        {
            INFO() << "pass " << name << " costs: " << (utils::get_time_in_us() - t0) / 1000. << "ms";
        }
        if (serialize_after_pass.size() && contains(serialize_after_pass, name))
        {
            g->serialize("after_pass" + fname + ".def", "after_pass" + fname + ".cbin");
        }
        if (png_after_pass.size() && contains(png_after_pass, name))
        {
            g->to_svg("after_pass" + fname + ".svg");
        }
    }
    return ret;
}
} // namespace aipubt
