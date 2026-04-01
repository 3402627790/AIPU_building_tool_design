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
 * @Date: 2020-03-13 17:31:13
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2026-01-07 10:38:45
 */
#include "core/config.h"
#include "builder/aipu_binfile.h"
#include "builder/target.h"
#include "core/constants.h"
#include <core/default_config.h>
#include <core/logger.h>
namespace aipubt
{

BuilderConfig::BuilderConfig()
{
    init();
}

void BuilderConfig::init()
{
    m_config.clear();

    auto dc = get_default_config_dict();
    for (auto &&kv : dc)
    {
        m_config[std::string(kv.key)] = constexpr_value_to_any(kv.value);
    }
}
template <typename T>
static bool is_default(const std::map<std::string, Any> &defa, const std::string &key)
{
    return BuilderConfig::get<T>(key) == defa.at(key).cast<T>();
}

void BuilderConfig::target_changed()
{
    static BuilderConfig cfg;
    if (is_default<int>(cfg.m_config, Constants::gm_size))
    {
        auto target_name = get_target()->name();
        auto core_num = get_target()->core_num();
        int gm_size = 0;
        if (target_name == "X2_1204")
        {
            gm_size = core_num == 1 ? 1024 * 1024 : core_num == 3 ? 4 * 1024 * 1024 : 0;
        }
        else if ((target_name == "X3_1202") || (target_name == "X3P_1202"))
        {
            gm_size = core_num == 1 ? 2 * 1024 * 1024 : 0;
        }
        else if ((target_name == "X3_1304") || (target_name == "X3_1204") || (target_name == "X3P_1304") || (target_name == "X3P_1204"))
        {
            gm_size = core_num == 1 ? 4 * 1024 * 1024 : core_num == 2 ? 8 * 1024 * 1024 : core_num == 3 ? 8 * 1024 * 1024 : core_num == 4 ? 8 * 1024 * 1024 : 0;
        }
        else if ((target_name == "X3_1302") || (target_name == "X3P_1302"))
        {
            gm_size = core_num == 1 ? 2 * 1024 * 1024 : core_num == 2 ? 2 * 1024 * 1024 : core_num == 3 ? 2 * 1024 * 1024 : core_num == 4 ? 2 * 1024 * 1024 : 0;
        }
        else // nothing change, just return
        {
            return;
        }

        if (gm_size == 0)
        {
            WARN() << "default gm size not set, will be 0";
        }
        BuilderConfig::set(Constants::gm_size, gm_size);
    }
}

BuilderConfig *BuilderConfig::get_instance()
{
    static BuilderConfig m_inst;
    return &m_inst;
}
std::ostream &operator<<(std::ostream &oss, const BuilderConfig &cfg)
{
    oss << "{\n";
    for (auto kv : cfg.m_config)
    {
        oss << kv.first << ":";
        kv.second.print(oss) << ",\n";
    }
    oss << "}\n";
    return oss;
}

}; // namespace aipubt
