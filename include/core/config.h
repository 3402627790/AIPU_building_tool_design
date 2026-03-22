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
 * @Date: 2020-03-13 17:31:09
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2026-01-07 10:38:36
 */
#ifndef BUILDER_CONFIG_H
#define BUILDER_CONFIG_H
#include <core/any.h>
#include <core/constants.h>
#include <map>
#include <string>
#include <vector>

namespace aipubt
{

class BuilderConfig
{

public:
    static BuilderConfig *get_instance();

    template <typename T>
    static T get(const std::string &key)
    {
        if (get_instance()->m_config.count(key))
        {
            return cast_any<T>(get_instance()->m_config[key]);
        }
        else
        {
            return T();
        }
    }
    static Any get(const std::string &key)
    {
        if (get_instance()->m_config.count(key))
        {
            return get_instance()->m_config[key];
        }
        else
        {
            return Any();
        }
    }
    template <typename T>
    static void set(const std::string &key, const T &value)
    {
        get_instance()->m_config[key] = value;
    }
    static void set(const std::string &key, const Any &value)
    {
        get_instance()->m_config[key] = value;
    }
    static bool exist(const std::string &key)
    {
        return get_instance()->m_config.count(key);
    }

    static void erase(const std::string &key)
    {
        if (exist(key))
        {
            get_instance()->m_config.erase(key);
        }
    }

    static std::vector<std::string> keys()
    {
        std::vector<std::string> ret;
        for (auto kv : get_instance()->m_config)
        {
            ret.push_back(kv.first);
        }
        return ret;
    }
    static void reset()
    {
        get_instance()->init();
    }

    static void target_changed();
    friend std::ostream &operator<<(std::ostream &oss, const BuilderConfig &cfg);

private:
    BuilderConfig();
    ~BuilderConfig() {}
    void init();
    std::map<std::string, Any> m_config;
};

}; // namespace aipubt

#endif //!BUILDER_CONFIG_H