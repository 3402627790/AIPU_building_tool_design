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
 * @Last Modified time: 2024-08-26 14:42:16
 */
#ifndef CORE_PLUGIN_MANAGER
#define CORE_PLUGIN_MANAGER

#include "core/plugin.h"
#include <cstdint>
#include <dlfcn.h>
#include <functional>
#include <map>
#include <string>

#define DEFAULT_LIB_PATH "plugin"
namespace aipubt
{

struct CPluginInfo
{
    unsigned int api_version;
    unsigned int plugin_version;
    unsigned int plugin_type;
    std::string name;
    std::string class_name;
    std::function<void *()> init_handle;
    CPluginInfo() {}
    CPluginInfo(const PluginInfo &p)
    {
        this->api_version = p.api_version;
        this->plugin_version = p.plugin_version;
        this->plugin_type = p.plugin_type;
        this->name = p.name;
        this->class_name = p.class_name;
        this->init_handle = p.init_handle;
    }
};
#ifdef REGISTER_PLUGININFO
#undef REGISTER_PLUGININFO
#endif
#define REGISTER_PLUGININFO(class_type)            \
    struct class_type##Info                        \
    {                                              \
        unsigned int api_version;                  \
        unsigned int plugin_version;               \
        unsigned int plugin_type;                  \
        std::string name;                          \
        std::string class_name;                    \
        std::function<class_type *()> init_handle; \
    };
//here we override the plugin info ~~ for code clean
class BuilderOpPlugin;
REGISTER_PLUGININFO(BuilderOpPlugin)
class CheckerOpPlugin;
REGISTER_PLUGININFO(CheckerOpPlugin)
class DtypeOpPlugin;
REGISTER_PLUGININFO(DtypeOpPlugin)
class GopPlugin;
REGISTER_PLUGININFO(GopPlugin)
void _buildtool_register_cppplugin(CPluginInfo p);
class PluginLoader
{
public:
    static PluginLoader *get_instance();
    static void add_pulgin(CPluginInfo *p);
    std::vector<CPluginInfo *> operator[](uint32_t plugin_type)
    {
        return m_plugins[plugin_type];
    }
    static void print_version();
    static std::string get_version();

private:
    PluginLoader();
    ~PluginLoader();
    void load_all_plugin(const char *lib_path);
    void load_plugin(const std::string &filename);
    std::map<uint32_t, std::vector<CPluginInfo *>> m_plugins;
    std::map<std::string, void *> m_handles;
    friend class BuilderOptimizer;
};

class BuiltinPluginRegister
{
public:
    BuiltinPluginRegister(CPluginInfo *p)
    {
        PluginLoader::add_pulgin(p);
    }
};
}; // namespace aipubt

#define REGISTER_BUILTIN_PLUGIN(class_type, plugin_version) \
                                                            \
    void *get_plugin_##class_type()                         \
    {                                                       \
        static class_type singleton;                        \
        return &singleton;                                  \
    }                                                       \
    static CPluginInfo plugin_infos_##class_type = {        \
        API_VERSION,                                        \
        plugin_version,                                     \
        BUILDERPLUGIN_TYPE,                                 \
        #class_type,                                        \
        #class_type,                                        \
        get_plugin_##class_type,                            \
    };                                                      \
    static BuiltinPluginRegister _builtin_plugin_register_##class_type(&plugin_infos_##class_type);

#endif //!CORE_PLUGIN_MANAGER