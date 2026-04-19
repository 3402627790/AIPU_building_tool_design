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
#include "core/plugin_loader.h"
#include "core/buildtool_version.h"
#include "core/core.h"
#include "core/logger.h"
#include "utils/utils.h"
#include <cstdint>
#include <dirent.h>
#include <string>

#define PATH_SPLITTER "/"
#define DLIB_SUFFIX ".so"
#ifdef _WIN32
#define PATH_SPLITTER "\\"
#define DLIB_SUFFIX ".dll"
#endif

#define AIPUPLUGIN_PATH "AIPUPLUGIN_PATH"
namespace aipubt
{

static std::map<std::string, CPluginInfo> _all_plugin_infos;
static std::map<size_t, std::string> plugin_type_name =
    {
        // {PluginType_Empty, "Empty"},
        {PluginType_BuilderOp, "Builder"},
        {PluginType_IRCheckerOp, "Checker"},
};
void _buildtool_register_plugin(PluginInfo *p)
{
    if (!_all_plugin_infos.count(p->name))
    {
        CPluginInfo mp(*p);
        _all_plugin_infos[p->name] = mp;
    }
}
void _buildtool_register_cppplugin(CPluginInfo p)
{
    if (!_all_plugin_infos.count(p.name))
    {
        _all_plugin_infos[p.name] = p;
    }
}

typedef PluginInfo *(*PluginGetter)();
typedef uint32_t (*PluginNumGetter)();
static inline bool ends_with(const std::string &s, const std::string &suffix)
{
    return s.rfind(suffix) == (s.size() - suffix.size()) && s.size() >= suffix.size();
}

PluginLoader::PluginLoader()
{
    auto plugin_path = utils::get_exe_dir() + "/plugin";

    //use cwd as prior
    load_all_plugin(DEFAULT_LIB_PATH);
    // second env variable
    char *env_plugin_path = getenv(AIPUPLUGIN_PATH);
    if (env_plugin_path)
    {
        for (auto p : utils::split_string(env_plugin_path, ':'))
        {
            load_all_plugin(p.c_str());
        }
    }
    //default place as third
    load_all_plugin(plugin_path.c_str());
    for (auto &&kv : _all_plugin_infos)
    {
        CPluginInfo *plugin_infos = &kv.second;
        // Check the API version
        if (plugin_infos->api_version != API_VERSION)
        {
            WARN() << "Plugin load error! Plugin " << plugin_infos->name
                   << "'s API is not compatible with the BuildTool API. The plugin's API version is "
                   << plugin_infos->api_version
                   << " BuildTool expect a Plugin with API version:"
                   << API_VERSION
                   << ". Please Try to recompile your plugin with the BuildTool version:"
                   << VERSION;
            continue;
        }
        auto type = plugin_infos->plugin_type & 0xffff;
        auto &plugins = m_plugins[type];
        // if not registered
        if (std::find(plugins.begin(), plugins.end(), plugin_infos) == plugins.end())
        {
            plugins.push_back(plugin_infos);
        }
    }

    // for (auto &plugins : m_plugins)
    // {
    //     std::ostringstream oss;
    //     for (auto p : plugins.second)
    //     {
    //         oss << p->name << " ";
    //     }
    //     if (plugin_type_name.count(plugins.first))
    //     {
    //         DEBUG() << "loaded " << plugin_type_name[plugins.first] << ", plugins(num=" << plugins.second.size() << "): " << oss.str();
    //     }
    //     else
    //     {
    //         DEBUG() << "loaded plugin type:" << plugins.first << ", plugins(num=" << plugins.second.size() << "): " << oss.str();
    //     }
    // }
}

PluginLoader::~PluginLoader()
{
    for (auto plugin : m_handles)
    {
        if (plugin.second)
        {
            // DEBUG() << "unload plugin: " << plugin.first;
            if (dlclose(plugin.second) != 0)
            {
                dlerror();
            }
        }
    }
}
void PluginLoader::load_all_plugin(const char *lib_path)
{
    DIR *dir;
    struct dirent *ent;
    if (utils::path_exists(lib_path))
    {
        if ((dir = opendir(lib_path)) != NULL)
        {
            while ((ent = readdir(dir)) != NULL)
            {
                std::string filename = ent->d_name;
                if (ends_with(filename, DLIB_SUFFIX))
                {
                    filename = std::string(lib_path) + PATH_SPLITTER + filename;
                    // DEBUG() << "load " << filename;
                    load_plugin(filename);
                }
            }
            closedir(dir);
        }
    }
}

void PluginLoader::load_plugin(const std::string &filename)
{
    if (m_handles.count(filename))
    {
        //loaded~~
        return;
    }
    void *handle = nullptr;
    handle = dlopen(filename.c_str(), RTLD_LAZY);
    if (handle == NULL)
    {
        DEBUG() << "Load plugin failed: " << filename;
        dlerror();
    }
    else
    {
        m_handles[filename] = handle;
    }
}

PluginLoader *PluginLoader::get_instance()
{
    static PluginLoader singleton;
    return &singleton;
}

void PluginLoader::add_pulgin(CPluginInfo *p)
{
    auto pl = get_instance();
    pl->m_plugins[p->plugin_type & 0xffff].push_back(p);
}
void PluginLoader::print_version()
{
    std::cout << get_version();
}
std::string PluginLoader::get_version()
{
    auto pl = get_instance();
    std::ostringstream oss;
    oss << std::left << "loaded plugin file:\n\t";
    for (auto plugin : pl->m_handles)
    {
        if (plugin.second)
        {
            oss << plugin.first << "\n\t";
        }
    }
    oss << "\n";
    for (auto &plugins : pl->m_plugins)
    {
        if (plugins.second.size() == 0)
        {
            continue;
        }

        if (plugin_type_name.count(plugins.first))
        {
            oss << std::left << std::setw(24) << "plugin types:" + plugin_type_name[plugins.first] << "\tversion:\n";
        }
        else
        {
            oss << std::left << std::setw(24) << "plugin types:" + std::to_string(plugins.first) << "\tversion:\n";
        }
        for (auto plugin : plugins.second)
        {
            oss << std::left << std::setw(24) << plugin->name
                << "\tversion:" << ((plugin->plugin_version >> 24) & 0xff) << "."
                << ((plugin->plugin_version >> 16) & 0xff) << "."
                << (plugin->plugin_version & 0xffff) << "\n";
        }
        oss << "\n";
    }

    return oss.str();
}

}; // namespace aipubt
