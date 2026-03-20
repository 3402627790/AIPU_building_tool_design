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
 * @Date: 2020-02-05 11:51:27 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-04-02 11:14:29
 */

#ifndef CORE_PLUGIN_H
#define CORE_PLUGIN_H

#include <map>
#include <string>
#include <vector>

#define API_VERSION 0x00000002

#if defined(_WIN32)
#define PLUGIN_EXPORT __declspec(dllexport)
#else
#define PLUGIN_EXPORT
#endif
#define MAX_PLUGIN_NUMBERS 64

/**
 * @brief PluginType use 32bit for type id
 *        The low 16bit is used to distigurich which catergary of  plugin
 *        The high 16bit is used to store the plugin feateure.
 * 
 */
enum PlugInType
{
    PluginType_Empty = 0x0,
    PluginType_BuilderOp = 0x314,
    PluginType_IRCheckerOp = 0x315,
    PluginType_DtypeOp = 0x316,
};

enum BuilderPluginFlag
{
    GBPlugin_DS = 0x1 << 16,
    GBPlugin_Internal = 0x1 << 17,
};
#define GBPluginFlagAll (GBPlugin_DS | GBPlugin_Internal)

struct PluginInfo;
namespace aipubt
{
void _buildtool_register_plugin(PluginInfo *p);
}; //namespace aipubt
class _PluginRegisterHelper
{
public:
    _PluginRegisterHelper(PluginInfo *p)
    {
        aipubt::_buildtool_register_plugin(p);
    }
};
#define REGISTER_PLUGININFO_WITH_NAME(class_type, info_name) \
    struct info_name                                         \
    {                                                        \
        unsigned int api_version;                            \
        unsigned int plugin_version;                         \
        unsigned int plugin_type;                            \
        const char *name;                                    \
        const char *class_name;                              \
        class_type *(*init_handle)();                        \
    };
#define REGISTER_PLUGININFO(class_type) REGISTER_PLUGININFO_WITH_NAME(class_type, class_type##Info)
#define REGISTER_PLUGIN(class_type, plugin_version, plugin_type) \
    extern "C"                                                   \
    {                                                            \
        PLUGIN_EXPORT void *get_plugin##class_type()             \
        {                                                        \
            static class_type singleton;                         \
            return &singleton;                                   \
        }                                                        \
        PLUGIN_EXPORT PluginInfo plugin_infos##class_type = {    \
            API_VERSION,                                         \
            plugin_version,                                      \
            plugin_type,                                         \
            #class_type,                                         \
            #class_type,                                         \
            get_plugin##class_type,                              \
        };                                                       \
    }                                                            \
    static _PluginRegisterHelper _plugin_register_##class_type((PluginInfo *)&plugin_infos##class_type);

REGISTER_PLUGININFO_WITH_NAME(void, PluginInfo);
#endif //!CORE_PLUGIN_H