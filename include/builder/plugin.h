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
#ifndef BUILDER_PLUGIN_H
#define BUILDER_PLUGIN_H

#include "builder/bldtypes.h"
#include "builder/readonly_param.h"
#include "core/optype.h"
#include "core/plugin.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

#define BUILDERPLUGIN_TYPE 0x314
namespace aipubt
{
struct ReadOnlyParam;
class BuilderOpPlugin;
enum OpType;
class Any;
using BaseNodePtrList = std::vector<BaseNodePtr>;

typedef std::pair<
    std::vector<std::pair<std::string, int>>,         //nodes
    std::vector<std::pair<std::string, std::string>>> //edges
    GraphMatcherPattern;

class BuilderOpPlugin
{
public:
    /**
     * @brief Destroy the Builder Op Plugin object
     *
     */
    virtual ~BuilderOpPlugin() = default;

    /**
     * @brief Get the graph pattern object
     *
     * @return const GraphMatcherPattern the match pattern for the plugin
     */
    virtual const GraphMatcherPattern get_graph_pattern() = 0;

    /**
     * @brief Get the score object
     *
     * @return uint32_t score, the plugin which has a higher score will be used first
     */
    virtual uint32_t get_score() = 0;

    /**
     * @brief Set the target object
     *
     * @param[in] target a string of the target code
     * @return true if the plugin supports this target
     * @return false if the plugin not supports this target
     */
    virtual bool set_target(const char *target) = 0;

    /**
     * @brief check the paramsters if the plugin supported
     *
     * @param[in] nodes  a list of Nodes for getting the paramsters
     * @return true if the plugin supports these node(s)
     * @return false if not
     */
    virtual bool check_params(BaseNodePtrList) { return true; }

    /**
     * @brief setup the plugin node
     *
     * @param[in] node the target node in the build process
     * @param[in] match_list the node list which match the plugin pattern
     */
    virtual void setup(BaseNodePtr, BaseNodePtrList){}

    /**
     * @brief generate the paramters for aipu code
     *
     * @param[in] node the target node in the build process
     * @param[in] nodes the node list which match the plugin pattern, you can read the original paramters from here
     * @return BuilderParams the generated paramters for AIPU code
     */
    virtual BuilderParams generate_params(BaseNodePtr node, BaseNodePtrList &nodes) = 0;

    /**
     * @brief generate the code name(the build binary file name of the plugin)
     *
     * @param[in] node the target node in the build process
     * @param[in] nodes the node list which match the plugin pattern, you can read the original paramters from here
     * @return const std::string the build binary file name of the plugin for this node
     */
    virtual const std::string generate_code_name(BaseNodePtr node, BaseNodePtrList &nodes) = 0;

    /**
     * @brief generate the descriptor for aipu code
     *
     * @param[in] node the target node in the build process
     * @param[in] nodes the node list which match the plugin pattern, you can read the original paramters from here
     * @return BuilderParams  the descriptor for this node
     */
    virtual BuilderParams generate_descriptor(BaseNodePtr, BaseNodePtrList &)
    {
        return BuilderParams();
    }

    /**
     * @brief generate the kernel function name(the entry function name)
     *        if return a empty string, it will trying to use the first kernel in elf for node,
     *        otherwise the begin of the binary will be treated as the kernel function offset.
     *
     * @param[in] node the target node in the build process
     * @param[in] nodes the node list which match the plugin pattern, you can read the original paramters from here
     * @return const std::string the kernel function name for this node
     */
    virtual const std::string generate_kernel(BaseNodePtr, BaseNodePtrList &)
    {
        return "";
    }

    /**
     * @brief internal plugin memory remapping
     *
     * @param[in] node the target node in the build process
     * @param[in] nodes the node list which match the plugin pattern, you can read the original paramters from here
     */
    virtual void memory_remapping(BaseNodePtr, BaseNodePtrList &)
    {
    }
};
} // namespace aipubt
#endif //!BUILDER_PLUGIN_H