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
 * @Date: 2020-01-29 14:27:07 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-11-15 16:34:53
 */
#ifndef BUILDER_OPTIMIZER_H
#define BUILDER_OPTIMIZER_H

#include "core/graph.h"
#include "core/optype.h"
#include "core/plugin_loader.h"
namespace aipubt
{
class BuilderOptimizer
{
public:
    BuilderOptimizer() {}
    void optimize(GraphPtr g);
    // plugin flag only high 16bit works
    static BuilderOpPlugin *get_single_op_plugin(OpType t, uint32_t flag = 0);

private:
    PluginLoader *m_plugin_loader = PluginLoader::get_instance();
};
}; // namespace aipubt

#endif //!BUILDER_OPTIMIZER_H