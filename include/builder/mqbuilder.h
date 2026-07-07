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
 * @Author: Kyle Huang
 * @Date: 2026-04-09 14:23:41
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2026-04-09 14:23:41
 */
#ifndef BUILDER_MQBUILDER_H
#define BUILDER_MQBUILDER_H

#include "builder/ibuilder.h"
#include "builder/readonly_param.h"
#include "core/memory_object.h"

namespace aipubt
{

enum BuildStage
{
    INIT,
    OPTIMIZE,
    SCHEDULE,
    KERNEL_ADAPT,
    MEM_ALLOC,
    CODE_GEN,
    ASSEMBLE,
    GEN_OBJ,
    DONE
};

struct BuildPhase
{
    std::string name;
    BuildStage stage;
    std::function<bool()> execute;
};

class MqBuilder : public IBuilder
{

public:
    MqBuilder(GraphPtr graph);
    ~MqBuilder() override;
    bool build() override;
    GraphPtr get_graph() override;
    CpsElfPtr get_elf() override;

    std::map<TensorPtr, uint32_t> get_tensor_index_map();

private:
    GraphPtr m_graph;
    CpsElfPtr m_elf;
    std::vector<BuildPhase> create_phases();
    bool phase_optimize();
    bool phase_schedule();
    bool phase_kernel_adapt();
    bool phase_mem_alloc();
    bool phase_code_gen();
    bool phase_assemble();
    bool phase_gen_obj();
};

} // namespace aipubt
#endif //!BUILDER_MQBUILDER_H