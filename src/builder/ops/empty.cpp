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
 * @Date: 2020-01-29 14:25:35
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2021-04-06 17:59:05
 */
#include "builder/ro_generator.h"
#include "builder/text_generator.h"
#include "core/core.h"

namespace aipubt
{

void ro_generator_func_empty(NodePtr node, std::vector<ReadOnlyParam> &ro)
{
}
const std::string code_generator_func_empty(NodePtr node)
{
    return "";
}
BuilderText text_generator_func_empty(NodePtr node)
{
    return {};
}
#define REGISTER_A_EMPTY_OP_GENERATOR(optype)                                                \
    static TEXTGeneratorRegister _create_text_generator_##optype(optype,                     \
                                                                 text_generator_func_empty,  \
                                                                 code_generator_func_empty); \
    static ROGeneratorRegister _create_ro_generator_##optype(optype, ro_generator_func_empty);
REGISTER_A_EMPTY_OP_GENERATOR(OpType_Input);
REGISTER_A_EMPTY_OP_GENERATOR(OpType_SubGraphInput);
REGISTER_A_EMPTY_OP_GENERATOR(OpType_InternalSpliter);
REGISTER_A_EMPTY_OP_GENERATOR(OpType_Constant);
REGISTER_A_EMPTY_OP_GENERATOR(OpType_Empty);

} // namespace aipubt
