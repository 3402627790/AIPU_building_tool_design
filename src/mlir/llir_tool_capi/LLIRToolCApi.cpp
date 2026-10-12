//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2026 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------

#include "LLIRToolCApi.h"

#include "aipugc/LLIRTool.h"

namespace
{

aipubt::aipugc::LLIRCompileOptions toCppOptions(const AipuLLIRCompileOptions *options)
{
    aipubt::aipugc::LLIRCompileOptions cpp;
    if (options == nullptr)
        return cpp;

    if (options->target_triple != nullptr && options->target_triple[0] != '\0')
        cpp.triple = options->target_triple;
    if (options->cpu != nullptr && options->cpu[0] != '\0')
        cpp.cpu = options->cpu;
    if (options->features != nullptr && options->features[0] != '\0')
        cpp.features = options->features;
    cpp.optLevel = options->opt_level;
    cpp.functionSections = (options->function_sections != 0);
    return cpp;
}

} // namespace

extern "C" int aipu_llir_compile(const char *input_filename,
                                 const char *output_filename,
                                 const AipuLLIRCompileOptions *options)
{
    if (input_filename == nullptr || output_filename == nullptr)
        return -1;

    bool ok = aipubt::aipugc::compileLLIRToObject(input_filename, output_filename,
                                                  toCppOptions(options));
    return ok ? 0 : -1;
}

extern "C" const char *aipu_llir_version(void)
{
    return "1.0.0";
}
