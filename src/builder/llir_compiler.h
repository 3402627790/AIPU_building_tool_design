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

#ifndef BUILDER_LLIR_COMPILER_H
#define BUILDER_LLIR_COMPILER_H

#include <string>

namespace aipubt
{

// Compile options for turning an LLVM IR file (.ll/.bc) into an object file.
// Empty strings keep the default values inside the C ABI library.
struct LLIRCompileOptions
{
    std::string targetTriple;    // default: "zhouyi-unknown-compass"
    std::string cpu;             // default: ""
    std::string features;        // default: ""
    unsigned optLevel = 2;       // default: 2, range 0-3
    bool functionSections = true;
};

// Compile input (.ll/.bc) into output (.o/.obj) through the dlopen-ed C ABI library. 
// Returns 0 on success; -1 if the shared library or symbol cannot be loaded;
int compileLLIRToObject(const std::string &input,
                        const std::string &output,
                        const LLIRCompileOptions &options = LLIRCompileOptions());

} // namespace aipubt

#endif // BUILDER_LLIR_COMPILER_H
