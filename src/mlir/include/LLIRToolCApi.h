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

#ifndef LLIR_TOOL_CAPI_H
#define LLIR_TOOL_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

// Compile options for turning an LLVM IR file (.ll or .bc) into an object file.
// NULL or empty strings keep the library defaults.
typedef struct AipuLLIRCompileOptions
{
    const char *target_triple;     // default: "zhouyi-unknown-compass"
    const char *cpu;               // default: "" (target default CPU)
    const char *features;          // default: "" (target default features)
    unsigned int opt_level;        // 0-3, recommended 2
    int function_sections;         // non-zero enables -ffunction-sections behaviour
} AipuLLIRCompileOptions;

// Compile input_filename (.ll/.bc) into output_filename (.o/.obj).
// Returns 0 on success, non-zero on failure.
int aipu_llir_compile(const char *input_filename,
                      const char *output_filename,
                      const AipuLLIRCompileOptions *options);

// Returns the ABI version string (e.g. "1.0.0").
const char *aipu_llir_version(void);

#ifdef __cplusplus
}
#endif

#endif // LLIR_TOOL_CAPI_H
