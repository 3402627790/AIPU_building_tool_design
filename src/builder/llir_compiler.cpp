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

#include "builder/llir_compiler.h"

#include "LLIRToolCApi.h"

#include <dlfcn.h>

#include <cstdlib>
#include <cstring>
#include <iostream>

namespace aipubt
{
namespace
{

using CompileFn = int (*)(const char *, const char *, const AipuLLIRCompileOptions *);

std::string resolveLibraryPath()
{
    // 1) Environment variable override (for debugging or switching versions).
    const char *env = std::getenv("AIPU_LLIR_TOOL_LIB");
    if (env != nullptr && env[0] != '\0')
        return env;

    // 2) Compile-time injected absolute path.
#ifdef AIPU_LLIR_TOOL_LIB_PATH
    return AIPU_LLIR_TOOL_LIB_PATH;
#else
    // 3) Fallback: rely on the dynamic linker search path (LD_LIBRARY_PATH / rpath).
    return "libAIPULLIRToolC.so";
#endif
}

CompileFn loadCompileFn()
{
    static CompileFn fn = nullptr;
    static bool tried = false;
    if (tried)  // only Lazy-load once and cache it to avoid repeated symbol lookup.
        return fn;
    tried = true;

    const std::string path = resolveLibraryPath();
    void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr)
    {
        std::cerr << "[llir_compiler] dlopen failed: " << dlerror() << "\n";
        return nullptr;
    }

    // Clear any pre-existing error state before dlsym.
    dlerror();
    void *symbol = dlsym(handle, "aipu_llir_compile");
    const char *err = dlerror();
    if (err != nullptr || symbol == nullptr)
    {
        std::cerr << "[llir_compiler] dlsym(aipu_llir_compile) failed: "
                  << (err != nullptr ? err : "symbol not found") << "\n";
        return nullptr;
    }

    CompileFn resolved = nullptr;
    std::memcpy(&resolved, &symbol, sizeof(symbol));
    fn = resolved;
    return fn;
}

} // namespace

int compileLLIRToObject(const std::string &input,
                        const std::string &output,
                        const LLIRCompileOptions &options)
{
    CompileFn fn = loadCompileFn();
    if (fn == nullptr)
        return -1;

    AipuLLIRCompileOptions c;
    c.target_triple = options.targetTriple.empty() ? nullptr : options.targetTriple.c_str();
    c.cpu = options.cpu.empty() ? nullptr : options.cpu.c_str();
    c.features = options.features.empty() ? nullptr : options.features.c_str();
    c.opt_level = options.optLevel;
    c.function_sections = options.functionSections ? 1 : 0;

    return fn(input.c_str(), output.c_str(), &c);
}

} // namespace aipubt
