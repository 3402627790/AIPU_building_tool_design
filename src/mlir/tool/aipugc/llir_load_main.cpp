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

#include "aipugc/Dialect.h"
#include "aipugc/LLIRTool.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>


namespace cl = llvm::cl;
using namespace aipubt;

static cl::opt<std::string> inputFilename(
    cl::Positional,
    cl::desc("<input .ll or .bc file>"),
    cl::value_desc("llir filename")
);

static cl::opt<bool> dumpMLIR(
    "dump-mlir",
    cl::desc("print the mlir llvm dialect module"),
    cl::init(false)
);

static cl::opt<std::string> emitObject(
    "emit",
    cl::desc("compile the llir file (.ll or .bc) to an object file (.o/.obj) at the given path"),
    cl::value_desc("output object file")
);

static cl::opt<std::string> targetTriple(
    "target-triple",
    cl::desc("target triple used to compile the llir file"),
    cl::init("zhouyi-unknown-compass")
);

static cl::opt<unsigned> compileOptLevel(
    "compile-opt-level",
    cl::desc("optimization level used when compiling the llir file (0-3)"),
    cl::init(2)
);

static cl::opt<bool> noFunctionSections(
    "no-function-sections",
    cl::desc("disable emitting each function into its own section"),
    cl::init(false)
);

static cl::list<std::string> priorities(
    "priority",
    cl::desc("Set kernel function priority, format <funcName>=<value>, repeatable"),
    cl::value_desc("func=value")
);

static cl::opt<int> defaultPriority(
    "default-priority",
    cl::desc("Default priority for kernel functions without explicit priority"),
    cl::init(0)
);

static bool parsePriorities(aipugc::KernelPriorityMap &out)
{
    for (const std::string &entry : priorities)
    {
        size_t eq = entry.find('=');
        if (eq == std::string::npos || eq == 0 || eq == entry.size() - 1)
        {
            llvm::errs() << "error: invalid --priority '" << entry
                         << "', expected format <funcName>=<value>\n";
            return false;
        }
        llvm::StringRef name(entry.data(), eq);
        llvm::StringRef valueStr(entry.data() + eq + 1, entry.size() - eq - 1);
        int value = 0;
        if (valueStr.getAsInteger(10, value))
        {
            llvm::errs() << "error: invalid priority value '" << valueStr
                         << "' for function '" << name << "', expected an integer\n";
            return false;
        }
        out[name] = value;
    }
    return true;
}

int main(int argc, char **argv)
{
    cl::ParseCommandLineOptions(argc, argv, "AIPUGC LLIR tool\n");

    aipugc::KernelPriorityMap priorityMap;
    if (!parsePriorities(priorityMap))
        return -1;

    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);

    mlir::MLIRContext context(registry);
    context.getOrLoadDialect<mlir::aipugc::AIPUDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

    mlir::OwningOpRef<mlir::ModuleOp> module =
        aipugc::loadLLIRFile(context, inputFilename, priorityMap, defaultPriority);
    if (!module)
    {
        llvm::errs() << "Failed to load llir file: " << inputFilename << "\n";
        return -1;
    }

    aipugc::printLLIRLoadSummary(*module, llvm::outs());

    if (!emitObject.empty())
    {
        aipugc::LLIRCompileOptions compileOptions;
        compileOptions.triple = targetTriple;
        compileOptions.optLevel = compileOptLevel;
        compileOptions.functionSections = !noFunctionSections;
        llvm::errs() << "Compiling " << inputFilename << " -> " << emitObject << "\n";
        if (!aipugc::compileLLIRToObject(inputFilename, emitObject, compileOptions))
        {
            llvm::errs() << "Failed to compile llir file to object file: "
                         << emitObject << "\n";
            return -1;
        }
        llvm::errs() << "Successfully emitted object file: " << emitObject << "\n";
    }

    if (dumpMLIR)
    {
        llvm::outs() << "\nImported MLIR module:\n";
        module->print(llvm::outs(), mlir::OpPrintingFlags().elideLargeElementsAttrs(32));
        llvm::outs() << "\n";
    }

    return 0;
}
