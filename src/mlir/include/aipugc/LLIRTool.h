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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>


namespace mlir
{
class MLIRContext;
} // namespace mlir

namespace aipubt
{
namespace aipugc
{

// Kernel function priority table: function name -> priority value.
using KernelPriorityMap = llvm::StringMap<int>;

// Compile options for emitting an object file from LLIR.
struct LLIRCompileOptions
{
    std::string triple = "zhouyi-unknown-compass";
    std::string cpu = "";
    std::string features = "";
    unsigned optLevel = 2;
    // Emit each function into its own section (.text.<funcname>), matching
    // cpscc's -ffunction-sections behavior.
    bool functionSections = true;
};

// Load an LLVM IR file into an MLIR module. The input may be either textual
// IR (.ll) or bitcode (.bc); llvm::parseIRFile auto-detects the format from
// the buffer contents.
mlir::OwningOpRef<mlir::ModuleOp> loadLLIRFile(
    mlir::MLIRContext &context, llvm::StringRef filename,
    const KernelPriorityMap &priorities, int defaultPriority);

// Compile an LLVM IR file (.ll or .bc) to an object file (.o/.obj) for the
// target triple. Internally it lowers the LLVM IR to assembly via the target
// backend, then assembles the text to an ELF relocatable object with the MC
// layer. Returns false on failure.
bool compileLLIRToObject(llvm::StringRef inputFilename,
                         llvm::StringRef outputFilename,
                         const LLIRCompileOptions &options = {});

void printLLIRLoadSummary(mlir::ModuleOp module, llvm::raw_ostream &os);


} // namespace aipugc

} // namespace aipubt
