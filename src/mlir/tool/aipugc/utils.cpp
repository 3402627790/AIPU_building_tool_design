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

#include "utils.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/Support/FileUtilities.h"
#include <iostream>
#include <memory>
#include <vector>

// The Zhouyi backend's instruction selector only matches device pointers in
// address space 1. The generic AffineToLLVM lowering emits opaque pointers in
// address space 0, which later fails with "Cannot select" on load/store/GEP.
// Rewrite every pointer in a kernel function to address space 1.
static void setZhouyiAddrSpace1(llvm::Function &F)
{
    llvm::LLVMContext &ctx = F.getContext();
    llvm::PointerType *as1 = llvm::PointerType::get(ctx, 1);

    for (llvm::Argument &arg : F.args())
    {
        if (arg.getType()->isPointerTy())
            arg.mutateType(as1);
    }

    for (llvm::BasicBlock &BB : F)
    {
        for (llvm::Instruction &I : BB)
        {
            if (I.getType()->isPointerTy())
                I.mutateType(as1);
            for (llvm::Value *op : I.operand_values())
            {
                if (op->getType()->isPointerTy())
                    op->mutateType(as1);
            }
        }
    }
}

int dumpLLVMIR(mlir::ModuleOp module, llvm::StringRef dumpFilename, llvm::StringRef llvmTarget, bool devicePointersLowered) {
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());
  mlir::registerNVVMDialectTranslation(*module->getContext());

  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR.\n";
    return -1;
  }

  if (llvmTarget == "zhouyi") {
    LLVMInitializeZhouyiTargetInfo();
    LLVMInitializeZhouyiTarget();
    LLVMInitializeZhouyiTargetMC();
    LLVMInitializeZhouyiAsmPrinter();

    llvm::Triple triple("zhouyi-unknown-compass");
    std::string error;
    const llvm::Target *theTarget = llvm::TargetRegistry::lookupTarget("", triple, error);
    if (!theTarget) {
      llvm::errs() << "Could not look up Zhouyi Target: " << error << "\n";
      return -1;
    }

    std::unique_ptr<llvm::TargetMachine> tm(theTarget->createTargetMachine(
      triple,
      "",   // CPU
      "",   // Features
      llvm::TargetOptions(),
      std::nullopt
    ));
    if (!tm) {
      llvm::errs() << "Could not create Zhouyi TargetMachine.\n";
      return -1;
    }

    mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(), tm.get());
    // Only device kernels may use the zhouyi_kernel calling convention.
    // Helper/library functions such as malloc keep the default C calling
    // convention. The Zhouyi calling convention requires a void return type,
    // so restricting the marker to defined void functions keeps the generated
    // module valid.
    for (llvm::Function &F : *llvmModule) {
      if (!F.isDeclaration() && F.getReturnType()->isVoidTy())
      {
        // The tile path constructs AS1 pointers through MLIR type conversion.
        // Retain the legacy baseline behavior here, but never mutate types on
        // the tile path: that would leave function and instruction types stale.
        if (!devicePointersLowered)
          setZhouyiAddrSpace1(F);
        F.setCallingConv(llvm::CallingConv::Zhouyi_Kernel);
      }
    }

  } else if (llvmTarget == "cpu") {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!tmBuilderOrError) {
      llvm::errs() << "Could not create JITTargetMachineBuilder.\n";
      return -1;
    }

    auto tmOrError = tmBuilderOrError->createTargetMachine();
    if (!tmOrError) {
      llvm::errs() << "Could not create TargetMachine.\n";
      return -1;
    }

    mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                          tmOrError.get().get());
  }


  if (devicePointersLowered && llvm::verifyModule(*llvmModule, &llvm::errs())) {
    llvm::errs() << "Invalid LLVM IR before optimization.\n";
    return -1;
  }
  auto optPipeline = mlir::makeOptimizingTransformer(3, 0, nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  
  if (devicePointersLowered && llvm::verifyModule(*llvmModule, &llvm::errs())) {
    llvm::errs() << "Invalid LLVM IR after optimization.\n";
    return -1;
  }
  llvm::errs() << "Below is LLVMIR:" << "\n" << *llvmModule << "\n";
  
  if (!dumpFilename.empty()) {
    std::error_code ec;
    llvm::raw_fd_ostream outfile(dumpFilename, ec);
    if (ec) {
      llvm::errs() << "Error opening file: " << dumpFilename << " - " << ec.message() << "\n";
      return -1;
    }
    llvmModule->print(outfile, nullptr);
    llvm::errs() << "Successfully saved LLVM IR to: " << dumpFilename << "\n";
  }
  return 0;
}

int runJit(mlir::ModuleOp module) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeAllTargetMCs();

  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();

  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());
  mlir::registerNVVMDialectTranslation(*module->getContext());

  auto optPipeline = mlir::makeOptimizingTransformer(
    /*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/nullptr
  );

  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;
  
  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  if (!maybeEngine) {
    llvm::errs() << "Failed to create ExecutionEngine\n";
    return -1;
  }
  auto &engine = maybeEngine.get();

  auto addSymbol = engine->lookup("Add");
  if (!addSymbol) {
    llvm::errs() << "Failed to lookup 'Add' function\n";
    return -1;
  }

  // Correct function signature: takes two float pointers, returns a float pointer
  auto addFunc = reinterpret_cast<float* (*)(float*, float*)>(*addSymbol);

  std::vector<float> dataA = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> dataB = {5.0f, 6.0f, 7.0f, 8.0f};

  float* rawResult = addFunc(dataA.data(), dataB.data());
  
  if (!rawResult) {
    std::cerr << "Function returned null pointer\n";
    return -1;
  }

  // Align pointer to 64-byte boundary (same as in LLVM IR)
  // The LLVM IR code aligns the malloc result to 64 bytes, so we need to do the same
  uintptr_t rawAddr = reinterpret_cast<uintptr_t>(rawResult);
  uintptr_t alignedAddr = (rawAddr + 63) & ~63;
  float* result = reinterpret_cast<float*>(alignedAddr);

  std::cout << "Result matrix:\n";
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      std::cout << result[i * 2 + j] << " ";
    }
    std::cout << "\n";
  }

  std::free(rawResult);

  return 0;
}
