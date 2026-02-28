#include "utils.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/Support/FileUtilities.h"
#include <iostream>
#include <vector>

int dumpLLVMIR(mlir::ModuleOp module, llvm::StringRef dumpFilename) {
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());
  mlir::registerNVVMDialectTranslation(*module->getContext());

  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR.\n";
    return -1;
  }

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
  auto optPipeline = mlir::makeOptimizingTransformer(3, 0, nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
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