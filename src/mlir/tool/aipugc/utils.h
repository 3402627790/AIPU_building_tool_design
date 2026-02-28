#ifndef UTILS_H
#define UTILS_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

int dumpLLVMIR(mlir::ModuleOp module, llvm::StringRef dumpFilename);
int runJit(mlir::ModuleOp module);

#endif // UTILS_H