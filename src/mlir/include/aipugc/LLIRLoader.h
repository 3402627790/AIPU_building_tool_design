#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir
{
class MLIRContext;
} // namespace mlir

namespace aipubt
{
namespace aipugc
{

mlir::OwningOpRef<mlir::ModuleOp> loadLLIRFile(mlir::MLIRContext &context, llvm::StringRef filename);

void printLLIRLoadSummary(mlir::ModuleOp module, llvm::raw_ostream &os);

} // namespace aipugc
} // namespace aipubt
