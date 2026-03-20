//===- CompassEmitter.h - Helpers to create C++ emitter -------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines helpers to emit C++ code using the EmitC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_CPP_CPPEMITTER_H
#define MLIR_TARGET_CPP_CPPEMITTER_H

#include "aipugc/Dialect.h"
#include "mlir/Support/LLVM.h"
#include <gcompiler_api.h>

namespace mlir {
class Operation;
namespace compass {

/// Translates the given operation to Compass IR. The operation or operations in
/// the region of 'op' need almost all be in EmitC dialect. The parameter
/// 'declareVariablesAtTop' enforces that all variables for op results and block
/// arguments are declared at the beginning of the function.
LogicalResult translateToCompass(Operation *op, aipubt::GraphPtr aipu_graph);

aipubt::GraphPtr translateToCompass(mlir::aipugc::FuncOp func);
LogicalResult translateToCompass(mlir::OwningOpRef<mlir::ModuleOp> &module);
// LogicalResult translateToCompass(mlir::OwningOpRef<mlir::ModuleOp> &module,
//                                  llvm::raw_ostream &os);
} // namespace compass
} // namespace mlir

#endif // MLIR_TARGET_CPP_CPPEMITTER_H
