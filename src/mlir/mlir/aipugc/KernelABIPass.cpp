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

#include "aipugc/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>

using namespace mlir;

namespace {

// Rewrites kernel-style funcs from "value-returning memref" ABI to the cps device ABI
struct KernelABIPass : public PassWrapper<KernelABIPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(KernelABIPass)

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    FunctionType funcType = func.getFunctionType();
    if (funcType.getNumResults() == 0)
      return;

    Block &entry = func.getBody().front();
    unsigned numInputs = funcType.getNumInputs();

    // newInputTypes store the origin input types and append the origin return types.
    SmallVector<Type> newInputTypes(funcType.getInputs().begin(),
                                    funcType.getInputs().end());
    for (Type resultType : funcType.getResults()) {
      entry.addArgument(resultType, func.getLoc());
      newInputTypes.push_back(resultType);
    }

    // Collect all func::ReturnOp to avoid modify and delete ReturnOp.
    SmallVector<func::ReturnOp> returns;
    for (func::ReturnOp ret : func.getOps<func::ReturnOp>()) {
      returns.push_back(ret);
    }

    for (func::ReturnOp ret : returns) {
      for (auto it : llvm::enumerate(ret.getOperands())) {
        BlockArgument outArg = entry.getArgument(numInputs + it.index());
        Value operand = it.value();
        
        if (auto alloc = operand.getDefiningOp<memref::AllocOp>()) {
          operand.replaceAllUsesWith(outArg);
          alloc.erase();
        }
      }
      ret.getOperandsMutable().clear();
    }
    func.setFunctionType(FunctionType::get(&getContext(), newInputTypes, {}));
  }
};

}  // namespace

std::unique_ptr<Pass> mlir::aipugc::createKernelABIPass() {
  return std::make_unique<KernelABIPass>();
}