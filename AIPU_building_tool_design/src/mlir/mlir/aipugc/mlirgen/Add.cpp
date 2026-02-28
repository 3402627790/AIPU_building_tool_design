//===- Add.cpp - MLIR Generation function 'mlirGen' for Add OP.
//-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements MLIR generation function 'mlirGen' for Add.
//
//===----------------------------------------------------------------------===//

#include "aipugc/MLIRGenRegister.h"
#include <numeric>

namespace mlir {
namespace aipugc {

class MLIRGenAdd : public MLIRGenBase {
public:
  MLIRGenAdd() {}
  ~MLIRGenAdd() {}
  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    // 1. Extract tensor information from node
    // Get input tensors (lhs and rhs) and output tensor
    auto ts_lhs = node->inputs()[0];
    auto ts_rhs = node->inputs()[1];
    auto ts_output = node->outputs()[0];

    // Create MLIR type for output tensor using shape and data type
    auto outType = utils::getType(builder, ts_output);
    auto location = loc(builder, node);

    // 2. Resolve input values from symbol table
    // Look up MLIR Values for input tensors using their names
    mlir::Value mlir_lhs = getsymbol(symbolTable, ts_lhs->name());
    mlir::Value mlir_rhs = getsymbol(symbolTable, ts_rhs->name());
    if (!mlir_lhs || !mlir_rhs) {
      llvm::errs() << "One of the inputs is nullptr: "
                   << (mlir_lhs ? "" : ts_lhs->name()) << " "
                   << (mlir_rhs ? "" : ts_rhs->name()) << "\n";
      return {};
    }

    // 3. Process operation parameters
    DictionaryAttr op_attrs = ParamsToMlirAttr(builder, node);

    // 4. Create MLIR Add operation
    auto output =
        builder.create<AddOp>(location, outType, mlir_lhs, mlir_rhs, op_attrs);

    return {output};
  }
};

// Registration macro for MLIR generator
// Associates OpType_Add with our MLIRGenAdd implementation
// Enables framework to dispatch generation for Add operations
REGISTER_OP_MLIRGEN(aipubt::OpType_Add, Add);

} // namespace aipugc
} // namespace mlir