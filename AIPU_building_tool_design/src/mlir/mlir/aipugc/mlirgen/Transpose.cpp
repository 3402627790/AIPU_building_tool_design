//===- Transpose.cpp - MLIR Generation function 'mlirGen' for transpose OP.===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements MLIR generation function 'mlirGen' for Transpose.
//
//===----------------------------------------------------------------------===//

#include "aipugc/MLIRGenRegister.h"
#include <numeric>

namespace mlir {
namespace aipugc {

class MLIRGenTranspose : public MLIRGenBase {
public:
  MLIRGenTranspose() {}
  ~MLIRGenTranspose() {}
  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    // 1. Extract tensor information from node
    // Get first input tensor (index 0) and first output tensor
    auto ts_input = node->inputs()[0];
    auto ts_output = node->outputs()[0];
    // Create MLIR type for output tensor using shape and data type
    auto outType = utils::getType(builder, ts_output);
    auto location = loc(builder, node);

    // 2. Resolve input value from symbol table
    // Look up MLIR Value for input tensor using its name
    mlir::Value mlir_input = getsymbol(symbolTable, ts_input->name());
    if (!mlir_input) {
      llvm::errs() << "The input " << ts_input->name() << " is nullptr \n";
      return {};
    }

    // 3. Process operation parameters
    // Extract permutation array from node parameters
    std::vector<int32_t> bt_perm = node->params()["perm"];

    // Convert to MLIR-compatible attribute
    std::vector<int64_t> perm_i64(bt_perm.begin(), bt_perm.end());
    ArrayAttr perm = builder.getI64ArrayAttr(perm_i64);
    DictionaryAttr op_attrs = ParamsToMlirAttr(builder, node);

    // 4. Create MLIR Transpose operation
    // Using builder to construct operation with:
    // - Location information for debugging
    // - Precomputed output type
    // - Input tensor value
    // - Permutation attribute
    // - Additional operation attributes
    auto output = builder.create<TransposeOp>(location, outType, mlir_input,
                                              perm, op_attrs);

    return {output};
  }
};

// Registration macro for MLIR generator
// Associates OpType_Transpose with our MLIRGenTranspose implementation
// Enables framework to dispatch generation for Transpose operations
REGISTER_OP_MLIRGEN(aipubt::OpType_Transpose, Transpose);

} // namespace aipugc
} // namespace mlir