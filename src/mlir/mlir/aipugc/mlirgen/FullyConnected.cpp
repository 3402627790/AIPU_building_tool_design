//===- Conv2D.cpp - MLIR Generation function 'mlirGen' for Input OP.
//-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements MLIR generation function 'mlirGen' for Conv2D.
//
//===----------------------------------------------------------------------===//

#include "aipugc/MLIRGenRegister.h"
#include <numeric>

namespace mlir {
namespace aipugc {

class MLIRGenFullyConnected : public MLIRGenBase {
public:
  MLIRGenFullyConnected() {}
  ~MLIRGenFullyConnected() {}
  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    auto ts_input = node->inputs()[0];
    auto ts_weights = node->constants()["weights"];
    auto ts_biases = node->constants()["biases"];
    auto ts_output = node->outputs()[0];
    auto out_type = utils::getType(builder, ts_output);
    auto location = loc(builder, node);
    mlir::Value mlir_input = getsymbol(symbolTable, ts_input->name());
    if (!mlir_input) {
      llvm::errs() << "The input " << ts_input->name() << "is nullptr \n";
      return {};
    }
    mlir::Value mlir_weights = getsymbol(symbolTable, ts_weights->name());
    if (!mlir_weights) {
      mlir_weights =
          ConstantTensorMlirGen(builder, location, ts_weights, symbolTable);
    }
    mlir::Value mlir_biases = getsymbol(symbolTable, ts_biases->name());
    if (!mlir_biases) {
      mlir_biases =
          ConstantTensorMlirGen(builder, location, ts_biases, symbolTable);
    }
    DictionaryAttr op_attrs = ParamsToMlirAttr(builder, node);
    auto output = builder.create<FullyConnectedOp>(
        location, out_type, mlir_input, mlir_weights, mlir_biases, op_attrs);
    return {output};
  }
};

REGISTER_MLIRGEN(FullyConnected);

} // namespace aipugc
} // namespace mlir