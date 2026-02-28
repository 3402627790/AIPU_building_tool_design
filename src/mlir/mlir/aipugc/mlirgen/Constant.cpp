//===- Constant.cpp - MLIR Generation function 'mlirGen' for Input OP.
//-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements MLIR generation function 'mlirGen' for Constant.
//
//===----------------------------------------------------------------------===//
#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenConstant : public MLIRGenBase {
public:
  MLIRGenConstant() {}
  ~MLIRGenConstant() {}
  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    auto const_tensor = node->constants()["weights"];
    auto location = loc(builder, node);
    auto output =
        ConstantTensorMlirGen(builder, location, const_tensor, symbolTable);
    return {output};
  }
};

REGISTER_MLIRGEN(Constant);

} // namespace aipugc
} // namespace mlir