//===- Input.cpp - MLIR Generation function 'mlirGen' for Input OP.
//-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements MLIR generation function 'mlirGen' for Input.
//
//===----------------------------------------------------------------------===//
#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

#include <numeric>

namespace mlir {
namespace aipugc {

class MLIRGenInput : public MLIRGenBase {
public:
  MLIRGenInput() {}
  ~MLIRGenInput() {}
  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    auto tensorType = utils::getType(builder, node->outputs()[0]);
    auto location = loc(builder, node);
    auto output = builder.create<InputOp>(location, tensorType);
    std::vector<mlir::Value> outputs;
    outputs.push_back(output);
    return outputs;
  }
};

REGISTER_MLIRGEN(Input);

} // namespace aipugc
} // namespace mlir