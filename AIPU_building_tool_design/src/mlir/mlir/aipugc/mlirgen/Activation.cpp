//===- Eltwise.cpp - MLIR Generation function 'mlirGen' for Eltwise OP.
//-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements MLIR generation function 'mlirGen' for Eltwise.
//
//===----------------------------------------------------------------------===//
#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenActivation : public MLIRGenBase {
public:
  MLIRGenActivation() {}
  ~MLIRGenActivation() {}
  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    mlir::Value input = getsymbol(symbolTable, node->inputs()[0]->name());
    if (!input) {
      llvm::errs() << "The input is nullptr \n ";
      return {};
    }
    auto outType = utils::getType(builder, node->outputs()[0]);
    auto location = loc(builder, node);
    std::string method = node->params()["method"];
    auto method_attr = builder.getStringAttr(method);
    DictionaryAttr op_attrs = ParamsToMlirAttr(
        builder, node,
        {"alpha", "beta", "lambd", "bias", "negative_slope_value", "lambd",
         "bias", "approximate"});
    auto output = builder.create<ActivationOp>(location, outType, input,
                                               method_attr, op_attrs);
    return {output};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_Activation, Activation);

} // namespace aipugc
} // namespace mlir