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

class MLIRGenSoftmax : public MLIRGenBase {
public:
  MLIRGenSoftmax() {}
  ~MLIRGenSoftmax() {}
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
    int32_t axis = node->params()["axis"];
    float beta = node->params()["beta"];
    auto axis_attr = builder.getIntegerAttr(builder.getIntegerType(32), axis);
    auto beta_attr = builder.getFloatAttr(builder.getF32Type(), beta);
    DictionaryAttr op_attrs =
        ParamsToMlirAttr(builder, node, {"quantize_method"});
    auto output = builder.create<SoftmaxOp>(location, outType, input, axis_attr,
                                            beta_attr, op_attrs);
    return {output};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_Softmax, Softmax);

} // namespace aipugc
} // namespace mlir