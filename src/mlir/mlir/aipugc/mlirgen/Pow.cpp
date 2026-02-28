//===- Pow.cpp - MLIR Generation function 'mlirGen' for Pow OP.
//-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements MLIR generation function 'mlirGen' for Pow.
//
//===----------------------------------------------------------------------===//

#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenPow : public MLIRGenBase {
public:
  MLIRGenPow() {}
  ~MLIRGenPow() {}
  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    auto ts_base = node->inputs()[0];
    auto ts_exponent = node->inputs()[1];
    auto ts_output = node->outputs()[0];

    auto outType = utils::getType(builder, ts_output);
    auto location = loc(builder, node);

    mlir::Value mlir_base = getsymbol(symbolTable, ts_base->name());
    mlir::Value mlir_exponent = getsymbol(symbolTable, ts_exponent->name());
    if (!mlir_base || !mlir_exponent) {
      llvm::errs() << "One of the inputs is nullptr: "
                   << (mlir_base ? "" : ts_base->name()) << " "
                   << (mlir_exponent ? "" : ts_exponent->name()) << "\n";
      return {};
    }

    DictionaryAttr op_attrs = ParamsToMlirAttr(builder, node);

    auto output = builder.create<PowOp>(location, outType, mlir_base,
                                        mlir_exponent, op_attrs);

    return {output};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_Pow, Pow);

} // namespace aipugc
} // namespace mlir
