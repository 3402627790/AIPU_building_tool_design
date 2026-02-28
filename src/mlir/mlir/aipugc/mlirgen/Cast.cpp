//===- Cast.cpp - MLIR Generation function 'mlirGen' for Cast OP.
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements MLIR generation function 'mlirGen' for Cast.
//
//===----------------------------------------------------------------------===//
#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenCast : public MLIRGenBase {
public:
  MLIRGenCast() = default;
  ~MLIRGenCast() = default;

  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override 
  {
    mlir::Value inputValue;
    if (!node->inputs().empty()) {
      inputValue = getsymbol(symbolTable, node->inputs()[0]->name());
    }
    if (!inputValue) {
      llvm::errs() << "CastOp missing input: " << node->name() << "\n";
      return {};
    }

    auto outType = utils::getType(builder, node->outputs()[0]);
    auto location = loc(builder, node);

    const auto& params = node->params();
    
    if (!params.count("to_dtype") || !params.count("clip_mode")) {
      llvm::errs() << "CastOp missing required parameters\n";
      return {};
    }

    std::string to_dtype = params.at("to_dtype");
    auto dtypeAttr = builder.getStringAttr(to_dtype);

    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    auto output = builder.create<CastOp>(
        location,
        outType,
        inputValue,
        dtypeAttr,
        opAttrs
    );

    return {output};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_Cast, Cast);

} // namespace aipugc
} // namespace mlir