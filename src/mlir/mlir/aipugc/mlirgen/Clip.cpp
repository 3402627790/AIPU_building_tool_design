//===- Clip.cpp - MLIR Generation function 'mlirGen' for Clip OP.
//-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements MLIR generation function 'mlirGen' for Clip.
//
//===----------------------------------------------------------------------===//

#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenClip : public MLIRGenBase {
public:
  MLIRGenClip() = default;
  ~MLIRGenClip() = default;

  std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    auto ts_input = node->inputs()[0];
    auto ts_output = node->outputs()[0];

    auto outType = utils::getType(builder, ts_output);
    auto location = loc(builder, node);

    mlir::Value input = getsymbol(symbolTable, ts_input->name());
    if (!input) {
      llvm::errs() << "The input " << ts_input->name() << " is nullptr\n";
      return {};
    }

    float min_val = node->params()["clip_min"];
    float max_val = node->params()["clip_max"];

    auto minAttr = builder.getFloatAttr(builder.getF32Type(), min_val);
    auto maxAttr = builder.getFloatAttr(builder.getF32Type(), max_val);

    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    auto output =
        builder.create<ClipOp>(location, outType, input, minAttr, maxAttr, opAttrs);
    return {output};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_Clip, Clip);

} // namespace aipugc
} // namespace mlir
