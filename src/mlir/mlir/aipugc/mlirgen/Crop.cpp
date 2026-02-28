//===- Crop.cpp - MLIR Generation function 'mlirGen' for Crop OP.
//-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements MLIR generation function 'mlirGen' for Crop.
//
//===----------------------------------------------------------------------===//
#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenCrop : public MLIRGenBase {
public:
  MLIRGenCrop() {}
  ~MLIRGenCrop() {}

  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    mlir::Value inputValue;
    if (!node->inputs().empty()) {
      inputValue = getsymbol(symbolTable, node->inputs()[0]->name());
    }
    if (!inputValue) {
      llvm::errs() << "CropOp missing input: " << node->name() << "\n";
      return {};
    }

    auto outType = utils::getType(builder, node->outputs()[0]);
    auto location = loc(builder, node);

    // Parsing the crops parameter (two-dimensional array)
    std::vector<std::vector<int64_t>> cropsVec = node->params()["crops"];

    llvm::SmallVector<mlir::Attribute> cropAttrs;
    for (auto& cropPair : cropsVec) {
      auto denseCrop = builder.getDenseI64ArrayAttr(cropPair);
      cropAttrs.push_back(denseCrop);
    }
    auto cropsAttr = builder.getArrayAttr(cropAttrs);

    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    auto output = builder.create<CropOp>(
        location,
        outType,
        inputValue,
        cropsAttr,
        opAttrs
    );

    return {output};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_Crop, Crop);

} // namespace aipugc
} // namespace mlir