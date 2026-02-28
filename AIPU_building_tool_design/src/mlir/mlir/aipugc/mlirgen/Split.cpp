//===- Split.cpp - MLIR Generation function 'mlirGen' for Split OP.
//-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements MLIR generation function 'mlirGen' for Split.
//
//===----------------------------------------------------------------------===//
#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenSplit : public MLIRGenBase {
public:
  MLIRGenSplit() {}
  ~MLIRGenSplit() {}

  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    mlir::Value inputValue;
    if (!node->inputs().empty()) {
      inputValue = getsymbol(symbolTable, node->inputs()[0]->name());
    }
    if (!inputValue) {
      llvm::errs() << "SplitOp missing input: " << node->name() << "\n";
      return {};
    }

    llvm::SmallVector<mlir::Type> outputTypes;
    for (auto& output : node->outputs()) {
      outputTypes.push_back(utils::getType(builder, output));
    }

    std::vector<int64_t> splits = node->params()["splits"];
    auto splits_attr = builder.getDenseI64ArrayAttr(splits);

    int32_t axis = node->params()["axis"];
    auto axis_attr = builder.getI32IntegerAttr(axis);
    DictionaryAttr op_attrs = ParamsToMlirAttr(builder, node);

    auto location = loc(builder, node);
    auto splitOp = builder.create<SplitOp>(
        location,
        outputTypes,
        inputValue,
        splits_attr,
        axis_attr,
        op_attrs
    );

    // Split op has more than one result.
    std::vector<mlir::Value> results;
    for (auto result : splitOp->getResults()) {
      results.push_back(result);
    }

    return results;
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_Split, Split);

} // namespace aipugc
} // namespace mlir