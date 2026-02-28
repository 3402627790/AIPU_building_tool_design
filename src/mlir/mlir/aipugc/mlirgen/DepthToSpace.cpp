//===- DepthToSpace.cpp - MLIR Generation function 'mlirGen' for DepthToSpace OP.
//-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements MLIR generation function 'mlirGen' for DepthToSpace.
//
//===----------------------------------------------------------------------===//
#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenDepthToSpace : public MLIRGenBase {
public:
  MLIRGenDepthToSpace() = default;
  ~MLIRGenDepthToSpace() = default;

  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override
  {
    mlir::Value inputValue;
    if (!node->inputs().empty()) {
      inputValue = getsymbol(symbolTable, node->inputs()[0]->name());
    }
    if (!inputValue) {
      llvm::errs() << "DepthToSpaceOp missing input: " << node->name() << "\n";
      return {};
    }

    auto outType = utils::getType(builder, node->outputs()[0]);
    auto location = loc(builder, node);

    const auto& params = node->params();

    int32_t block_x = params.at("block_size_x");
    int32_t block_y = params.at("block_size_y");
    auto blockXAttr = builder.getIntegerAttr(builder.getI32Type(), block_x);
    auto blockYAttr = builder.getIntegerAttr(builder.getI32Type(), block_y);

    std::string mode = params.at("mode");
    auto modeAttr = builder.getStringAttr(mode);

    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    auto output = builder.create<DepthToSpaceOp>(
        location,
        outType,
        inputValue,
        blockXAttr,
        blockYAttr,
        modeAttr,
        opAttrs
    );

    return {output};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_DepthToSpace, DepthToSpace);

} // namespace aipugc
} // namespace mlir