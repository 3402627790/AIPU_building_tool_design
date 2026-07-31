//===- CacheUpdate.cpp - MLIRGen for CacheUpdate OP ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

#include <cstdint>

namespace mlir {
namespace aipugc {

class MLIRGenCacheUpdate : public MLIRGenBase {
public:
  MLIRGenCacheUpdate() = default;
  ~MLIRGenCacheUpdate() = default;

  std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    if (node->inputs().size() != 3) {
      llvm::errs() << "CacheUpdate expects 3 inputs, got "
                   << node->inputs().size() << ": " << node->name() << "\n";
      return {};
    }
    if (node->outputs().size() != 2) {
      llvm::errs() << "CacheUpdate expects 2 outputs, got "
                   << node->outputs().size() << ": " << node->name() << "\n";
      return {};
    }

    mlir::Value cache = getsymbol(symbolTable, node->inputs()[0]->name());
    mlir::Value positionId = getsymbol(symbolTable, node->inputs()[1]->name());
    mlir::Value value = getsymbol(symbolTable, node->inputs()[2]->name());
    if (!cache || !positionId || !value) {
      llvm::errs() << "CacheUpdate missing input: " << node->name() << "\n";
      return {};
    }

    auto currentCacheType = utils::getType(builder, node->outputs()[0]);
    auto totalCacheType = utils::getType(builder, node->outputs()[1]);
    auto location = loc(builder, node);

    const auto &params = node->params();
    int32_t blockIndex = 0;
    if (params.count("block_index"))
      blockIndex = params.at("block_index");
    int32_t axis = 2;
    if (params.count("axis"))
      axis = params.at("axis");

    auto blockIndexAttr = builder.getI32IntegerAttr(blockIndex);
    auto axisAttr = builder.getI32IntegerAttr(axis);
    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    auto cacheUpdateOp = builder.create<CacheUpdateOp>(
        location, currentCacheType, totalCacheType, cache, positionId, value,
        blockIndexAttr, axisAttr, opAttrs);

    return {cacheUpdateOp.getCurrentCache(), cacheUpdateOp.getTotalCache()};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_CacheUpdate, CacheUpdate);

} // namespace aipugc
} // namespace mlir
