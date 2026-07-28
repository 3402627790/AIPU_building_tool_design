//===- ScaledDotProductAttention.cpp - MLIRGen for SDPA OP ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

#include <cmath>

namespace mlir {
namespace aipugc {

class MLIRGenScaledDotProductAttention : public MLIRGenBase {
public:
  MLIRGenScaledDotProductAttention() = default;
  ~MLIRGenScaledDotProductAttention() = default;

  std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    if (node->inputs().size() < 3 || node->inputs().size() > 4) {
      llvm::errs() << "ScaledDotProductAttention node expects 3 or 4 inputs, "
                   << "but got " << node->inputs().size() << ": "
                   << node->name() << "\n";
      return {};
    }

    mlir::Value query = getsymbol(symbolTable, node->inputs()[0]->name());
    mlir::Value key = getsymbol(symbolTable, node->inputs()[1]->name());
    mlir::Value value = getsymbol(symbolTable, node->inputs()[2]->name());
    if (!query || !key || !value) {
      llvm::errs() << "ScaledDotProductAttentionOp missing query/key/value: "
                   << node->name() << "\n";
      return {};
    }

    llvm::SmallVector<mlir::Value> attnMask;
    if (node->inputs().size() == 4) {
      mlir::Value mask = getsymbol(symbolTable, node->inputs()[3]->name());
      if (!mask) {
        llvm::errs() << "ScaledDotProductAttentionOp missing attn_mask: "
                     << node->inputs()[3]->name() << "\n";
        return {};
      }
      attnMask.push_back(mask);
    }

    auto outType = utils::getType(builder, node->outputs()[0]);
    auto location = loc(builder, node);
    const auto &params = node->params();

    bool isCausal = false;
    if (params.count("is_causal"))
      isCausal = params.at("is_causal");

    bool enableGqa = false;
    if (params.count("enable_gqa"))
      enableGqa = params.at("enable_gqa");

    float scale = 0.0f;
    if (params.count("scale")) {
      scale = params.at("scale");
    } else {
      auto queryShape = node->inputs()[0]->shape();
      if (queryShape.size() >= 4 && queryShape[3] > 0)
        scale = 1.0f / std::sqrt(static_cast<float>(queryShape[3]));
    }

    auto isCausalAttr = builder.getBoolAttr(isCausal);
    auto scaleAttr = builder.getF32FloatAttr(scale);
    auto enableGqaAttr = builder.getBoolAttr(enableGqa);
    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    auto output = builder.create<ScaledDotProductAttentionOp>(
        location, outType, query, key, value, attnMask, isCausalAttr, scaleAttr,
        enableGqaAttr, opAttrs);

    return {output};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_ScaledDotProductAttention,
                    ScaledDotProductAttention);

} // namespace aipugc
} // namespace mlir
