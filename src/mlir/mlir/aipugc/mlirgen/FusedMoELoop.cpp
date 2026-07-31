//===- FusedMoELoop.cpp - MLIRGen for FusedMoELoop OP --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

#include <cctype>

#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace aipugc {

class MLIRGenFusedMoELoop : public MLIRGenBase {
public:
  MLIRGenFusedMoELoop() = default;
  ~MLIRGenFusedMoELoop() = default;

  std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    if (node->inputs().size() < 2) {
      llvm::errs() << "FusedMoELoop expects at least 2 inputs, got "
                   << node->inputs().size() << ": " << node->name() << "\n";
      return {};
    }
    if (node->outputs().empty()) {
      llvm::errs() << "FusedMoELoop expects at least 1 output: "
                   << node->name() << "\n";
      return {};
    }

    auto ts_sorted_tokens = node->inputs()[0];
    auto ts_tokens_per_expert = node->inputs()[1];
    auto ts_output = node->outputs()[0];

    mlir::Value sortedTokens = getsymbol(symbolTable, ts_sorted_tokens->name());
    mlir::Value tokensPerExpert =
        getsymbol(symbolTable, ts_tokens_per_expert->name());
    if (!sortedTokens || !tokensPerExpert) {
      llvm::errs() << "FusedMoELoop missing input: " << node->name() << "\n";
      return {};
    }

    auto location = loc(builder, node);
    auto outType = utils::getType(builder, ts_output);

    std::vector<mlir::Value> weights;
    std::vector<mlir::Value> biases;

    auto collectConstants = [&](StringRef prefix, std::vector<mlir::Value> &dst) {
      for (auto &kv : node->constants()) {
        StringRef name(kv.first);
        bool match = false;
        if (name == prefix) {
          match = true;
        } else if (name.starts_with(prefix) &&
                   name.size() > prefix.size() + 1 &&
                   name[prefix.size()] == '_') {
          StringRef suffix = name.drop_front(prefix.size() + 1);
          match = llvm::all_of(suffix, [](char c) {
            return std::isdigit(static_cast<unsigned char>(c)) != 0;
          });
        }
        if (!match)
          continue;
        auto tensor = kv.second;
        mlir::Value value = getsymbol(symbolTable, tensor->name());
        if (!value) {
          value = ConstantTensorMlirGen(builder, location, tensor, symbolTable);
        }
        dst.push_back(value);
      }
    };

    collectConstants("weights", weights);
    collectConstants("biases", biases);

    if (weights.empty() && biases.empty() && node->inputs().size() > 2) {
      const size_t extraCount = node->inputs().size() - 2;
      const size_t half = extraCount / 2;
      for (size_t i = 2; i < node->inputs().size(); ++i) {
        auto ts = node->inputs()[i];
        mlir::Value value = getsymbol(symbolTable, ts->name());
        if (!value) {
          llvm::errs() << "FusedMoELoop missing extra input: " << ts->name()
                       << "\n";
          return {};
        }
        if (i - 2 < half)
          weights.push_back(value);
        else
          biases.push_back(value);
      }
    }

    const auto &params = node->params();
    std::string moeType = "unknown";
    if (params.count("moe_type"))
      moeType = params.at("moe_type");
    auto moeTypeAttr = builder.getStringAttr(moeType);

    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    llvm::SmallVector<mlir::Value> weightValues(weights.begin(), weights.end());
    llvm::SmallVector<mlir::Value> biasValues(biases.begin(), biases.end());
    auto output = builder.create<FusedMoELoopOp>(
        location, outType, sortedTokens, tokensPerExpert, weightValues,
        biasValues, moeTypeAttr, opAttrs);

    return {output};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_FusedMoELoop, FusedMoELoop);

} // namespace aipugc
} // namespace mlir
