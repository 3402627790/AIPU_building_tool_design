//===- Concat.cpp - MLIR Generation function 'mlirGen' for Concat OP.
//-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements MLIR generation function 'mlirGen' for Concat.
//
//===----------------------------------------------------------------------===//
#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenConcat : public MLIRGenBase {
public:
  MLIRGenConcat() {}
  ~MLIRGenConcat() {}
  
  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    // 1. Process multiple input tensors
    llvm::SmallVector<mlir::Value> inputValues;
    for (auto& input_node : node->inputs()) {
      mlir::Value val = getsymbol(symbolTable, input_node->name());
      if (!val) {
        llvm::errs() << "Missing input: " << input_node->name() << "\n";
        return {};
      }
      inputValues.push_back(val);
    }

    // 2. Prepare output type and location info
    auto outType = utils::getType(builder, node->outputs()[0]);
    auto location = loc(builder, node);

    // 3. Extract axis parameter
    int32_t axis = node->params()["axis"];
    auto axis_attr = builder.getIntegerAttr(builder.getIntegerType(32), axis);

    // 4. Handle additional attributes
    DictionaryAttr op_attrs = ParamsToMlirAttr(builder, node);

    // 5. Create Concat operation
    auto output = builder.create<ConcatOp>(
        location, 
        outType,
        inputValues,    // Input tensor list (ValueRange)
        axis_attr,
        op_attrs
    );

    return {output};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_Concat, Concat);

} // namespace aipugc
} // namespace mlir