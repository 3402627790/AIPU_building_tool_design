//===- Eltwise.cpp - MLIR Generation function 'mlirGen' for Eltwise OP.
//-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements MLIR generation function 'mlirGen' for Eltwise.
//
//===----------------------------------------------------------------------===//
#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

template <typename MLIROPType> class MLIRGenEltwiseBase : public MLIRGenBase {
public:
  MLIRGenEltwiseBase() {}
  ~MLIRGenEltwiseBase() {}
  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    mlir::Value lhs = getsymbol(symbolTable, node->inputs()[0]->name());
    if (!lhs) {
      llvm::errs() << "The lhs is nullptr \n ";
      return {};
    }
    mlir::Value rhs = getsymbol(symbolTable, node->inputs()[1]->name());
    if (!rhs) {
      llvm::errs() << "The rhs is nullptr \n ";
      return {};
    }

    auto outType = utils::getType(builder, node->outputs()[0]);
    auto location = loc(builder, node);

    const auto& params = node->params();
    std::string method = params.at("method");
    auto methodAttr = builder.getStringAttr(method);
    std::string with_activation = params.at("with_activation");
    auto with_activation_attr = builder.getStringAttr(with_activation);
  
    DictionaryAttr op_attrs = ParamsToMlirAttr(builder, node);
    auto output = builder.create<MLIROPType>(
        location,
        outType,
        lhs, rhs,
        method,
        with_activation_attr,
        op_attrs
    );
    return {output};
  }
};

class MLIRGenEltwiseAdd : public MLIRGenEltwiseBase<EltwiseOp> {
public:
  using MLIRGenEltwiseBase::MLIRGenEltwiseBase;
  ~MLIRGenEltwiseAdd() {}
};
class MLIRGenEltwiseSub : public MLIRGenEltwiseBase<EltwiseOp> {
public:
  using MLIRGenEltwiseBase::MLIRGenEltwiseBase;
  ~MLIRGenEltwiseSub() {}
};
class MLIRGenEltwiseMul : public MLIRGenEltwiseBase<EltwiseOp> {
public:
  using MLIRGenEltwiseBase::MLIRGenEltwiseBase;
  ~MLIRGenEltwiseMul() {}
};

REGISTER_OP_MLIRGEN(aipubt::OpType_Eltwise, EltwiseAdd);
REGISTER_OP_MLIRGEN(aipubt::OpType_Eltwise, EltwiseSub);
REGISTER_OP_MLIRGEN(aipubt::OpType_Eltwise, EltwiseMul);

} // namespace aipugc
} // namespace mlir