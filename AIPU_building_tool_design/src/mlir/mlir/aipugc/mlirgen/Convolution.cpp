//===- Conv2D.cpp - MLIR Generation function 'mlirGen' for Input OP.
//-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements MLIR generation function 'mlirGen' for Conv2D.
//
//===----------------------------------------------------------------------===//

#include "aipugc/MLIRGenRegister.h"
#include <numeric>

namespace mlir {
namespace aipugc {
template <typename MLIROPType> class MLIRGenConvolution : public MLIRGenBase {
public:
  MLIRGenConvolution() {}
  ~MLIRGenConvolution() {}
  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    auto ts_input = node->inputs()[0];
    auto ts_weights = node->constants()["weights"];
    auto ts_biases = node->constants()["biases"];
    auto ts_output = node->outputs()[0];
    auto outType = utils::getType(builder, ts_output);
    auto location = loc(builder, node);
    mlir::Value mlir_input = getsymbol(symbolTable, ts_input->name());
    if (!mlir_input) {
      llvm::errs() << "The input " << ts_input->name() << "is nullptr \n";
      return {};
    }
    mlir::Value mlir_weights = getsymbol(symbolTable, ts_weights->name());
    if (!mlir_weights) {
      mlir_weights =
          ConstantTensorMlirGen(builder, location, ts_weights, symbolTable);
    }
    mlir::Value mlir_biases = getsymbol(symbolTable, ts_biases->name());
    if (!mlir_biases) {
      mlir_biases =
          ConstantTensorMlirGen(builder, location, ts_biases, symbolTable);
    }
    std::vector<int32_t> bt_kernel = {
        node->params()["kernel_y"],
        node->params()["kernel_x"],
    };
    std::vector<int32_t> bt_padding = {
        node->params()["pad_top"],
        node->params()["pad_bottom"],
        node->params()["pad_left"],
        node->params()["pad_right"],
    };
    std::vector<int32_t> bt_strides = {
        node->params()["stride_y"],
        node->params()["stride_x"],
    };
    std::vector<int32_t> bt_dilation = {
        node->params()["dilation_y"],
        node->params()["dilation_x"],
    };

    DenseI64ArrayAttr kernel = utils::DenseI64ArrayAttrGet(builder, bt_kernel);
    DenseI64ArrayAttr pad = utils::DenseI64ArrayAttrGet(builder, bt_padding);
    DenseI64ArrayAttr strides =
        utils::DenseI64ArrayAttrGet(builder, bt_strides);
    DenseI64ArrayAttr dilation =
        utils::DenseI64ArrayAttrGet(builder, bt_dilation);
    DictionaryAttr op_attrs = ParamsToMlirAttr(builder, node);
    auto output = builder.create<MLIROPType>(location, outType, mlir_input,
                                             mlir_weights, mlir_biases,
                                             kernel, pad, strides, dilation,
                                             op_attrs);
    return {output};
  }
};
class MLIRGenConv2D : public MLIRGenConvolution<Conv2DOp> {
public:
  using MLIRGenConvolution::MLIRGenConvolution;
  ~MLIRGenConv2D() {}
};

class MLIRGenDepthwiseConv : public MLIRGenConvolution<DepthwiseConvOp> {
public:
  using MLIRGenConvolution::MLIRGenConvolution;
  ~MLIRGenDepthwiseConv() {}
};

// class MLIRGenTransposeConv2D : public MLIRGenConvolution<TransposeConv2DOp> {
//  public:
//    using MLIRGenConvolution::MLIRGenConvolution;
//    ~MLIRGenTransposeConv2D() {}
//  };

REGISTER_OP_MLIRGEN(aipubt::OpType_Convolution, Conv2D);

REGISTER_MLIRGEN(DepthwiseConv);

// REGISTER_OP_MLIRGEN(aipubt::OpType_ConvTranspose, TransposeConv2D);
} // namespace aipugc
} // namespace mlir