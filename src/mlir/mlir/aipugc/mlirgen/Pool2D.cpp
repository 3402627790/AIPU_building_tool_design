//===- Pool2D.cpp - MLIR Generation function 'mlirGen' for Input OP.
//-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements MLIR generation function 'mlirGen' for Pool2D.
//
//===----------------------------------------------------------------------===//

#include "aipugc/MLIRGenRegister.h"
#include <numeric>

namespace mlir {
namespace aipugc {

class MLIRGenPool2D : public MLIRGenBase {
public:
  MLIRGenPool2D() {}
  ~MLIRGenPool2D() {}
  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    // Retrieve input and output tensors
    auto ts_input = node->inputs()[0];
    auto ts_output = node->outputs()[0];
    auto outType = utils::getType(builder, ts_output);
    auto location = loc(builder, node);

    // Retrieve MLIR Value for the input tensor
    mlir::Value mlir_input = getsymbol(symbolTable, ts_input->name());
    if (!mlir_input) {
      llvm::errs() << "The input " << ts_input->name() << " is nullptr \n";
      return {};
    }

    // Extract pooling parameters
    std::vector<int32_t> bt_kernel = {
        node->params()["kernel_y"],
        node->params()["kernel_x"],
    };
    std::vector<int32_t> bt_strides = {
        node->params()["stride_y"],
        node->params()["stride_x"],
    };
    std::vector<int32_t> bt_padding = {
        node->params()["pad_top"],
        node->params()["pad_bottom"],
        node->params()["pad_left"],
        node->params()["pad_right"],
    };
    std::vector<int32_t> bt_dilation = {
        node->params()["dilation_y"],
        node->params()["dilation_x"],
    };
    bool ceil_mode = node->params()["ceil_mode"];
    bool count_include_pad = node->params()["count_include_pad"];
    std::string method = node->params()["method"];

    // Convert to MLIR attributes
    DenseI64ArrayAttr kernel = utils::DenseI64ArrayAttrGet(builder, bt_kernel);
    DenseI64ArrayAttr strides =
        utils::DenseI64ArrayAttrGet(builder, bt_strides);
    DenseI64ArrayAttr padding =
        utils::DenseI64ArrayAttrGet(builder, bt_padding);
    DenseI64ArrayAttr dilation =
        utils::DenseI64ArrayAttrGet(builder, bt_dilation);
    BoolAttr ceil_mode_attr = builder.getBoolAttr(ceil_mode);
    BoolAttr count_include_pad_attr = builder.getBoolAttr(count_include_pad);
    auto method_attr = builder.getStringAttr(method);
    DictionaryAttr op_attrs = ParamsToMlirAttr(builder, node);

    // Create Pool2DOp
    auto output = builder.create<Pool2DOp>(
        location,
        outType,
        mlir_input,
        kernel,
        strides,
        dilation,
        padding,
        ceil_mode_attr,
        count_include_pad_attr,
        method_attr,
        op_attrs);

    return {output};
  }
};

// Register MLIR generator for Pooling2D
REGISTER_OP_MLIRGEN(aipubt::OpType_Pooling, Pool2D);

} // namespace aipugc
} // namespace mlir