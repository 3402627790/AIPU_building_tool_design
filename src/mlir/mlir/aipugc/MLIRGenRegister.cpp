//===- Eltwise.cpp - MLIR Generation function 'mlirGen' for Eltwise OP.
//-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements MLIR generation function ‘mlirGen’ for Eltwise.
//
//===----------------------------------------------------------------------===//
#include "aipugc/MLIRGenRegister.h"
#include "aipugc/AipuParamKeys.h"
#include "aipugc/utils.h"
namespace mlir {
namespace aipugc {

Attribute
MLIRGenBase::ParamsToAttribute(mlir::OpBuilder &builder,
                               const aipubt::NodeParamValue &paramValue) {
  switch ((paramValue.t)) {
  case aipubt::NodeParamValue::NodeParamType_bool:
    return builder.getBoolAttr(bool(paramValue));
  case aipubt::NodeParamValue::NodeParamType_dtype:
    return builder.getStringAttr(
        aipubt::Tensor::dtype_to_string(aipubt::Tensor::Dtype(paramValue)));
  case aipubt::NodeParamValue::NodeParamType_int:
    return builder.getI32IntegerAttr(int(paramValue));
  case aipubt::NodeParamValue::NodeParamType_uint:
    return builder.getUI32IntegerAttr(uint32_t(paramValue));
  case aipubt::NodeParamValue::NodeParamType_float:
    return builder.getF32FloatAttr(float(paramValue));
  case aipubt::NodeParamValue::NodeParamType_string:
    return builder.getStringAttr(std::string(paramValue));
  case aipubt::NodeParamValue::NodeParamType_shape: {
    aipubt::TensorShape value = paramValue;
    return builder.getStringAttr(value.to_string());
  }
  case aipubt::NodeParamValue::NodeParamType_arr_bool: {
    SmallVector<bool> values;
    for (auto p : paramValue.arr) {
      values.push_back(bool(p));
    }
    return builder.getBoolArrayAttr(values);
  }
  case aipubt::NodeParamValue::NodeParamType_arr_dtype: {
    SmallVector<StringRef> values;
    for (auto p : paramValue.arr) {
      values.push_back(
          aipubt::Tensor::dtype_to_string(aipubt::Tensor::Dtype(paramValue)));
    }
    return builder.getStrArrayAttr(values);
  }
  case aipubt::NodeParamValue::NodeParamType_arr_int: {
    SmallVector<int32_t> values;
    for (auto p : paramValue.arr) {
      values.push_back(int32_t(p));
    }
    return builder.getI32ArrayAttr(values);
  }
  case aipubt::NodeParamValue::NodeParamType_arr_uint: {
    SmallVector<int32_t> values;
    for (auto p : paramValue.arr) {
      values.push_back(int32_t(p));
    }
    return builder.getI32ArrayAttr(values);
  }
  case aipubt::NodeParamValue::NodeParamType_arr_float: {
    SmallVector<float> values;
    for (auto p : paramValue.arr) {
      values.push_back(float(p));
    }
    return builder.getF32ArrayAttr(values);
  }
  case aipubt::NodeParamValue::NodeParamType_arr_string: {
    SmallVector<StringRef> values;
    for (auto p : paramValue.arr) {
      values.push_back(std::string(p));
    }
    return builder.getStrArrayAttr(values);
  }
  case aipubt::NodeParamValue::NodeParamType_arr_shape: {
    SmallVector<StringRef> values;
    for (auto p : paramValue.arr) {
      aipubt::TensorShape sp = paramValue;
      values.push_back(sp.to_string());
    }
    return builder.getStrArrayAttr(values);
  }
  case aipubt::NodeParamValue::NodeParamType_arr: {
    SmallVector<Attribute> values;
    for (auto p : paramValue.arr) {
      values.push_back(ParamsToAttribute(builder, p));
    }
    return builder.getArrayAttr(values);
    break;
  }
  default:
    break;
  }
  return {};
}

DictionaryAttr MLIRGenBase::ParamsToMlirAttr(mlir::OpBuilder &builder,
                                             aipubt::NodePtr node,
                                             std::set<std::string> uniqueKeys) {
  DictionaryAttr attrs;
  SmallVector<NamedAttribute> named_attrs;
  // parameter
  for (auto &param : node->params()) {
    if ((!paramsKeys.count(param.first)) && (!uniqueKeys.count(param.first)))
      continue;

    auto attr = ParamsToAttribute(builder, param.second);
    if (attr) {
      named_attrs.push_back(builder.getNamedAttr(param.first, attr));
    }
  }
  // quantization tensor to DenseElementsAttr
  for (auto cst : node->constants()) {
    if (!quantConstantKeys.count(cst.first)) {
      continue;
    }
    auto tensor = cst.second;
    auto dataType = utils::getType(builder, tensor);
    auto dataAttribute = utils::DenseElementsAttrGet(builder, tensor);
    named_attrs.push_back(builder.getNamedAttr(cst.first, dataAttribute));
  }

  return builder.getDictionaryAttr(named_attrs);
}

mlir::Value MLIRGenBase::ConstantTensorMlirGen(
    mlir::OpBuilder &builder, mlir::Location loc, aipubt::TensorPtr tensor,
    std::map<std::string, mlir::Value> &symbolTable) {
  auto dataType = utils::getType(builder, tensor);
  auto dataAttribute = utils::DenseElementsAttrGet(builder, tensor);

  auto output = builder.create<ConstantOp>(loc, dataType, dataAttribute);
  if (!symbolTable.count(tensor->name())) {
    symbolTable.insert(std::make_pair(tensor->name(), output));
  }
  return output;
}

} // namespace aipugc
} // namespace mlir