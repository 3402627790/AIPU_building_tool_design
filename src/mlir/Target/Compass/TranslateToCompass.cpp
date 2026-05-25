//===- TranslateToCompass.cpp - Translating to C++ calls
//----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Target/Compass/CompassEmitter.h"
#include "aipugc/AipuParamKeys.h"
#include "aipugc/QuantUtils.h"
#include "aipugc/utils.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/FormatVariadic.h"

#include "llvm/ADT/SmallPtrSet.h"

#include <cstddef>
#include <stack>
#include <utility>

#define DEBUG_TYPE "translate-to-Compass"

using namespace mlir;
using namespace mlir::compass;
using llvm::formatv;
namespace {

// Only when the name defined in the td file is inconsistent with the name of
// the aipu OpType, it is necessary to define it in the following file.
// Most OPs should not need to be defined.
std::map<std::string, aipubt::OpType> &get_dialect_optype_dict() {
  static std::map<std::string, aipubt::OpType> DialectDict = {
      {"conv2d", aipubt::OpType_Convolution},
      {"depthwise_conv2d", aipubt::OpType_DepthwiseConv},
      {"transpose_conv2d", aipubt::OpType_ConvTranspose},
      {"pool2d", aipubt::OpType_Pooling},
  };
  return DialectDict;
}

class AipuGraphEmitter {
public:
  explicit AipuGraphEmitter(aipubt::GraphPtr aipu_graph);

  LogicalResult emitFunction(mlir::aipugc::FuncOp func);
  LogicalResult emitOperation(Operation &op);
  // RAII helper function to manage entering/exiting C++ scopes.
  struct Scope {
    Scope(AipuGraphEmitter &emitter)
        : valueMapperScope(emitter.valueMapper), emitter(emitter) {
      emitter.valueInScopeCount.push(emitter.valueInScopeCount.top());
      emitter.labelInScopeCount.push(emitter.labelInScopeCount.top());
    }
    ~Scope() {
      emitter.valueInScopeCount.pop();
      emitter.labelInScopeCount.pop();
    }

  private:
    llvm::ScopedHashTableScope<Value, aipubt::TensorPtr> valueMapperScope;
    AipuGraphEmitter &emitter;
  };

private:
  aipubt::TensorShape emitShape(TensorType type);
  aipubt::Quantization emitQuantization(Type type);
  aipubt::TensorPtr emitTensor(Value value, std::string name = {});
  aipubt::TensorPtr emitConstTensor(std::string name, Attribute &attr);
  /// Emits operation 'op' with/without training semicolon or returns failure.
  LogicalResult emitInput(Value value);

  // emit function:1 operand , 1 output
  template <typename OpTy>
  LogicalResult emitUnaryOp(OpTy &op, llvm::ArrayRef<NamedAttribute> attrs);

  // emit function:greater than or equal to 2 operands
  template <typename OpTy>
  LogicalResult emitMultiOperandOp(OpTy &op,
                                   llvm::ArrayRef<NamedAttribute> attrs);

  // emit function:greater than or equal to 2 outputs
  template <typename OpTy>
  LogicalResult emitMultiOutputOp(OpTy &op,
                                  llvm::ArrayRef<NamedAttribute> attrs);

  // emit function:greater than or equal to 2 operands
  template <typename OpTy>
  LogicalResult emitMultiOutputsOp(OpTy &op,
                                   llvm::ArrayRef<NamedAttribute> attrs);

  // emit 'aipugc.constant' to aipu node 'OpType_Constant'
  LogicalResult emitConstant(aipugc::ConstantOp &op);

  template <typename OpTy>
  LogicalResult emitConv(OpTy &op, llvm::ArrayRef<NamedAttribute> attrs);

  // emit return value to graph output tensors
  LogicalResult emitReturn(aipugc::ReturnOp &op);

  /// A utility method that returns if the given operation has all of its
  /// operands declared.
  bool allOperandsDeclared(Operation *op) {
    for (auto operand : op->getOperands()) {
      if (!OperandsDeclared(operand))
        return false;
    }
    return true;
  }

  bool OperandsDeclared(Value operand) { return (valueMapper.count(operand)); }

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  void declare(Value value, aipubt::TensorPtr tensor) {
    if (!valueMapper.count(value))
      valueMapper.insert(value, tensor);
  }

  aipubt::TensorPtr getTensor(Value val) {
    if (valueMapper.count(val))
      return *valueMapper.begin(val);
    else
      return nullptr;
  }

  aipubt::OpType GetOpType(StringRef OpName);
  aipubt::NodeParamValue ParamsToAttribute(Attribute &attr);
  void MlirAttrToParams(llvm::ArrayRef<NamedAttribute> attrs,
                        aipubt::NodePtr node);

  using ValueMapper = llvm::ScopedHashTable<Value, aipubt::TensorPtr>;

  /// Output stream to emit to.
  aipubt::GraphPtr aipuGraph;

  /// Map from name to aipu tensor that contain the name.
  ValueMapper valueMapper;

  /// Map from block to name of C++ label.
  // BlockMapper blockMapper;

  /// The number of values in the current scope. This is used to declare the
  /// names of values in a scope.
  std::stack<int64_t> valueInScopeCount;
  std::stack<int64_t> labelInScopeCount;
};
} // namespace

AipuGraphEmitter::AipuGraphEmitter(aipubt::GraphPtr aipuGraph)
    : aipuGraph(aipuGraph) {
  valueInScopeCount.push(0);
  labelInScopeCount.push(0);
}

aipubt::OpType AipuGraphEmitter::GetOpType(StringRef OpName) {
  std::string name = OpName.str().substr(7); // remove "aipugc."
  auto type = aipubt::string_upper_to_optype(name);
  if (type == 0) {
    type = get_dialect_optype_dict()[name];
  }
  return type;
}

aipubt::NodeParamValue AipuGraphEmitter::ParamsToAttribute(Attribute &attr) {
  if (llvm::dyn_cast<BoolAttr>(attr)) {
    return aipubt::NodeParamValue(llvm::dyn_cast<BoolAttr>(attr).getValue());
  } else if (llvm::dyn_cast<StringAttr>(attr)) {
    std::string str = llvm::dyn_cast<StringAttr>(attr).str();
    auto dtypeStr = aipubt::get_dtype_dict();
    if (dtypeStr.count(str)) {
      return aipubt::NodeParamValue(dtypeStr[str]);
    } // TODO: check shape
    else {
      return aipubt::NodeParamValue(str);
    }
  } else if (llvm::dyn_cast<IntegerAttr>(attr)) {
    auto intAttr = llvm::dyn_cast<IntegerAttr>(attr);
    auto type = intAttr.getType();
    if (type.isInteger() || type.isSignedInteger()) {
      int32_t value = intAttr.getInt();
      return aipubt::NodeParamValue(value);
    } else if (type.isSignlessInteger() || type.isUnsignedInteger()) {
      uint32_t value = intAttr.getUInt();
      return aipubt::NodeParamValue(value);
    }
  } else if (llvm::dyn_cast<FloatAttr>(attr)) {
    auto floatAttr = llvm::dyn_cast<FloatAttr>(attr);
    auto type = floatAttr.getType();
    if (type.isF32()) {
      float value = floatAttr.getValue().convertToFloat();
      return aipubt::NodeParamValue(value);
    }
  } else if (llvm::dyn_cast<DenseI64ArrayAttr>(attr)) {
    auto arr = llvm::dyn_cast<DenseI64ArrayAttr>(attr);
    auto size = arr.getSize();
    std::vector<int32_t> values(size);
    for (auto i = 0; i < size; ++i) {
      int64_t value = arr[i];
      values[i] = value;
    }
    return aipubt::NodeParamValue(values);
  } else if (llvm::dyn_cast<ArrayAttr>(attr)) {
    auto arr = llvm::dyn_cast<ArrayAttr>(attr);
    auto arrSize = arr.size();
    if (arrSize <= 0) {
      return aipubt::NodeParamValue();
    }
    if (llvm::dyn_cast<BoolAttr>(arr[0])) {
      std::vector<bool> values;
      for (auto a : arr) {
        values.push_back(llvm::dyn_cast<BoolAttr>(a).getValue());
      }
    } else if (llvm::dyn_cast<StringAttr>(arr[0])) {
      auto dtypeStr = aipubt::get_dtype_dict();
      if (dtypeStr.count(llvm::dyn_cast<StringAttr>(arr[0]).str())) {
        std::vector<aipubt::Tensor::Dtype> values;
        for (auto a : arr) {
          values.push_back(dtypeStr[llvm::dyn_cast<StringAttr>(a).str()]);
        }
        return aipubt::NodeParamValue(values);
      } // TODO: check shape
      else {
        std::vector<std::string> values;
        for (auto a : arr) {
          values.push_back(llvm::dyn_cast<StringAttr>(a).str());
        }
        return aipubt::NodeParamValue(values);
      }
    } else if (llvm::dyn_cast<IntegerAttr>(arr[0])) {
      auto intAttr = llvm::dyn_cast<IntegerAttr>(arr[0]);
      auto type = intAttr.getType();
      if (type.isInteger() || type.isSignedInteger()) {
        std::vector<int32_t> values;
        for (auto a : arr) {
          int32_t value = llvm::dyn_cast<IntegerAttr>(a).getInt();
          values.push_back(value);
        }
        return aipubt::NodeParamValue(values);
      } else if (type.isSignlessInteger() || type.isUnsignedInteger()) {
        std::vector<uint32_t> values;
        for (auto a : arr) {
          uint32_t value = llvm::dyn_cast<IntegerAttr>(a).getUInt();
          values.push_back(value);
        }
        return aipubt::NodeParamValue(values);
      }
    } else if (llvm::dyn_cast<FloatAttr>(arr[0])) {
      auto floatAttr = llvm::dyn_cast<FloatAttr>(arr[0]);
      auto type = floatAttr.getType();
      if (type.isF32()) {
        std::vector<float> values;
        for (auto a : arr) {
          float value =
              llvm::dyn_cast<FloatAttr>(a).getValue().convertToFloat();
          return aipubt::NodeParamValue(values);
        }
      } else {
        std::vector<aipubt::NodeParamValue> values;
        for (auto a : arr) {
          values.push_back(ParamsToAttribute(a));
        }
        return aipubt::NodeParamValue(values);
      }
    }
  }
  return aipubt::NodeParamValue();
}

aipubt::TensorPtr AipuGraphEmitter::emitConstTensor(std::string name,
                                                    Attribute &attr) {
  auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(attr);
  auto shapeType = denseAttr.getType();
  auto aipu_dtype = aipugc::utils::GetDtype(shapeType.getElementType());
  int32_t size = denseAttr.size();
  aipubt::TensorShape shape({size});
  auto tensor = std::make_shared<aipubt::Tensor>(
      aipuGraph->get_valid_tensor_name(name), shape);
  tensor->set_type(aipu_dtype);
  const char *rawData = denseAttr.getRawData().data();
  uint8_t *data = new uint8_t[tensor->mem_size()];
  memcpy(data, rawData, tensor->mem_size());
  tensor->set_data(data, tensor->mem_size());
  return tensor;
}

void AipuGraphEmitter::MlirAttrToParams(llvm::ArrayRef<NamedAttribute> attrs,
                                        aipubt::NodePtr node) {
  auto &params = node->params();
  std::set<std::string> dialectKeys = {
      "strides",
      "dilations",
      "padding",
      "kernel",
  };
  auto dialectParamsPush = [&params](std::string name,
                                     aipubt::NodeParamValue value) {
    if (name == "padding") {
      params["pad_top"] = value[0];
      params["pad_bottom"] = value[1];
      params["pad_left"] = value[2];
      params["pad_right"] = value[3];
    } else if (name == "dilations") {
      params["dilation_y"] = value[0];
      params["dilation_x"] = value[1];
    } else if (name == "strides") {
      params["stride_y"] = value[0];
      params["stride_x"] = value[1];
    } else if (name == "kernel") {
      params["kernel_y"] = value[0];
      params["kernel_x"] = value[1];
    }
  };
  auto processDictAttr = [&](const mlir::DictionaryAttr &dict) {
    for (auto innerNamedAttr : dict) {
      auto innerName = innerNamedAttr.getName().str();
      auto innerValue = innerNamedAttr.getValue();

      if (mlir::aipugc::quantConstantKeys.count(innerName)) {
        node->constants()[innerName] = emitConstTensor(innerName, innerValue);
      } else {
        auto param = ParamsToAttribute(innerValue);
        if (param.t == aipubt::NodeParamValue::NodeParamType_void) {
          llvm::dbgs() << "Skip void param: " << innerName << "\n";
          continue;
        }

        if (dialectKeys.count(innerName)) {
          dialectParamsPush(innerName, param);
        } else {
          params[innerName] = param;
        }
      }
    }
  };
  for (auto NamedAttr : attrs) {
    auto name = NamedAttr.getName().str();
    auto value = NamedAttr.getValue();

    if (name == "dictAttr") {
      if (auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(value)) {
        processDictAttr(dict);
      } else {
        llvm::dbgs() << "WARNING: Invalid dictAttr type\n";
      }
      continue;
    }

    if (mlir::aipugc::quantConstantKeys.count(name)) {
      node->constants()[name] = emitConstTensor(name, value);
    } else {
      auto param = ParamsToAttribute(value);
      if (param.t == aipubt::NodeParamValue::NodeParamType_void) {
        llvm::dbgs() << "Void NodeParam name: " << name << ", value: " << value
                     << "\n";
        continue;
      }
      if (dialectKeys.count(name)) {
        dialectParamsPush(name, param);
      } else {
        params[name] = param;
      }
    }
  }
}

aipubt::Quantization AipuGraphEmitter::emitQuantization(Type type) {
  auto tensorType = llvm::dyn_cast<TensorType>(type);
  auto quant = mlir::aipugc::GetQuantization(tensorType.getElementType());
  auto aipu_dtype = quant.aipu_dtype();
  if (aipu_dtype == aipubt::Tensor::Dtype::VOID) {
    aipu_dtype = aipugc::utils::GetDtype(tensorType.getElementType());
    quant.set_aipu_dtype(aipu_dtype);
  }
  return quant;
}
aipubt::TensorShape AipuGraphEmitter::emitShape(TensorType type) {
  auto shape = type.getShape();
  std::vector<int> dims;
  for (auto s : shape)
    dims.push_back(s);
  aipubt::TensorShape new_shape(dims);
  return new_shape;
}
aipubt::TensorPtr AipuGraphEmitter::emitTensor(Value value, std::string name) {
  auto output_type = value.getType();
  auto quant = emitQuantization(output_type);
  auto tensorType = llvm::dyn_cast<TensorType>(output_type);
  auto outputShape = emitShape(tensorType);
  auto outputTensor = std::make_shared<aipubt::Tensor>(
      aipuGraph->get_valid_tensor_name(name), outputShape);
  outputTensor->quantization() = quant;
  outputTensor->set_type(quant.aipu_dtype());
  return outputTensor;
}

// emit function:greater than or equal to 2 operands
template <typename OpTy>
LogicalResult
AipuGraphEmitter::emitUnaryOp(OpTy &op, llvm::ArrayRef<NamedAttribute> attrs) {
  llvm::dbgs() << op.getOperationName() << " \n";
  aipubt::OpType optype = GetOpType(op.getOperationName());
  if (optype == 0) {
    return op.emitError() << "'" << op.getOperationName()
                          << "' transpose to aipu OpType failed";
  }
  auto node = aipubt::Node::create(
      aipuGraph->get_valid_node_name(aipubt::optype_to_string(optype)), optype);
  auto input = getTensor(op.getOperand());
  node->add_input(input);
  auto output = op.getOutput();
  auto outputTensor =
      emitTensor(output, aipubt::optype_to_string(optype) + "/out");
  node->add_output(outputTensor);
  aipuGraph->add_node(node);
  MlirAttrToParams(attrs, node);
  declare(output, outputTensor);
  return success();
}

// emit function:greater than or equal to 2 operands
template <typename OpTy>
LogicalResult
AipuGraphEmitter::emitMultiOperandOp(OpTy &op,
                                     llvm::ArrayRef<NamedAttribute> attrs) {
  aipubt::OpType optype = GetOpType(op.getOperationName());
  if (optype == 0) {
    return op.emitError() << "'" << op.getOperationName()
                          << "' transpose to aipu OpType failed";
  }
  auto node = aipubt::Node::create(
      aipuGraph->get_valid_node_name(aipubt::optype_to_string(optype)), optype);
  for (auto operand : op.getOperands()) {
    auto input = getTensor(operand);
    node->add_input(input);
  }
  auto output = op.getOutput();
  auto outputTensor =
      emitTensor(output, aipubt::optype_to_string(optype) + "/out");
  node->add_output(outputTensor);
  aipuGraph->add_node(node);
  MlirAttrToParams(attrs, node);
  declare(output, outputTensor);
  return success();
}

// emit function:greater than or equal to 2 outputs
template <typename OpTy>
LogicalResult
AipuGraphEmitter::emitMultiOutputOp(OpTy &op,
                                    llvm::ArrayRef<NamedAttribute> attrs) {
  aipubt::OpType optype = GetOpType(op.getOperationName());
  if (optype == 0) {
    return op.emitError() << "'" << op.getOperationName()
                          << "' transpose to aipu OpType failed";
  }
  auto node = aipubt::Node::create(
      aipuGraph->get_valid_node_name(aipubt::optype_to_string(optype)), optype);
  auto input = getTensor(op.getOperand());
  node->add_input(input);
  for (auto output : op.getOutputs()) {
    auto outputTensor =
        emitTensor(output, aipubt::optype_to_string(optype) + "/out");
    node->add_output(outputTensor);
    declare(output, outputTensor);
  }
  aipuGraph->add_node(node);
  MlirAttrToParams(attrs, node);
  return success();
}

// emit constant
LogicalResult AipuGraphEmitter::emitConstant(aipugc::ConstantOp &op) {
  aipubt::OpType optype = GetOpType(op.getOperationName());
  if (optype == 0) {
    return op.emitError() << "'" << op.getOperationName()
                          << "' transpose to aipu OpType failed";
  }
  auto node = aipubt::Node::create(
      aipuGraph->get_valid_node_name(aipubt::optype_to_string(optype)), optype);
  auto output = op.getOutput();
  auto outputTensor =
      emitTensor(output, aipubt::optype_to_string(optype) + "/output");
  auto constTensor =
      emitTensor(output, aipubt::optype_to_string(optype) + "/const_tensor");

  auto dstElementsAttr = dyn_cast<DenseElementsAttr>(op.getValue());
  const char *rawData = dstElementsAttr.getRawData().data();
  uint8_t *data = new uint8_t[outputTensor->mem_size()];
  memcpy(data, rawData, outputTensor->mem_size());
  constTensor->set_data(data, outputTensor->mem_size());

  node->add_output(outputTensor);
  node->constants()["weights"] = constTensor;
  aipuGraph->add_node(node);
  declare(output, outputTensor);
  return success();
}

template <typename OpTy>
LogicalResult AipuGraphEmitter::emitConv(OpTy &op,
                                         llvm::ArrayRef<NamedAttribute> attrs) {
  aipubt::OpType optype = GetOpType(op.getOperationName());
  if (optype == 0) {
    return op.emitError() << "'" << op.getOperationName()
                          << "' transpose to aipu OpType failed";
  }
  auto node = aipubt::Node::create(
      aipuGraph->get_valid_node_name(aipubt::optype_to_string(optype)), optype);
  auto input = getTensor(op.getInput());
  node->add_input(input);

  auto weight = getTensor(op.getWeight());
  if (weight->node()->type() == aipubt::OpType::OpType_Constant) {
    auto weightTensor = weight->node()->constants()["weights"];
    node->constants()["weights"] = weightTensor;
  } else {
    node->add_input(weight);
  }

  auto bias = getTensor(op.getBias());
  if (bias->node()->type() == aipubt::OpType::OpType_Constant) {
    auto biasTensor = bias->node()->constants()["weights"];
    node->constants()["biases"] = biasTensor;
  } else {
    node->add_input(bias);
  }

  auto output = op.getOutput();
  auto outputTensor =
      emitTensor(output, aipubt::optype_to_string(optype) + "/out");
  node->add_output(outputTensor);
  aipuGraph->add_node(node);
  MlirAttrToParams(attrs, node);
  declare(output, outputTensor);
  return success();
}

LogicalResult AipuGraphEmitter::emitReturn(aipugc::ReturnOp &op) {
  aipubt::TensorList outputTensors;
  for (auto operand : op.getOperands()) {
    auto tensor = getTensor(operand);
    if (tensor) {
      outputTensors.push_back(getTensor(operand));
    }
  }
  aipuGraph->record_output_tensors(outputTensors);
  return success();
}

LogicalResult AipuGraphEmitter::emitOperation(Operation &op) {
  auto attrs = op.getAttrs();
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          // Builtin ops.
          .Case<ModuleOp>([&](auto op) {
            llvm::dbgs() << "ModuleOp skip \n";
            return success();
          })
          //   ops.
          .Case<aipugc::EltwiseOp, aipugc::AddOp, aipugc::SubOp, aipugc::MulOp,
                aipugc::ConcatOp, aipugc::ReverseSequenceOp,
                aipugc::BasicLSTMOp, aipugc::MatMulOp,
                aipugc::SquaredDifferenceOp, aipugc::GatherOp>(
              [&](auto op) { return emitMultiOperandOp(op, attrs); })
          .Case<aipugc::ConstantOp>([&](auto op) { return emitConstant(op); })
          .Case<aipugc::Conv2DOp, aipugc::DepthwiseConvOp,
                aipugc::FullyConnectedOp, aipugc::TransposeConv2DOp>(
              [&](auto op) { return emitConv(op, attrs); })
          .Case<aipugc::Pool2DOp, aipugc::SoftmaxOp, aipugc::ReshapeOp,
                aipugc::TransposeOp, aipugc::SpaceToDepthOp,
                aipugc::ActivationOp, aipugc::CropOp, aipugc::CastOp,
                aipugc::DepthToSpaceOp, aipugc::ReduceOp, aipugc::SliceOp,
                aipugc::PadOp, aipugc::ResizeOp, aipugc::RsqrtOp,
                aipugc::QuantizeOp>(
              [&](auto op) { return emitUnaryOp(op, attrs); })
          .Case<aipugc::SplitOp>(
              [&](auto op) { return emitMultiOutputOp(op, attrs); })
          .Case<aipugc::FuncOp>([&](auto op) {
            llvm::dbgs() << "FuncOp skip \n";
            return success();
          })
          .Case<aipugc::ReturnOp>([&](auto op) { return emitReturn(op); })

          .Default([&](Operation *) {
            return op.emitOpError("unable to find emit for op");
          });

  if (failed(status))
    return failure();

  return success();
}

LogicalResult AipuGraphEmitter::emitInput(Value value) {
  auto type = value.getType();
  auto new_shape = emitShape(llvm::dyn_cast<TensorType>(type));
  auto quant = emitQuantization(type);
  auto inputTensor =
      aipubt::ops::input(new_shape, quant.aipu_dtype(), quant, aipuGraph);
  declare(value, inputTensor);
  return success();
}

LogicalResult AipuGraphEmitter::emitFunction(mlir::aipugc::FuncOp func) {
  // Let's start the body of the function now!
  mlir::Block &entryBlock = func.front();
  for (auto value : func.getArguments()) {
    if (failed(emitInput(value))) {
      return failure();
    }
  }

  // Emit all of the operations in the function.
  auto f = func.getOperation();
  // Populate the worklist with theoperations.
  llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
  f->walk([&](mlir::Operation *op) { opWorklist.insert(op); });
  auto findIfOperationDeclared =
      [=](llvm::SmallPtrSet<mlir::Operation *, 16> &opWorklist) {
        for (auto it = opWorklist.begin(); it != opWorklist.end(); ++it) {
          if (allOperandsDeclared(*it))
            return it;
        }
        return opWorklist.end();
      };

  while (!opWorklist.empty()) {
    auto nextop = findIfOperationDeclared(opWorklist);
    if (nextop == opWorklist.end())
      break;

    Operation *op = *nextop;
    if (failed(emitOperation(*op))) {
      return failure();
    }
    opWorklist.erase(op);
  }

  // remove the overhanging constant node
  std::vector<aipubt::NodePtr> overhangNodes;
  for (auto node : aipuGraph->nodes()) {
    if (node->type() == aipubt::OpType::OpType_Constant &&
        node->outputs().size() > 0 &&
        node->outputs()[0]->consumers().size() == 0) {
      overhangNodes.push_back(node);
    }
  }
  for (auto node : overhangNodes) {
    aipuGraph->remove_node(node);
  }

  return success();
}

LogicalResult compass::translateToCompass(Operation *op,
                                          aipubt::GraphPtr aipuGraph) {
  AipuGraphEmitter emitter(aipuGraph);
  return emitter.emitOperation(*op);
}

aipubt::GraphPtr compass::translateToCompass(mlir::aipugc::FuncOp func) {
  std::string name = func.getSymName().str();
  auto aipuGraph =
      aipubt::Graph::create(name, aipubt::DataLayout::DataLayout_NHWC);
  AipuGraphEmitter emitter(aipuGraph);
  AipuGraphEmitter::Scope scope(emitter);

  if (failed(emitter.emitFunction(func)))
    return nullptr;

  return aipuGraph;
}

LogicalResult
compass::translateToCompass(mlir::OwningOpRef<mlir::ModuleOp> &module) {
  llvm::dbgs() << "translateToCompass\n";
  for (auto aipuFunc : (*module).getBody()->getOps<mlir::aipugc::FuncOp>()) {
    auto aipuGraph = translateToCompass(aipuFunc);
    aipuGraph->serialize("graph.def", "weight.bin");
  }
  return success();
}

// LogicalResult
// compass::translateToCompass(mlir::OwningOpRef<mlir::ModuleOp> &module,
//                             llvm::raw_ostream &os) {
//   llvm::dbgs() << "translateToCompass\n";
//   for (auto aipuFunc : (*module).getBody()->getOps<mlir::aipugc::FuncOp>())
//   {
//     llvm::dbgs() << "aipuFunc:" << aipuFunc->getName() << "\n";
//     auto aipuGraph = translateToCompass(aipuFunc);
//     aipuGraph->serialize("graph.def", "weight.bin");
//   }
//   return success();
// }