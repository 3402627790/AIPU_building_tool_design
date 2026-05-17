//===- Dialect.cpp - AIPU IR Dialect registration in MLIR -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the dialect for the AIPU IR: custom type parsing and
// operation verification.
//
//===----------------------------------------------------------------------===//

#include "aipugc/Dialect.h"
#include "aipugc/QuantUtils.h"

// #include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>

// C-exported helper so Python (or any dlopen caller) can register the
// aipugc dialect into a live MLIRContext without needing C++ templates.
extern "C" __attribute__((visibility("default")))
void aipugcRegisterDialects(void *rawCtx) {
  static_cast<mlir::MLIRContext *>(rawCtx)
      ->getOrLoadDialect<mlir::aipugc::AIPUDialect>();
}

#define DEBUG_TYPE "dialect"

using namespace mlir;
using namespace mlir::aipugc;

namespace llvm {

// hash_code hash_combine(const int a, const float &arg) {
//   return hash_combine(a, APFloat(arg));
// }
} // namespace llvm
namespace mlir {
template <> struct FieldParser<APFloat> {
  static FailureOr<APFloat> parse(AsmParser &parser) {
    double a;
    if (parser.parseFloat(a))
      return failure();
    return (APFloat)((float)a);
  }
};
} // namespace mlir
#include "aipugc/AipuOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// AIPUInlinerInterface
//===----------------------------------------------------------------------===//

/// This class defines the interface for handling inlining with AIPU
/// operations.
struct AIPUInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// All call operations within aipugc can be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// All operations within aipugc can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  // All functions within aipugc can be inlined.
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator(aipugc.return) by replacing it with a
  /// new operation as necessary.
  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    // Only "aipugc.return" needs to be handled here.
    auto returnOp = cast<ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }

  /// Attempts to materialize a conversion for a type mismatch between a call
  /// from this dialect, and a callable region. This method should generate an
  /// operation that takes 'input' as the only operand, and produces a single
  /// result of 'resultType'. If a conversion can not be generated, nullptr
  /// should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    return builder.create<CastOp>(conversionLoc, resultType, input);
  }
};

//===----------------------------------------------------------------------===//
// AIPUDialect
//===----------------------------------------------------------------------===//

/// Print a type registered to this dialect.
void AIPUDialect::printType(mlir::Type type,
                            mlir::DialectAsmPrinter &printer) const {
  if (auto uniformType = llvm::dyn_cast<UniformQuantizedType>(type))
    printUniformQuantizedType(uniformType, printer);
  else if (auto perAxisType = llvm::dyn_cast<UniformQuantizedPerAxisType>(type))
    printUniformQuantizedPerAxisType(perAxisType, printer);
  else
    llvm_unreachable("Unhandled quantized type");
}

/// Parse a type registered to this dialect.
/// Formats:
///   !aipugc.quant<i8:f32, 1.000000>
///   !aipugc.quant<i8:f32, 1.000000:5>
///   !aipugc.quant_per_axis<i8:f32:1, {1.000000:0,2.000000:0}>
mlir::Type AIPUDialect::parseType(mlir::DialectAsmParser &parser) const {
  StringRef mnemonic;
  if (parser.parseKeyword(&mnemonic))
    return {};

  auto parseSingleStorageType = [&](bool &isSigned, unsigned &width)
      -> LogicalResult {
    StringRef storageStr;
    if (failed(parser.parseKeyword(&storageStr)))
      return failure();
    if (storageStr.starts_with("i")) {
      isSigned = true;
    } else if (storageStr.starts_with("u")) {
      isSigned = false;
    } else {
      parser.emitError(parser.getCurrentLocation(),
                       "expected 'i' or 'u' for storage type");
      return failure();
    }
    if (storageStr.drop_front(1).getAsInteger(10, width))
      return failure();
    return success();
  };

  if (mnemonic == "quant") {
    // !aipugc.quant<i8:f32, scale[:zeroPoint]>
    if (parser.parseLess())
      return {};

    bool isSigned = true;
    unsigned width = 8;
    if (failed(parseSingleStorageType(isSigned, width)))
      return {};

    if (parser.parseColon())
      return {};

    Type expressedType;
    if (parser.parseType(expressedType))
      return {};

    if (parser.parseComma())
      return {};

    double scale = 0.0;
    if (parser.parseFloat(scale))
      return {};

    int64_t zeroPoint = 0;
    if (succeeded(parser.parseOptionalColon())) {
      if (parser.parseInteger(zeroPoint))
        return {};
    }

    if (parser.parseGreater())
      return {};

    unsigned flags = isSigned ? QuantizationFlags::Signed : 0;
    Type storageType = IntegerType::get(getContext(), width);
    int64_t storageTypeMin =
        QuantizedType::getDefaultMinimumForInteger(isSigned, width);
    int64_t storageTypeMax =
        QuantizedType::getDefaultMaximumForInteger(isSigned, width);
    return UniformQuantizedType::get(flags, storageType, expressedType, scale,
                                    zeroPoint, storageTypeMin, storageTypeMax);
  }

  if (mnemonic == "quant_per_axis") {
    // !aipugc.quant_per_axis<i8:f32:dim, {s0:z0,s1:z1,...}>
    if (parser.parseLess())
      return {};

    bool isSigned = true;
    unsigned width = 8;
    if (failed(parseSingleStorageType(isSigned, width)))
      return {};

    if (parser.parseColon())
      return {};

    Type expressedType;
    if (parser.parseType(expressedType))
      return {};

    if (parser.parseColon())
      return {};

    int32_t quantizedDimension = 0;
    if (parser.parseInteger(quantizedDimension))
      return {};

    if (parser.parseComma())
      return {};

    SmallVector<double> scales;
    SmallVector<int64_t> zeroPoints;

    if (parser.parseLBrace())
      return {};

    auto parseOneParam = [&]() -> ParseResult {
      double s = 0.0;
      if (parser.parseFloat(s))
        return failure();
      int64_t z = 0;
      if (succeeded(parser.parseOptionalColon())) {
        if (parser.parseInteger(z))
          return failure();
      }
      scales.push_back(s);
      zeroPoints.push_back(z);
      return success();
    };

    if (parser.parseCommaSeparatedList(parseOneParam))
      return {};

    if (parser.parseRBrace() || parser.parseGreater())
      return {};

    unsigned flags = isSigned ? QuantizationFlags::Signed : 0;
    Type storageType = IntegerType::get(getContext(), width);
    int64_t storageTypeMin =
        QuantizedType::getDefaultMinimumForInteger(isSigned, width);
    int64_t storageTypeMax =
        QuantizedType::getDefaultMaximumForInteger(isSigned, width);
    return UniformQuantizedPerAxisType::get(flags, storageType, expressedType,
                                           scales, zeroPoints,
                                           quantizedDimension, storageTypeMin,
                                           storageTypeMax);
  }

  parser.emitError(parser.getCurrentLocation(),
                   "unknown aipugc type: " + mnemonic);
  return {};
}

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void AIPUDialect::initialize() {
  addTypes<UniformQuantizedType, UniformQuantizedPerAxisType>();
  addOperations<
#define GET_OP_LIST
#include "aipugc/AipuOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "aipugc/AipuAttributes.cpp.inc"
      >();
  addInterfaces<AIPUInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// AIPU Operations
//===----------------------------------------------------------------------===//

/// A generalized parser for binary operations. This parses the different forms
/// of 'printBinaryOp' below.
static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
  SMLoc operandsLoc = parser.getCurrentLocation();
  Type type;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return mlir::failure();

  // If the type is a function type, it contains the input and result types of
  // this operation.
  if (FunctionType funcType = llvm::dyn_cast<FunctionType>(type)) {
    if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                               result.operands))
      return mlir::failure();
    result.addTypes(funcType.getResults());
    return mlir::success();
  }

  // Otherwise, the parsed type is the type of both operands and results.
  if (parser.resolveOperands(operands, {type, type}, operandsLoc,
                             result.operands))
    return mlir::failure();
  result.addTypes(type);
  return mlir::success();
}

/// A generalized printer for binary operations. It prints in two different
/// forms depending on if all of the types match.
static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
  printer << " " << op->getOperands();
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  // If all of the types are the same, print the type directly.
  Type resultType = *op->result_type_begin();
  if (llvm::all_of(op->getOperandTypes(),
                   [=](Type type) { return type == resultType; })) {
    printer << resultType;
    return;
  }

  // Otherwise, print a functional type.
  printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

/// The 'OpAsmPrinter' class is a stream that allows for formatting
/// strings, attributes, operands, types, etc.
void InputOp::print(mlir::OpAsmPrinter &printer) {
  printer << " : ";
  mlir::Operation *op = *this;
  printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.
mlir::ParseResult InputOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  SmallVector<mlir::OpAsmParser::UnresolvedOperand, 1> operands;
  Type type;
  if (parser.resolveOperands(operands, type, result.operands))
    return mlir::failure();
  result.addTypes(type);

  return success();
}

/// Verifier for the input operation. This corresponds to the
/// `let hasVerifier = 1` in the op definition.
llvm::LogicalResult InputOp::verify() {
  // If the return type of the input is not an unranked tensor,.
  auto resultType =
      llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
  if (!resultType)
    return mlir::success();

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.
mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}

/// The 'OpAsmPrinter' class is a stream that allows for formatting
/// strings, attributes, operands, types, etc.
void ConstantOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << getValue();
}

/// Verifier for the constant operation. This corresponds to the
/// `let hasVerifier = 1` in the op definition.
llvm::LogicalResult ConstantOp::verify() {
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto resultType =
      llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
  if (!resultType)
    return success();

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto attrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
  if (attrType.getRank() != resultType.getRank()) {
    return emitOpError("return type must match the one of the attached value "
                       "attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  // Check that each of the dimensions match between the two types.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return emitOpError(
                 "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AbsOp
//===----------------------------------------------------------------------===//
void AbsOp::build(OpBuilder &builder, OperationState &state, Type outputType,
                  Value input, DictionaryAttr dictAttr) {
  state.addOperands(input);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}
LogicalResult AbsOp::verify() {
  auto inputType = getInput().getType();
  if (!mlir::isa<RankedTensorType>(inputType)) {
    return emitOpError("expected ranked tensor type for input");
  }
  return success();
}
void AbsOp::inferShapes() {
  auto inputType = mlir::cast<RankedTensorType>(getInput().getType());
  getResult().setType(inputType);
}

//===----------------------------------------------------------------------===//
// NegativeOp
//===----------------------------------------------------------------------===//
void NegativeOp::build(OpBuilder &builder, OperationState &state, Type outputType,
                  Value input, DictionaryAttr dictAttr) {
  state.addOperands(input);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}
LogicalResult NegativeOp::verify() {
  auto inputType = getInput().getType();
  if (!mlir::isa<RankedTensorType>(inputType)) {
    return emitOpError("expected ranked tensor type for input");
  }
  return success();
}
void NegativeOp::inferShapes() {
  auto inputType = mlir::cast<RankedTensorType>(getInput().getType());
  getResult().setType(inputType);
}

//===----------------------------------------------------------------------===//
// SquareOp
//===----------------------------------------------------------------------===//
void SquareOp::build(OpBuilder &builder, OperationState &state, Type outputType,
                  Value input, DictionaryAttr dictAttr) {
  state.addOperands(input);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}
LogicalResult SquareOp::verify() {
  auto inputType = getInput().getType();
  if (!mlir::isa<RankedTensorType>(inputType)) {
    return emitOpError("expected ranked tensor type for input");
  }
  return success();
}
void SquareOp::inferShapes() {
  auto inputType = mlir::cast<RankedTensorType>(getInput().getType());
  getResult().setType(inputType);
}

// Common verifier function for binary element-wise ops (Add/Sub/Mul/Div)
namespace {
  static LogicalResult verifyBinaryElementwiseOp(Operation* op, Value lhs, 
                                                  Value rhs, StringRef opName) {
    ShapedType lhsType = mlir::dyn_cast<ShapedType>(lhs.getType());
    ShapedType rhsType = mlir::dyn_cast<ShapedType>(rhs.getType());
    if (!lhsType || !rhsType) {
      return op->emitError() << opName << " operands must be shaped tensor type.";
    }
  
    auto getCompareType = [](Type t) {
      if (auto q = mlir::dyn_cast<QuantizedType>(t)) {
        return q.getStorageType();
      }
      return t;
    };
    auto lhsEleType = lhsType.getElementType();
    auto rhsEleType = rhsType.getElementType();
    auto lhsCmp = getCompareType(lhsEleType);
    auto rhsCmp = getCompareType(rhsEleType);
    bool compatible = false;
    if (auto lhsInt = mlir::dyn_cast<IntegerType>(lhsCmp)) {
      if (auto rhsInt = mlir::dyn_cast<IntegerType>(rhsCmp)) {
        compatible = lhsInt.getWidth() == rhsInt.getWidth();
      }
    } else {
      compatible = lhsCmp == rhsCmp;
    }
    if (!compatible) {
      llvm::errs() << "lhsElementType: " << lhsEleType << "\n";
      llvm::errs() << "rhsElementType: " << rhsEleType << "\n";
      return op->emitError() << opName << " operands must have compatible storage "
                             "element types (same bit-width for integers, or "
                             "identical floating-point types).";
    }
  
    SmallVector<int64_t> resultShape;
    if (!OpTrait::util::getBroadcastedShape(lhsType.getShape(),
                                            rhsType.getShape(), resultShape)) {
      return op->emitError()
             << opName << " operands have incompatible shapes for broadcasting.";
    }
    return success();
  }
  } // end of verifyBinaryElementwiseOp namespace

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//
void AddOp::build(OpBuilder &builder, OperationState &state, Type outputType,
                  Value lhs, Value rhs, DictionaryAttr dictAttr) {
  state.addOperands({lhs, rhs});
  state.attributes.append(dictAttr.getValue().begin(),
                          dictAttr.getValue().end());
  state.addTypes(outputType);
}

void AddOp::inferShapes() {
  ShapedType lhsType = mlir::dyn_cast<ShapedType>(getLhs().getType());
  ShapedType rhsType = mlir::dyn_cast<ShapedType>(getRhs().getType());
  SmallVector<int64_t> resultShape;
  if (OpTrait::util::getBroadcastedShape(lhsType.getShape(), rhsType.getShape(),
                                         resultShape)) {
    getResult().setType(
        RankedTensorType::get(resultShape, lhsType.getElementType()));
  } else {
    // Fallback, verify should catch errors
    getResult().setType(getLhs().getType());
  }
}
LogicalResult AddOp::verify() {
  return verifyBinaryElementwiseOp(getOperation(), getLhs(), getRhs(), "Add");
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//
void SubOp::build(OpBuilder &builder, OperationState &state, Type outputType,
                  Value lhs, Value rhs, DictionaryAttr dictAttr) {
  state.addOperands({lhs, rhs});
  state.attributes.append(dictAttr.getValue().begin(),
                          dictAttr.getValue().end());
  state.addTypes(outputType);
}

void SubOp::inferShapes() {
  ShapedType lhsType = mlir::dyn_cast<ShapedType>(getLhs().getType());
  ShapedType rhsType = mlir::dyn_cast<ShapedType>(getRhs().getType());
  SmallVector<int64_t> resultShape;
  if (OpTrait::util::getBroadcastedShape(lhsType.getShape(), rhsType.getShape(),
                                         resultShape)) {
    getResult().setType(
        RankedTensorType::get(resultShape, lhsType.getElementType()));
  } else {
    getResult().setType(getLhs().getType());
  }
}

LogicalResult SubOp::verify() {
  return verifyBinaryElementwiseOp(getOperation(), getLhs(), getRhs(), "Sub");
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//
void MulOp::build(OpBuilder &builder, OperationState &state, Type outputType,
                  Value lhs, Value rhs, DictionaryAttr dictAttr) {
  state.addOperands({lhs, rhs});
  state.attributes.append(dictAttr.getValue().begin(),
                          dictAttr.getValue().end());
  state.addTypes(outputType);
}

void MulOp::inferShapes() {
  ShapedType lhsType = mlir::dyn_cast<ShapedType>(getLhs().getType());
  ShapedType rhsType = mlir::dyn_cast<ShapedType>(getRhs().getType());
  SmallVector<int64_t> resultShape;
  if (OpTrait::util::getBroadcastedShape(lhsType.getShape(), rhsType.getShape(),
                                         resultShape)) {
    getResult().setType(
        RankedTensorType::get(resultShape, lhsType.getElementType()));
  } else {
    getResult().setType(getLhs().getType());
  }
}

LogicalResult MulOp::verify() {
  return verifyBinaryElementwiseOp(getOperation(), getLhs(), getRhs(), "Mul");
}
//===----------------------------------------------------------------------===//
// PowOp
//===----------------------------------------------------------------------===//
void PowOp::build(OpBuilder &builder, OperationState &state, Type outputType,
                  Value base, Value exponent, DictionaryAttr dictAttr) {
  state.addOperands({base, exponent});
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}

void PowOp::inferShapes() {
  ShapedType baseType = mlir::dyn_cast<ShapedType>(getBase().getType());
  ShapedType expType = mlir::dyn_cast<ShapedType>(getExponent().getType());
  if (!baseType || !expType)
    return;
  SmallVector<int64_t> resultShape;
  if (OpTrait::util::getBroadcastedShape(baseType.getShape(),
                                         expType.getShape(), resultShape)) {
    getResult().setType(
        RankedTensorType::get(resultShape, baseType.getElementType()));
  } else {
    getResult().setType(getBase().getType());
  }
}

LogicalResult PowOp::verify() {
  ShapedType baseType = mlir::dyn_cast<ShapedType>(getBase().getType());
  ShapedType expType = mlir::dyn_cast<ShapedType>(getExponent().getType());
  if (!baseType || !expType)
    return emitOpError("expected ranked tensor types for base and exponent");
  SmallVector<int64_t> resultShape;
  if (!OpTrait::util::getBroadcastedShape(baseType.getShape(),
                                         expType.getShape(), resultShape)) {
    return emitOpError(
        "base and exponent have incompatible shapes for broadcasting");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//
void DivOp::build(OpBuilder &builder, OperationState &state, Type outputType,
                  Value lhs, Value rhs, DictionaryAttr dictAttr) {
  state.addOperands({lhs, rhs});
  state.attributes.append(dictAttr.getValue().begin(),
                          dictAttr.getValue().end());
  state.addTypes(outputType);
}

void DivOp::inferShapes() {
  ShapedType lhsType = mlir::dyn_cast<ShapedType>(getLhs().getType());
  ShapedType rhsType = mlir::dyn_cast<ShapedType>(getRhs().getType());
  SmallVector<int64_t> resultShape;
  if (OpTrait::util::getBroadcastedShape(lhsType.getShape(), rhsType.getShape(),
                                         resultShape)) {
    getResult().setType(
        RankedTensorType::get(resultShape, lhsType.getElementType()));
  } else {
    getResult().setType(getLhs().getType());
  }
}

LogicalResult DivOp::verify() {
  return verifyBinaryElementwiseOp(getOperation(), getLhs(), getRhs(), "Div");
}

//===----------------------------------------------------------------------===//
// EltwiseOp
//===----------------------------------------------------------------------===//
void EltwiseOp::build(OpBuilder &builder, OperationState &state,
                      Type outputType, Value lhs, Value rhs, StringAttr method,
                      StringAttr with_activation, DictionaryAttr dictAttr) {
  state.addOperands({lhs, rhs});
  state.addAttribute("method", method);
  state.addAttribute("with_activation", with_activation);
  state.attributes.append(dictAttr.getValue().begin(),
                          dictAttr.getValue().end());
  state.addTypes(outputType);
}
LogicalResult EltwiseOp::verify() {
  ShapedType lhsType = mlir::dyn_cast<ShapedType>(getLhs().getType());
  ShapedType rhsType = mlir::dyn_cast<ShapedType>(getRhs().getType());
  // if (lhsType.getElementType() != rhsType.getElementType()) {
  //   return emitOpError("Eltwise op element types must be identical.");
  // }

  if (lhsType.getShape() != rhsType.getShape()) {
    return emitOpError(
               "Eltwise operand shapes must be exactly the same. Found [")
           << lhsType.getShape() << "] vs [" << rhsType.getShape() << "]";
  }
  return mlir::success();
}
void EltwiseOp::inferShapes() {
  auto lhsType = mlir::dyn_cast<TensorType>(getLhs().getType());
  getResult().setType(lhsType);
}

//===----------------------------------------------------------------------===//
// ActivationOp
//===----------------------------------------------------------------------===//

/// Infer the output shape of the ActivationOp, this is required by the shape
/// inference interface.
void ActivationOp::inferShapes() { getResult().setType(getInput().getType()); }

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//
void CastOp::build(OpBuilder &builder, OperationState &state, Type outputType,
                   Value input, mlir::StringAttr to_dtype,
                   DictionaryAttr dictAttr) {
  state.addOperands(input);

  state.addAttribute("to_dtype", to_dtype);
  state.attributes.append(dictAttr.getValue().begin(),
                          dictAttr.getValue().end());
  state.addTypes(outputType);
}

/// Infer the output shape of the CastOp, this is required by the shape
/// inference interface.
void CastOp::inferShapes() { getResult().setType(getInput().getType()); }

/// Returns true if the given set of input and result types are compatible with
/// this cast operation. This is required by the `CastOpInterface` to verify
/// this operation and provide other additional utilities.
bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  // The inputs must be Tensors with the same element type.
  TensorType input = llvm::dyn_cast<TensorType>(inputs.front());
  TensorType output = llvm::dyn_cast<TensorType>(outputs.front());
  if (!input || !output || input.getElementType() == output.getElementType())
    llvm::errs() << "input dtype and output dtype must be different\n";
  return !input.hasRank() || !output.hasRank() || input != output;
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  // FunctionOpInterface provides a convenient `build` method that will populate
  // the state of our FuncOp, and create an entry block.
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  // Dispatch to the FunctionOpInterface provided utility method that parses the
  // function operation.
  auto buildFuncType =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(mlir::OpAsmPrinter &p) {
  // Dispatch to the FunctionOpInterface provided utility method that prints the
  // function operation.
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// GenericCallOp
//===----------------------------------------------------------------------===//

void GenericCallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          StringRef callee, ArrayRef<mlir::Value> arguments) {
  // Generic call always returns an unranked Tensor initially.
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(arguments);
  state.addAttribute("callee",
                     mlir::SymbolRefAttr::get(builder.getContext(), callee));
}

/// Return the callee of the generic call operation, this is required by the
/// call interface.
CallInterfaceCallable GenericCallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// Set the callee for the generic call operation, this is required by the call
/// interface.
void GenericCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range GenericCallOp::getArgOperands() { return getInputs(); }

/// Get the argument operands to the called function as a mutable range, this is
/// required by the call interface.
MutableOperandRange GenericCallOp::getArgOperandsMutable() {
  return getInputsMutable();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult ReturnOp::verify() {
  // We know that the parent operation is a function, because of the 'HasParent'
  // trait attached to the operation definition.
  auto function = cast<FuncOp>((*this)->getParentOp());

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError() << "does not return the same number of values ("
                         << getNumOperands() << ") as the enclosing function ("
                         << results.size() << ")";

  // If the operation does not have an input, we are done.
  if (!hasOperand())
    return mlir::success();

  auto inputType = *operand_type_begin();
  auto resultType = results.front();

  // Check that the result type of the function matches the operand type.
  if (inputType == resultType ||
      llvm::isa<mlir::UnrankedTensorType>(inputType) ||
      llvm::isa<mlir::UnrankedTensorType>(resultType))
    return mlir::success();

  return emitError() << "type of return operand (" << inputType
                     << ") doesn't match function result type (" << resultType
                     << ")";
}

//===----------------------------------------------------------------------===//
// AIPU Operator Quantization Builders.
//===----------------------------------------------------------------------===//

/// This builder is called on all convolution operators except TransposeConv,
/// which has specialized output shape semantics. The builder also defines the
/// bitwidth of the output given the bit width of the input & weight content.
static void buildConv(OpBuilder &builder, OperationState &result,
                      Type outputType, Value input, Value weight, Value bias,
                      DenseI64ArrayAttr kernel, DenseI64ArrayAttr pad,
                      DenseI64ArrayAttr stride, DenseI64ArrayAttr dilation,
                      DictionaryAttr dictAttr) {

  result.addOperands({input, weight, bias});
  result.addAttribute("kernel", kernel);
  result.addAttribute("padding", pad);
  result.addAttribute("strides", stride);
  result.addAttribute("dilations", dilation);
  result.attributes.append(dictAttr.getValue().begin(),
                           dictAttr.getValue().end());
  result.addTypes(outputType);
}

/// Infer the output shape of the Conv2DOp, this is required by the shape
/// inference interface.
void Conv2DOp::inferShapes() { return; }

//===----------------------------------------------------------------------===//
// AIPU Pooling Operator Builder
//===----------------------------------------------------------------------===//
void Pool2DOp::build(OpBuilder &builder, OperationState &state, Type outputType,
                     Value input, DenseI64ArrayAttr kernel,
                     DenseI64ArrayAttr strides, DenseI64ArrayAttr dilations,
                     DenseI64ArrayAttr padding, BoolAttr ceil_mode,
                     BoolAttr count_include_pad, StringAttr method,
                     DictionaryAttr dictAttr) {
  state.addOperands(input);
  state.addAttribute("kernel", kernel);
  state.addAttribute("strides", strides);
  state.addAttribute("dilations", dilations);
  state.addAttribute("padding", padding);
  state.addAttribute("ceil_mode", ceil_mode);
  state.addAttribute("count_include_pad", count_include_pad);
  state.addAttribute("method", method);
  state.attributes.append(dictAttr.getValue().begin(),
                          dictAttr.getValue().end());
  state.addTypes(outputType);
}
void Pool2DOp::inferShapes() { return; }

//===----------------------------------------------------------------------===//
// TileOp
//===----------------------------------------------------------------------===//
void TileOp::build(OpBuilder &builder, OperationState &state, Type outputType,
                   Value input, DenseI64ArrayAttr repeats,
                   DictionaryAttr dictAttr) {
  state.addOperands(input);
  state.addAttribute("repeats", repeats);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}

LogicalResult TileOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  if (!inputType)
    return success();

  int64_t rank = inputType.getRank();
  auto repeats = getRepeats();
  if ((int64_t)repeats.size() != rank) {
    return emitOpError("repeats size (")
           << repeats.size() << ") must match input rank (" << rank << ")";
  }

  for (auto it : llvm::enumerate(repeats)) {
    if (it.value() < 1) {
      return emitOpError("repeats[")
             << it.index() << "] must be >= 1, got " << it.value();
    }
  }

  return success();
}

void TileOp::inferShapes() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  if (!inputType) {
    auto tensorType = llvm::dyn_cast<TensorType>(getInput().getType());
    if (!tensorType) {
      getResult().setType(getInput().getType());
      return;
    }
    getResult().setType(
        UnrankedTensorType::get(tensorType.getElementType()));
    return;
  }

  auto repeats = getRepeats();
  SmallVector<int64_t> outShape;
  for (int64_t i = 0; i < inputType.getRank(); ++i) {
    int64_t dim = inputType.getDimSize(i);
    if (ShapedType::isDynamic(dim)) {
      outShape.push_back(ShapedType::kDynamic);
    } else {
      outShape.push_back(dim * repeats[i]);
    }
  }

  getResult().setType(
      RankedTensorType::get(outShape, inputType.getElementType()));
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//
void ReshapeOp::build(OpBuilder &builder, OperationState &state,
                      Type outputType, Value input, DenseI64ArrayAttr shape,
                      DictionaryAttr dictAttr) {
  state.addOperands({input});
  state.addAttribute("shape", shape);
  state.attributes.append(dictAttr.getValue().begin(),
                          dictAttr.getValue().end());
  state.addTypes(outputType);
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//
void TransposeOp::build(OpBuilder &builder, OperationState &state,
                        Type outputType, Value input, ArrayAttr perm,
                        DictionaryAttr dictAttr) {
  state.addOperands(input);
  state.addAttribute("perm", perm);
  state.attributes.append(dictAttr.getValue().begin(),
                          dictAttr.getValue().end());
  state.addTypes(outputType);
}

void TransposeOp::inferShapes() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  if (!inputType) {
    auto tensorType = llvm::dyn_cast<TensorType>(getInput().getType());
    if (!tensorType) {
      getResult().setType(getInput().getType());
      return;
    }
    getResult().setType(
        UnrankedTensorType::get(tensorType.getElementType()));
    return;
  }

  ArrayRef<int64_t> inputShape = inputType.getShape();

  // Get permutation attribute defining dimension order
  ArrayAttr permAttr = getPerm();

  // Compute output shape by reordering dimensions according to perm
  SmallVector<int64_t> outputShape;
  for (auto attr : permAttr) {
    int64_t axis = llvm::cast<IntegerAttr>(attr).getInt();
    if (axis >= 0 && axis < (int64_t)inputShape.size())
      outputShape.push_back(inputShape[axis]);
    else
      outputShape.push_back(ShapedType::kDynamic);
  }

  // Update output tensor type with new shape
  getResult().setType(
      RankedTensorType::get(outputShape, inputType.getElementType()));
}

LogicalResult TransposeOp::verify() {
  // 1. Type Checking
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  ArrayAttr permAttr = getPerm();

  if (!inputType)
    return success();

  // 2. Permutation Validation
  const int64_t rank = inputType.getRank();

  // Check permutation length matches input rank
  if ((int64_t)permAttr.size() != rank) {
    return emitOpError() << "Permutation length (" << permAttr.size()
                       << ") must match input tensor rank (" << rank << ")";
  }

  // Track seen axes to detect duplicates
  llvm::SmallDenseSet<int64_t> seen;
  for (auto attr : permAttr) {
    int64_t axis = llvm::cast<IntegerAttr>(attr).getInt();
    if (axis < 0 || axis >= rank) {
      return emitOpError() << "Invalid axis index " << axis
                         << " in permutation (valid range: 0-" << (rank - 1)
                         << ")";
    }

    // Check for duplicate axes
    if (!seen.insert(axis).second) {
      return emitOpError() << "Duplicate axis " << axis << " in permutation";
    }
  }

  return success();
}
/// TODO:
/// Infer the output shape , this is required by the shape inference interface.
void DepthwiseConvOp::inferShapes() { return; }

void FullyConnectedOp::build(OpBuilder &builder, OperationState &state,
                             Type outputType, Value input, Value weight,
                             Value bias, DictionaryAttr dictAttr) {
  state.addOperands({input, weight, bias});
  state.attributes.append(dictAttr.getValue().begin(),
                          dictAttr.getValue().end());
  state.addTypes(outputType);
}
void FullyConnectedOp::inferShapes() { return; }

//===----------------------------------------------------------------------===//
// BatchNormOp
//===----------------------------------------------------------------------===//
void BatchNormOp::build(OpBuilder &builder, OperationState &state,
                        Type outputType, Value input, Value weight, Value bias,
                        IntegerAttr axis, FloatAttr epsilon,
                        DictionaryAttr dictAttr) {
  state.addOperands({input, weight, bias});
  state.addAttribute("axis", axis);
  state.addAttribute("epsilon", epsilon);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}

LogicalResult BatchNormOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  if (!inputType)
    return success();

  int64_t rank = inputType.getRank();
  if (rank < 2 || rank > 5) {
    return emitOpError("input rank must be between 2 and 4, got ") << rank;
  }

  int64_t axis = getAxis();
  int64_t normalizedAxis = axis;
  if (axis < 0)
    normalizedAxis += rank;
  if (normalizedAxis < 0 || normalizedAxis >= rank) {
    return emitOpError("axis value ")
           << axis << " out of range for input rank " << rank;
  }

  return success();
}

void BatchNormOp::inferShapes() { return; }

//===----------------------------------------------------------------------===//
// InstanceNormOp
//===----------------------------------------------------------------------===//
void LayerNormOp::build(OpBuilder &builder, OperationState &state,
  Type outputType, Value input, Value weight, Value bias,
  DenseI64ArrayAttr axis, FloatAttr epsilon,
  DictionaryAttr dictAttr,
  ArrayRef<Type> meanInvstddevTypes) {
state.addOperands({input, weight, bias});
state.addAttribute("axis", axis);
state.addAttribute("epsilon", epsilon);
if (dictAttr) {
state.attributes.append(dictAttr.getValue().begin(),
      dictAttr.getValue().end());
}
state.addTypes(outputType);
state.addTypes(meanInvstddevTypes);
}

LogicalResult LayerNormOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  if (!inputType)
    return success();

  int64_t rank = inputType.getRank();
  if (rank < 2 || rank > 4) {
    return emitOpError("input rank must be between 2 and 4, got ") << rank;
  }

  for (int64_t axisVal : getAxis()) {
    int64_t normalized = axisVal < 0 ? axisVal + rank : axisVal;
    if (normalized < 0 || normalized >= rank) {
      return emitOpError("axis value ")
             << axisVal << " out of range for input rank " << rank;
    }
  }

  // out_mean_invstddev is an optional flag carried in the generic dict
  bool outMeanInvstddev = false;
  if (auto dict = getDictAttr()) {
    if (auto flag = dict->getAs<BoolAttr>("out_mean_invstddev"))
      outMeanInvstddev = flag.getValue();
  }

  if (outMeanInvstddev && getMeanInvstddev().size() != 2u) {
    return emitOpError("when out_mean_invstddev is true, must have 2 "
                       "mean_invstddev results, got ")
           << getMeanInvstddev().size();
  }
  if (!outMeanInvstddev && getMeanInvstddev().size() != 0u) {
    return emitOpError("when out_mean_invstddev is false, must have 0 "
                       "mean_invstddev results, got ")
           << getMeanInvstddev().size();
  }

  return success();
}

void LayerNormOp::inferShapes() { return; }

//===----------------------------------------------------------------------===//
// InstanceNormOp
//===----------------------------------------------------------------------===//
void InstanceNormOp::build(OpBuilder &builder, OperationState &state,
                           Type outputType, Value input, Value weight, Value bias,
                           FloatAttr epsilon, DictionaryAttr dictAttr) {
  state.addOperands({input, weight, bias});
  state.addAttribute("epsilon", epsilon);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}

LogicalResult InstanceNormOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  if (!inputType)
    return success();

  int64_t rank = inputType.getRank();
  if (rank < 2 || rank > 5) {
    return emitOpError("input rank must be between 2 and 5, got ") << rank;
  }

  return success();
}

void InstanceNormOp::inferShapes() { return; }

//===----------------------------------------------------------------------===//
// GroupNormOp
//===----------------------------------------------------------------------===//
void GroupNormOp::build(OpBuilder &builder, OperationState &state,
                        Type outputType, Value input, Value weight, Value bias,
                        IntegerAttr axis, IntegerAttr group, FloatAttr epsilon,
                        DictionaryAttr dictAttr) {
  state.addOperands({input, weight, bias});
  state.addAttribute("axis", axis);
  state.addAttribute("group", group);
  state.addAttribute("epsilon", epsilon);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}

LogicalResult GroupNormOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  if (!inputType)
    return success();

  int64_t rank = inputType.getRank();
  if (rank < 3 || rank > 4) {
    return emitOpError("input rank must be 3 or 4, got ") << rank;
  }

  int64_t axis = getAxis();
  int64_t normalizedAxis = axis < 0 ? axis + rank : axis;
  if (normalizedAxis < 0 || normalizedAxis >= rank) {
    return emitOpError("axis value ")
           << axis << " out of range for input rank " << rank;
  }

  if (getGroup() <= 0) {
    return emitOpError("group must be positive, got ") << getGroup();
  }

  return success();
}

void GroupNormOp::inferShapes() { return; }

//===----------------------------------------------------------------------===//
// RMSNormOp
//===----------------------------------------------------------------------===//
void RMSNormOp::build(OpBuilder &builder, OperationState &state,
                      Type outputType, Value input, Value weight,
                      DenseI64ArrayAttr axis, FloatAttr epsilon,
                      DictionaryAttr dictAttr) {
  state.addOperands({input, weight});
  state.addAttribute("axis", axis);
  state.addAttribute("epsilon", epsilon);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}

LogicalResult RMSNormOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  if (!inputType)
    return success();

  int64_t rank = inputType.getRank();
  if (rank < 2 || rank > 4) {
    return emitOpError("input rank must be between 2 and 4, got ") << rank;
  }

  for (int64_t axisVal : getAxis()) {
    int64_t normalized = axisVal < 0 ? axisVal + rank : axisVal;
    if (normalized < 0 || normalized >= rank) {
      return emitOpError("axis value ")
             << axisVal << " out of range for input rank " << rank;
    }
  }

  // Basic type/shape consistency check between input and weight.
  auto weightType = dyn_cast<RankedTensorType>(getWeight().getType());
  if (weightType && inputType) {
    // Element types must match.
    if (inputType.getElementType() != weightType.getElementType()) {
      return emitOpError("input and weight must have the same element type");
    }

    auto axes = getAxis();
    if ((int64_t)axes.size() != weightType.getRank()) {
      return emitOpError("weight rank must equal axis size, got weight rank ")
             << weightType.getRank() << " and axis size " << axes.size();
    }

    for (auto it : llvm::enumerate(axes)) {
      int64_t axisVal = it.value();
      int64_t normalized = axisVal < 0 ? axisVal + rank : axisVal;
      if (!ShapedType::isDynamic(inputType.getDimSize(normalized)) &&
          !ShapedType::isDynamic(weightType.getDimSize(it.index())) &&
          inputType.getDimSize(normalized) !=
              weightType.getDimSize(it.index())) {
        return emitOpError("weight dimension ")
               << it.index() << " (size "
               << weightType.getDimSize(it.index())
               << ") does not match input dimension " << normalized
               << " (size " << inputType.getDimSize(normalized) << ")";
      }
    }
  }

  return success();
}

void RMSNormOp::inferShapes() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  if (!inputType) {
    auto tensorType = llvm::dyn_cast<TensorType>(getInput().getType());
    if (!tensorType) {
      getResult().setType(getInput().getType());
      return;
    }
    getResult().setType(
        UnrankedTensorType::get(tensorType.getElementType()));
    return;
  }

  getResult().setType(inputType);
}

void TransposeConv2DOp::inferShapes() { return; }

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//
void SoftmaxOp::build(OpBuilder &builder, OperationState &state,
                      Type outputType, Value input, mlir::IntegerAttr axis,
                      mlir::FloatAttr beta, DictionaryAttr dictAttr) {
  state.addOperands({input});
  state.addAttribute("axis", axis);
  state.addAttribute("beta", beta);
  state.attributes.append(dictAttr.getValue().begin(),
                          dictAttr.getValue().end());
  state.addTypes(outputType);
}

void SoftmaxOp::inferShapes() { return; }

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//
void ConcatOp::build(OpBuilder &builder, OperationState &state, Type outputType,
                     ValueRange inputs, mlir::IntegerAttr axis,
                     DictionaryAttr dictAttr) {
  state.addOperands(inputs); // Add all input operands in ConcatOp
  state.addAttribute("axis", axis);
  state.attributes.append(dictAttr.getValue().begin(),
                          dictAttr.getValue().end());
  state.addTypes(outputType);
}

void ConcatOp::inferShapes() { return; }

//===----------------------------------------------------------------------===//
// SplitOp
//===----------------------------------------------------------------------===//
void SplitOp::build(OpBuilder &builder, OperationState &state,
                    TypeRange outputType, Value input,
                    mlir::DenseI64ArrayAttr splits, mlir::IntegerAttr axis,
                    DictionaryAttr dictAttr) {
  state.addOperands(input);
  state.addAttribute("splits", splits);
  state.addAttribute("axis", axis);
  state.attributes.append(dictAttr.getValue().begin(),
                          dictAttr.getValue().end());
  state.addTypes(outputType);
}
void SplitOp::inferShapes() { return; }

//===----------------------------------------------------------------------===//
// ChannelShuffleOp
//===----------------------------------------------------------------------===//
void ChannelShuffleOp::build(OpBuilder &builder, OperationState &state,
                             TypeRange outputTypes, Value input,
                             IntegerAttr group, IntegerAttr splits,
                             DictionaryAttr dictAttr) {
  state.addOperands(input);
  state.addAttribute("group", group);
  state.addAttribute("splits", splits);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputTypes);
}

LogicalResult ChannelShuffleOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  if (inputType) {
    llvm::errs() << "ChannelShuffleOp inputType: " << inputType
                 << " (rank=" << inputType.getRank() << ", shape=[";
    for (int64_t i = 0; i < inputType.getRank(); ++i)
      llvm::errs() << (i ? ", " : "") << inputType.getDimSize(i);
    llvm::errs() << "], elementType=" << inputType.getElementType() << ")\n";
  } else {
    llvm::errs() << "ChannelShuffleOp inputType: (not RankedTensorType) "
                 << getInput().getType() << "\n";
  }
  if (!inputType)
    return success();

  int64_t rank = inputType.getRank();
  if (rank > 4) {
    return emitOpError("does not support input with rank > 4");
  }

  // ChannelShuffle 固定沿最后一维切分（axis = -1）
  int64_t axisVal = rank - 1;

  int64_t splitVal = getSplits();
  if (splitVal < 1) {
    return emitOpError("splits must be a positive integer");
  }

  if (!ShapedType::isDynamic(inputType.getDimSize(axisVal))) {
    int64_t dimAtAxis = inputType.getDimSize(axisVal);
    if (dimAtAxis % splitVal != 0) {
      return emitOpError("input dimension at axis ")
             << axisVal << " (" << dimAtAxis
             << ") must be divisible by splits (" << splitVal << ")";
    }
  }

  return success();
}

void ChannelShuffleOp::inferShapes() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  if (!inputType) {
    auto tensorType = llvm::dyn_cast<TensorType>(getInput().getType());
    if (!tensorType) {
      for (OpResult result : getResults())
        result.setType(getInput().getType());
      return;
    }
    auto unranked = UnrankedTensorType::get(tensorType.getElementType());
    for (OpResult result : getResults())
      result.setType(unranked);
    return;
  }

  int64_t rank = inputType.getRank();
  // ChannelShuffle 固定沿最后一维切分（axis = -1）
  int64_t axisVal = rank - 1;
  int64_t splitVal = getSplits();

  SmallVector<int64_t, 4> outShape(inputType.getShape().begin(),
                                   inputType.getShape().end());
  if (axisVal >= 0 && axisVal < rank &&
      !ShapedType::isDynamic(inputType.getDimSize(axisVal))) {
    int64_t dimAtAxis = inputType.getDimSize(axisVal);
    if (splitVal > 0 && dimAtAxis % splitVal == 0)
      outShape[axisVal] = dimAtAxis / splitVal;
  }

  auto outType =
      RankedTensorType::get(outShape, inputType.getElementType());
  for (OpResult result : getResults())
    result.setType(outType);
}

//===----------------------------------------------------------------------===//
// CropOp
//===----------------------------------------------------------------------===//
void CropOp::build(OpBuilder &builder, OperationState &state, Type outputType,
                   Value input, mlir::ArrayAttr crops,
                   DictionaryAttr dictAttr) {
  state.addOperands(input);
  state.addAttribute("crops", crops);
  state.attributes.append(dictAttr.getValue().begin(),
                          dictAttr.getValue().end());
  state.addTypes(outputType);
}
void CropOp::inferShapes() { return; }
LogicalResult CropOp::verify() {
  auto inputType = mlir::cast<ShapedType>(getInput().getType());

  int64_t dimIndex = 0;
  for (Attribute cropAttr : getCrops()) {
    auto denseCrop = mlir::dyn_cast<DenseI64ArrayAttr>(cropAttr);
    if (!denseCrop || denseCrop.size() != 2) {
      return emitOpError(
          "crops must be an array of [begin, end] pairs (DenseI64ArrayAttr)");
    }

    int64_t begin = denseCrop.asArrayRef()[0];
    int64_t end = denseCrop.asArrayRef()[1];

    if (begin < 0 || end < 0) {
      return emitOpError("crop values cannot be negative");
    }
    // dynamic shape skip this verification, only check static shape
    if (inputType.hasRank()) {
      int64_t dimSize = inputType.getDimSize(dimIndex);
      if (dimSize != ShapedType::kDynamic) { // only check static shape
        if (begin > dimSize) {
          return emitOpError()
                 << "crop begin value " << begin << " exceeds dimension size "
                 << dimSize << " in dimension " << dimIndex;
        }
        if (end > dimSize) {
          return emitOpError()
                 << "crop end value " << end << " exceeds dimension size "
                 << dimSize << " in dimension " << dimIndex;
        }
        if (begin > end) {
          return emitOpError()
                 << "crop begin value " << begin << " > end value " << end
                 << " in dimension " << dimIndex;
        }
      }
    }
    dimIndex++;
  }

  if (inputType.hasRank() &&
      (int64_t)getCrops().size() != inputType.getRank()) {
    return emitOpError("crops size must match input tensor rank");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DepthToSpaceOp
//===----------------------------------------------------------------------===//
void DepthToSpaceOp::build(OpBuilder &builder, OperationState &state,
                           Type outputType, Value input,
                           mlir::IntegerAttr block_size_x,
                           mlir::IntegerAttr block_size_y,
                           mlir::StringAttr mode, DictionaryAttr dictAttr) {
  state.addOperands(input);
  state.addAttribute("block_size_x", block_size_x);
  state.addAttribute("block_size_y", block_size_y);
  state.addAttribute("mode", mode);
  state.attributes.append(dictAttr.getValue().begin(),
                          dictAttr.getValue().end());
  state.addTypes(outputType);
}
void DepthToSpaceOp::inferShapes() { return; }

//===----------------------------------------------------------------------===//
// SpaceToDepthOp
//===----------------------------------------------------------------------===//
void SpaceToDepthOp::build(OpBuilder &builder, OperationState &state,
                           Type outputType, Value input,
                           mlir::IntegerAttr block_size_x,
                           mlir::IntegerAttr block_size_y,
                           DictionaryAttr dictAttr) {
  state.addOperands(input);

  state.addAttribute("block_size_x", block_size_x);
  state.addAttribute("block_size_y", block_size_y);

  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }

  state.addTypes(outputType);
}
void SpaceToDepthOp::inferShapes() { return; }

void BatchToSpaceOp::build(OpBuilder &builder, OperationState &state,
                           Type outputType, Value input,
                           IntegerAttr block_size_y, IntegerAttr block_size_x,
                           IntegerAttr crop_top, IntegerAttr crop_bottom,
                           IntegerAttr crop_left, IntegerAttr crop_right,
                           DictionaryAttr dictAttr) {
  state.addOperands(input);
  state.addAttribute("block_size_y", block_size_y);
  state.addAttribute("block_size_x", block_size_x);
  state.addAttribute("crop_top", crop_top);
  state.addAttribute("crop_bottom", crop_bottom);
  state.addAttribute("crop_left", crop_left);
  state.addAttribute("crop_right", crop_right);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}

LogicalResult BatchToSpaceOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  if (!inputType)
    return success();

  if (inputType.getRank() != 4) {
    return emitOpError("input rank must be 4");
  }

  int64_t n = inputType.getDimSize(0);
  int64_t h = inputType.getDimSize(2);
  int64_t w = inputType.getDimSize(3);

  int64_t by = getBlockSizeY();
  int64_t bx = getBlockSizeX();
  int64_t cropTop = getCropTop();
  int64_t cropBottom = getCropBottom();
  int64_t cropLeft = getCropLeft();
  int64_t cropRight = getCropRight();

  if (by <= 0 || bx <= 0) {
    return emitOpError("block_size_y and block_size_x must be positive");
  }

  if (cropTop < 0 || cropBottom < 0 || cropLeft < 0 || cropRight < 0) {
    return emitOpError("crop values cannot be negative");
  }

  if (!ShapedType::isDynamic(n)) {
    int64_t factor = by * bx;
    if (factor == 0 || n % factor != 0) {
      return emitOpError("input batch dimension ")
             << n << " must be divisible by block_size_y * block_size_x ("
             << factor << ")";
    }
  }

  if (!ShapedType::isDynamic(h)) {
    int64_t outH = h * by - cropTop - cropBottom;
    if (outH <= 0) {
      return emitOpError("computed output height ")
             << outH << " must be positive";
    }
  }

  if (!ShapedType::isDynamic(w)) {
    int64_t outW = w * bx - cropLeft - cropRight;
    if (outW <= 0) {
      return emitOpError("computed output width ")
             << outW << " must be positive";
    }
  }

  return success();
}

void BatchToSpaceOp::inferShapes() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  if (!inputType) {
    auto tensorType = llvm::dyn_cast<TensorType>(getInput().getType());
    if (!tensorType) {
      getResult().setType(getInput().getType());
      return;
    }
    getResult().setType(
        UnrankedTensorType::get(tensorType.getElementType()));
    return;
  }

  if (inputType.getRank() != 4) {
    getResult().setType(inputType);
    return;
  }

  int64_t n = inputType.getDimSize(0);
  int64_t c = inputType.getDimSize(1);
  int64_t h = inputType.getDimSize(2);
  int64_t w = inputType.getDimSize(3);

  int64_t by = getBlockSizeY();
  int64_t bx = getBlockSizeX();
  int64_t cropTop = getCropTop();
  int64_t cropBottom = getCropBottom();
  int64_t cropLeft = getCropLeft();
  int64_t cropRight = getCropRight();

  int64_t outN = ShapedType::kDynamic;
  if (!ShapedType::isDynamic(n) && by > 0 && bx > 0) {
    int64_t factor = by * bx;
    if (factor != 0 && n % factor == 0)
      outN = n / factor;
  }

  int64_t outH = ShapedType::kDynamic;
  if (!ShapedType::isDynamic(h)) {
    outH = h * by - cropTop - cropBottom;
  }

  int64_t outW = ShapedType::kDynamic;
  if (!ShapedType::isDynamic(w)) {
    outW = w * bx - cropLeft - cropRight;
  }

  SmallVector<int64_t, 4> outShape{outN, c, outH, outW};
  getResult().setType(
      RankedTensorType::get(outShape, inputType.getElementType()));
}

void SpaceToBatchOp::build(OpBuilder &builder, OperationState &state,
                           Type outputType, Value input,
                           IntegerAttr block_size_y, IntegerAttr block_size_x,
                           IntegerAttr pad_top, IntegerAttr pad_bottom,
                           IntegerAttr pad_left, IntegerAttr pad_right,
                           DictionaryAttr dictAttr) {
  state.addOperands(input);
  state.addAttribute("block_size_y", block_size_y);
  state.addAttribute("block_size_x", block_size_x);
  state.addAttribute("pad_top", pad_top);
  state.addAttribute("pad_bottom", pad_bottom);
  state.addAttribute("pad_left", pad_left);
  state.addAttribute("pad_right", pad_right);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}

LogicalResult SpaceToBatchOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  if (!inputType)
    return success();

  if (inputType.getRank() != 4) {
    return emitOpError("input rank must be 4");
  }

  int64_t h = inputType.getDimSize(2);
  int64_t w = inputType.getDimSize(3);

  int64_t by = getBlockSizeY();
  int64_t bx = getBlockSizeX();
  int64_t padTop = getPadTop();
  int64_t padBottom = getPadBottom();
  int64_t padLeft = getPadLeft();
  int64_t padRight = getPadRight();

  if (by <= 0 || bx <= 0) {
    return emitOpError("block_size_y and block_size_x must be positive");
  }

  if (padTop < 0 || padBottom < 0 || padLeft < 0 || padRight < 0) {
    return emitOpError("pad values cannot be negative");
  }

  if (!ShapedType::isDynamic(h)) {
    int64_t paddedH = h + padTop + padBottom;
    if (paddedH % by != 0) {
      return emitOpError("(height + pad_top + pad_bottom) ")
             << paddedH << " must be divisible by block_size_y " << by;
    }
  }

  if (!ShapedType::isDynamic(w)) {
    int64_t paddedW = w + padLeft + padRight;
    if (paddedW % bx != 0) {
      return emitOpError("(width + pad_left + pad_right) ")
             << paddedW << " must be divisible by block_size_x " << bx;
    }
  }

  return success();
}

void SpaceToBatchOp::inferShapes() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  if (!inputType) {
    auto tensorType = llvm::dyn_cast<TensorType>(getInput().getType());
    if (!tensorType) {
      getResult().setType(getInput().getType());
      return;
    }
    getResult().setType(
        UnrankedTensorType::get(tensorType.getElementType()));
    return;
  }

  if (inputType.getRank() != 4) {
    getResult().setType(inputType);
    return;
  }

  int64_t n = inputType.getDimSize(0);
  int64_t c = inputType.getDimSize(1);
  int64_t h = inputType.getDimSize(2);
  int64_t w = inputType.getDimSize(3);

  int64_t by = getBlockSizeY();
  int64_t bx = getBlockSizeX();
  int64_t padTop = getPadTop();
  int64_t padBottom = getPadBottom();
  int64_t padLeft = getPadLeft();
  int64_t padRight = getPadRight();

  int64_t outN = ShapedType::kDynamic;
  if (!ShapedType::isDynamic(n) && by > 0 && bx > 0) {
    outN = n * by * bx;
  }

  int64_t outH = ShapedType::kDynamic;
  if (!ShapedType::isDynamic(h)) {
    int64_t paddedH = h + padTop + padBottom;
    if (by > 0 && paddedH % by == 0)
      outH = paddedH / by;
  }

  int64_t outW = ShapedType::kDynamic;
  if (!ShapedType::isDynamic(w)) {
    int64_t paddedW = w + padLeft + padRight;
    if (bx > 0 && paddedW % bx == 0)
      outW = paddedW / bx;
  }

  SmallVector<int64_t, 4> outShape{outN, c, outH, outW};
  getResult().setType(
      RankedTensorType::get(outShape, inputType.getElementType()));
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//
void ReduceOp::build(OpBuilder &builder, OperationState &state, Type outputType,
                     Value input, mlir::DenseI64ArrayAttr axis,
                     mlir::StringAttr method, DictionaryAttr dictAttr) {
  state.addOperands(input);
  state.addAttribute("axis", axis);
  state.addAttribute("method", method);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}
LogicalResult ReduceOp::verify() {
  auto inputType = mlir::dyn_cast<RankedTensorType>(getInput().getType());
  const int64_t inputRank = inputType.getRank();

  for (auto axis : getAxis()) {
    if (axis < 0 || axis >= inputRank) {
      return emitOpError("axis value ")
             << axis << " out of range for input rank " << inputRank;
    }
  }

  llvm::StringRef method = getMethod();
  if (!method.equals_insensitive("mean") && !method.equals_insensitive("sum") &&
      !method.equals_insensitive("max") && !method.equals_insensitive("min")) {
    return emitOpError("unsupported reduction method: ") << method;
  }

  return success();
}
void ReduceOp::inferShapes() { return; }

void ArgMinMaxOp::build(OpBuilder &builder, OperationState &state,
                        Type outputType, Value input, IntegerAttr axis,
                        StringAttr method, DictionaryAttr dictAttr) {
  state.addOperands(input);
  state.addAttribute("axis", axis);
  state.addAttribute("method", method);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}

LogicalResult ArgMinMaxOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  if (!inputType)
    return success();

  int64_t rank = inputType.getRank();
  int64_t axis = getAxis();
  if (axis < 0 || axis >= rank) {
    return emitOpError("axis value ") << axis
           << " out of range for input rank " << rank;
  }

  StringRef method = getMethod();
  if (!method.equals_insensitive("min") && !method.equals_insensitive("max")) {
    return emitOpError("unsupported method: ") << method;
  }

  return success();
}

void ArgMinMaxOp::inferShapes() { return; }

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//
void GatherOp::build(OpBuilder &builder, OperationState &state, Type outputType,
                     Value input, Value indices, mlir::IntegerAttr axis,
                     mlir::IntegerAttr batch_dims, DictionaryAttr dictAttr) {
  state.addOperands({input, indices});
  state.addAttribute("axis", axis);
  state.addAttribute("batch_dims", batch_dims);

  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}
void GatherOp::inferShapes() { return; }

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//
void SliceOp::build(OpBuilder &builder, OperationState &state, Type outputType,
                    Value input, mlir::DenseI64ArrayAttr begin,
                    mlir::DenseI64ArrayAttr end,
                    mlir::DenseI64ArrayAttr strides, DictionaryAttr dictAttr) {
  state.addOperands(input);
  state.addAttribute("begin", begin);
  state.addAttribute("end", end);
  state.addAttribute("strides", strides);

  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}
void SliceOp::inferShapes() { return; }
//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//
void PadOp::build(OpBuilder &builder, OperationState &state, Type outputType,
                  Value input, ArrayAttr pads, StringAttr mode,
                  DictionaryAttr dictAttr) {
  state.addOperands(input);
  state.addAttribute("pads", pads);
  state.addAttribute("mode", mode);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}

void PadOp::inferShapes() {
  auto inputType = mlir::cast<ShapedType>(getInput().getType());
  if (!inputType.hasRank()) {
    getResult().setType(UnrankedTensorType::get(inputType.getElementType()));
    return;
  }

  SmallVector<int64_t> outShape;
  ArrayRef<int64_t> inShape = inputType.getShape();
  ArrayAttr pads = getPads();

  for (auto dim : llvm::enumerate(pads)) {
    auto padPair = mlir::dyn_cast<DenseI64ArrayAttr>(dim.value());
    if (!padPair || padPair.size() != 2) {
      getResult().setType(UnrankedTensorType::get(inputType.getElementType()));
      return;
    }

    int64_t padBegin = padPair[0];
    int64_t padEnd = padPair[1];
    int64_t dimSize = inShape[dim.index()];

    if (ShapedType::isDynamic(dimSize)) {
      outShape.push_back(ShapedType::kDynamic);
    } else {
      outShape.push_back(dimSize + padBegin + padEnd);
    }
  }

  getResult().setType(
      RankedTensorType::get(outShape, inputType.getElementType()));
}

LogicalResult PadOp::verify() {
  auto inputType = mlir::cast<ShapedType>(getInput().getType());
  ArrayAttr pads = getPads();

  if (inputType.hasRank() && pads.size() != inputType.getRank()) {
    return emitOpError() << "pads size " << pads.size()
                         << " must match input rank " << inputType.getRank();
  }

  for (auto dimPad : llvm::enumerate(pads)) {
    auto padPair = mlir::dyn_cast<DenseI64ArrayAttr>(dimPad.value());
    if (!padPair || padPair.size() != 2) {
      return emitOpError() << "pad dimension " << dimPad.index()
                           << " must be a DenseI64ArrayAttr with 2 elements";
    }

    int64_t padBegin = padPair[0];
    int64_t padEnd = padPair[1];

    if (padBegin < 0 || padEnd < 0) {
      return emitOpError() << "negative pad value in dimension "
                           << dimPad.index() << ": [" << padBegin << ", "
                           << padEnd << "]";
    }
  }

  llvm::StringRef mode = getMode();
  if (!mode.equals_insensitive("CONSTANT") &&
      !mode.equals_insensitive("REFLECT") &&
      !mode.equals_insensitive("SYMMETRIC")) {
    return emitOpError() << "invalid padding mode: " << mode
                         << ", must be CONSTANT/REFLECT/SYMMETRIC";
  }

  return success();
}
//===----------------------------------------------------------------------===//
// ResizeOp
//===----------------------------------------------------------------------===//
void ResizeOp::build(OpBuilder &builder, OperationState &state, Type outputType,
                     Value input, FloatAttr ratio_x, FloatAttr ratio_y,
                     StringAttr method, StringAttr mode,
                     DictionaryAttr dictAttr) {
  state.addOperands(input);
  state.addAttribute("ratio_x", ratio_x);
  state.addAttribute("ratio_y", ratio_y);
  state.addAttribute("method", method);
  state.addAttribute("mode", mode);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}
LogicalResult ResizeOp::verify() {
  auto inputType = dyn_cast<ShapedType>(getInput().getType());
  auto outputType = dyn_cast<ShapedType>(getOutput().getType());

  if (inputType && outputType && inputType.hasRank() && outputType.hasRank() &&
      inputType.getRank() != outputType.getRank()) {
    return emitOpError() << "input rank " << inputType.getRank()
                         << " must match output rank " << outputType.getRank();
  }

  if (getRatioX().convertToDouble() <= 0.0 ||
      getRatioY().convertToDouble() <= 0.0) {
    return emitOpError("ratio_x and ratio_y must be positive");
  }

  StringRef method = getMethod();
  if (!method.equals_insensitive("BILINEAR") &&
      !method.equals_insensitive("NEAREST")) {
    return emitOpError() << "invalid resize method: " << method
                         << ", must be BILINEAR/NEAREST";
  }

  StringRef mode = getMode();
  if (!mode.equals_insensitive("HALF_PIXEL") &&
      !mode.equals_insensitive("ALIGN_CORNERS") &&
      !mode.equals_insensitive("ASYMMETRIC") &&
      !mode.equals_insensitive("PYTORCH_HALF_PIXEL") &&
      !mode.equals_insensitive("TF_HALF_PIXEL_FOR_NN")) {
    return emitOpError() << "invalid resize mode: " << mode
                         << ", must be HALF_PIXEL/ALIGN_CORNERS/ASYMMETRIC/"
                            "PYTORCH_HALF_PIXEL/TF_HALF_PIXEL_FOR_NN";
  }

  if (auto dict = getDictAttr()) {
    if (auto nmAttr = dict->getAs<StringAttr>("nearest_mode")) {
      StringRef nearestMode = nmAttr.getValue();
      if (!nearestMode.equals_insensitive("ROUND_PREFER_FLOOR") &&
          !nearestMode.equals_insensitive("ROUND_PREFER_CEIL") &&
          !nearestMode.equals_insensitive("FLOOR") &&
          !nearestMode.equals_insensitive("CEIL")) {
        return emitOpError()
               << "invalid nearest_mode: " << nearestMode
               << ", must be ROUND_PREFER_FLOOR/ROUND_PREFER_CEIL/FLOOR/CEIL";
      }
      if (!method.equals_insensitive("NEAREST")) {
        return emitOpError()
               << "nearest_mode is only valid when method is NEAREST, "
                  "but method="
               << method;
      }
    }
  }

  return success();
}
void ResizeOp::inferShapes() { return; }
//===----------------------------------------------------------------------===//
// GridSampleOp
//===----------------------------------------------------------------------===//
void GridSampleOp::build(OpBuilder &builder, OperationState &state,
                         Type outputType, Value input, Value grid,
                         StringAttr mode, StringAttr padding_mode,
                         BoolAttr align_corners, DictionaryAttr dictAttr) {
  state.addOperands({input, grid});
  state.addAttribute("mode", mode);
  state.addAttribute("padding_mode", padding_mode);
  state.addAttribute("align_corners", align_corners);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}

LogicalResult GridSampleOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto gridType = dyn_cast<RankedTensorType>(getGrid().getType());

  // Allow dynamic/unranked types to pass; detailed checks only for ranked.
  if (!inputType || !gridType)
    return success();

  int64_t inRank = inputType.getRank();
  int64_t gridRank = gridType.getRank();

  // Both input and grid must be 4D or 5D.
  auto isValidRank = [](int64_t r) { return r == 4 || r == 5; };
  if (!isValidRank(inRank) || !isValidRank(gridRank)) {
    return emitOpError("input and grid must be rank 4 or 5, but got ranks (")
           << inRank << ", " << gridRank << ")";
  }

  // Batch dimension must match.
  if (!ShapedType::isDynamic(inputType.getDimSize(0)) &&
      !ShapedType::isDynamic(gridType.getDimSize(0)) &&
      inputType.getDimSize(0) != gridType.getDimSize(0)) {
    return emitOpError("input and grid batch dimension must match, but got ")
           << inputType.getDimSize(0) << " and " << gridType.getDimSize(0);
  }

  // Grid last "channel" dimension must be 2 for 4D or 3 for 5D (if statically known).
  bool is5D = gridRank == 5;
  int64_t gridChannelIndex = is5D ? 4 : 3;
  int64_t expectedCoordDim = is5D ? 3 : 2;

  if (!ShapedType::isDynamic(gridType.getDimSize(gridChannelIndex)) &&
      gridType.getDimSize(gridChannelIndex) != expectedCoordDim) {
    return emitOpError("grid last coordinate dimension must be ")
           << expectedCoordDim << " for " << gridRank
           << "D grid, but got " << gridType.getDimSize(gridChannelIndex);
  }

  // Optional: basic validation of mode / padding_mode strings.
  StringRef mode = getMode();
  if (!mode.equals_insensitive("bilinear") &&
      !mode.equals_insensitive("nearest") &&
      !mode.equals_insensitive("linear") &&
      !mode.equals_insensitive("bicubic")) {
    return emitOpError("unsupported grid_sample mode: ") << mode;
  }

  StringRef paddingMode = getPaddingMode();
  if (!paddingMode.equals_insensitive("zeros") &&
      !paddingMode.equals_insensitive("border") &&
      !paddingMode.equals_insensitive("reflection")) {
    return emitOpError("unsupported grid_sample padding_mode: ")
           << paddingMode;
  }

  return success();
}

void GridSampleOp::inferShapes() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto gridType = dyn_cast<RankedTensorType>(getGrid().getType());

  if (!inputType || !gridType) {
    // Fall back: keep existing result type if shapes are not known.
    return;
  }

  int64_t inRank = inputType.getRank();
  int64_t gridRank = gridType.getRank();
  if (!((inRank == 4 || inRank == 5) && (gridRank == 4 || gridRank == 5))) {
    return;
  }

  // Follow backend shape rule:
  // new_shape = input.shape
  // new_shape[1] = grid.shape[1]
  // new_shape[2] = grid.shape[2]
  // if 5D: new_shape[3] = grid.shape[3]
  SmallVector<int64_t, 5> outShape(inputType.getShape().begin(),
                                   inputType.getShape().end());

  if (gridRank >= 3) {
    outShape[1] = gridType.getDimSize(1);
    outShape[2] = gridType.getDimSize(2);
    if (inRank == 5 && gridRank == 5)
      outShape[3] = gridType.getDimSize(3);
  }

  getResult().setType(
      RankedTensorType::get(outShape, inputType.getElementType()));
}
//===----------------------------------------------------------------------===//
// ReverseSequenceOp
//===----------------------------------------------------------------------===//
void ReverseSequenceOp::build(OpBuilder &builder, OperationState &state,
                              Type outputType, Value input, Value seq_lens,
                              IntegerAttr batch_axis, IntegerAttr time_axis,
                              DictionaryAttr dictAttr) {
  state.addOperands({input, seq_lens});
  state.addAttribute("batch_axis", batch_axis);
  state.addAttribute("time_axis", time_axis);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}
void ReverseSequenceOp::inferShapes() { return; }
//===----------------------------------------------------------------------===//
// SquaredDifferenceOp
//===----------------------------------------------------------------------===//
void SquaredDifferenceOp::build(OpBuilder &builder, OperationState &state,
                                Type outputType, Value x, Value y,
                                DictionaryAttr dictAttr) {
  state.addOperands({x, y});
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}
void SquaredDifferenceOp::inferShapes() { return; }
//===----------------------------------------------------------------------===//
// SqrtOp
//===----------------------------------------------------------------------===//
void SqrtOp::build(OpBuilder &builder, OperationState &state, Type outputType,
                   Value input, DictionaryAttr dictAttr) {
  state.addOperands(input);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}

LogicalResult SqrtOp::verify() {
  auto inputType = getInput().getType();
  if (!mlir::isa<RankedTensorType>(inputType))
    return emitOpError("expected ranked tensor type for input");
  return success();
}

void SqrtOp::inferShapes() {
  auto inputType = mlir::cast<RankedTensorType>(getInput().getType());
  getResult().setType(inputType);
}

//===----------------------------------------------------------------------===//
// RsqrtOp
//===----------------------------------------------------------------------===//
void RsqrtOp::build(OpBuilder &builder, OperationState &state, Type outputType,
                    Value input, DictionaryAttr dictAttr) {
  state.addOperands(input);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}
void RsqrtOp::inferShapes() { return; }
//===----------------------------------------------------------------------===//
// CosineOp
//===----------------------------------------------------------------------===//
void CosineOp::build(OpBuilder &builder, OperationState &state,
                     Type outputType, Value input, DictionaryAttr dictAttr) {
  state.addOperands(input);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}

LogicalResult CosineOp::verify() {
  auto inputType = getInput().getType();
  if (!mlir::isa<RankedTensorType>(inputType))
    return emitOpError("expected ranked tensor type for input");
  return success();
}

void CosineOp::inferShapes() {
  auto inputType = mlir::cast<RankedTensorType>(getInput().getType());
  getResult().setType(inputType);
}

//===----------------------------------------------------------------------===//
// ExpOp
//===----------------------------------------------------------------------===//
void ExpOp::build(OpBuilder &builder, OperationState &state, Type outputType,
                  Value input, DictionaryAttr dictAttr) {
  state.addOperands(input);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}

void ExpOp::inferShapes() {
  auto inputType = mlir::cast<RankedTensorType>(getInput().getType());
  getResult().setType(inputType);
}

//===----------------------------------------------------------------------===//
// LogOp
// Note: log(x) requires x > 0. Compile-time verification of operand values
// is not feasible since only types/shapes are available, not runtime values.
//===----------------------------------------------------------------------===//
void LogOp::build(OpBuilder &builder, OperationState &state, Type outputType,
                  Value input, DictionaryAttr dictAttr) {
  state.addOperands(input);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}

void LogOp::inferShapes() {
  auto inputType = mlir::cast<RankedTensorType>(getInput().getType());
  getResult().setType(inputType);
}

//===----------------------------------------------------------------------===//
// QuantizeOp
//===----------------------------------------------------------------------===//
void QuantizeOp::build(OpBuilder &builder, OperationState &state,
                       Type outputType, Value input, DictionaryAttr dictAttr) {
  state.addOperands(input);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}
void QuantizeOp::inferShapes() { return; }
//===----------------------------------------------------------------------===//
// BasicLSTMOp
//===----------------------------------------------------------------------===//
void BasicLSTMOp::build(OpBuilder &builder, OperationState &state,
                        Type outputType, Value input, Value hidden_state,
                        Value cell_state, StringAttr out_sequence,
                        StringAttr activations, StringAttr direction,
                        DictionaryAttr dictAttr) {
  state.addOperands({input, hidden_state, cell_state});
  state.addAttribute("out_sequence", out_sequence);
  state.addAttribute("activations", activations);
  state.addAttribute("direction", direction);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}
void BasicLSTMOp::inferShapes() { return; }
//===----------------------------------------------------------------------===//
// MatMulOp
//===----------------------------------------------------------------------===//
void MatMulOp::build(OpBuilder &builder, OperationState &state, Type outputType,
                     Value input, Value weight, BoolAttr trans_a,
                     BoolAttr trans_b, DictionaryAttr dictAttr) {
  state.addOperands({input, weight});
  state.addAttribute("trans_a", trans_a);
  state.addAttribute("trans_b", trans_b);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes(outputType);
}
void MatMulOp::inferShapes() { return; }

//===----------------------------------------------------------------------===//
// RegionOp
//===----------------------------------------------------------------------===//
void RegionOp::build(OpBuilder &builder, OperationState &state,
                     Type scoresType, Type coordinatesType, Type boxesNumberType,
                     Type validClassesType, Type totalClassType, Value input,
                     FloatAttr obj_thresh, IntegerAttr max_box_num,
                     IntegerAttr box_per_grid, IntegerAttr class_num,
                     IntegerAttr grid_width, IntegerAttr grid_height,
                     BoolAttr grid_compensate, DenseF64ArrayAttr anchors,
                     DictionaryAttr dictAttr) {
  state.addOperands({input});
  state.addAttribute("obj_thresh", obj_thresh);
  state.addAttribute("max_box_num", max_box_num);
  state.addAttribute("box_per_grid", box_per_grid);
  state.addAttribute("class_num", class_num);
  state.addAttribute("grid_width", grid_width);
  state.addAttribute("grid_height", grid_height);
  state.addAttribute("grid_compensate", grid_compensate);
  state.addAttribute("anchors", anchors);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes({scoresType, coordinatesType, boxesNumberType,
                  validClassesType, totalClassType});
}

LogicalResult RegionOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  if (!inputType)
    return success();

  // Input should be 5D: [batch_size, s, s, anchor_num, num_classes + 5]
  if (inputType.getRank() != 5) {
    return emitOpError("input must be 5D tensor, got rank ")
           << inputType.getRank();
  }

  // Check that the last dimension has at least 6 elements (5 + num_classes)
  int64_t lastDim = inputType.getDimSize(4);
  if (!ShapedType::isDynamic(lastDim) && lastDim < 6) {
    return emitOpError("input last dimension must be >= 6, got ") << lastDim;
  }

  // Check anchors size is valid (should be 2 * box_per_grid values)
  int32_t boxPerGrid = getBoxPerGrid();
  auto anchors = getAnchors();
  if ((int64_t)anchors.size() != boxPerGrid * 2) {
    return emitOpError("anchors size must be 2*box_per_grid (")
           << boxPerGrid * 2 << "), got " << anchors.size();
  }

  return success();
}
void RegionOp::inferShapes() { return; }

//===----------------------------------------------------------------------===//
// NmsOp
//===----------------------------------------------------------------------===//
void NmsOp::build(OpBuilder &builder, OperationState &state,
                  Type selectedCoordinatesType, Type selectedBoxNumberType,
                  Type selectedScoreType, Type selectedIndicesType,
                  Value box_coor, Value box_num_per_class,
                  Value class_num_per_batch, Value score,
                  FloatAttr score_threshold, FloatAttr iou_threshold,
                  IntegerAttr max_output_size, FloatAttr soft_nms_sigma,
                  BoolAttr center_point_box, StringAttr method,
                  DictionaryAttr dictAttr) {
  state.addOperands({box_coor, box_num_per_class, class_num_per_batch, score});
  state.addAttribute("score_threshold", score_threshold);
  state.addAttribute("iou_threshold", iou_threshold);
  state.addAttribute("max_output_size", max_output_size);
  state.addAttribute("soft_nms_sigma", soft_nms_sigma);
  state.addAttribute("center_point_box", center_point_box);
  state.addAttribute("method", method);
  if (dictAttr) {
    state.attributes.append(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());
  }
  state.addTypes({selectedCoordinatesType, selectedBoxNumberType,
                  selectedScoreType, selectedIndicesType});
}

LogicalResult NmsOp::verify() {
  // box_coor must be 3D with last dim == 4
  if (auto boxCoorType = dyn_cast<RankedTensorType>(getBoxCoor().getType())) {
    if (boxCoorType.getRank() != 3) {
      return emitOpError("box_coor must be 3D, got rank ")
             << boxCoorType.getRank();
    }
    int64_t lastDim = boxCoorType.getDimSize(2);
    if (!ShapedType::isDynamic(lastDim) && lastDim != 4) {
      return emitOpError("box_coor last dimension must be 4, got ") << lastDim;
    }
  }

  // box_num_per_class / class_num_per_batch / score must all be 2D
  auto check2D = [&](Value v, StringRef name) -> LogicalResult {
    if (auto t = dyn_cast<RankedTensorType>(v.getType())) {
      if (t.getRank() != 2)
        return emitOpError(name) << " must be 2D, got rank " << t.getRank();
    }
    return success();
  };
  if (failed(check2D(getBoxNumPerClass(), "box_num_per_class")))
    return failure();
  if (failed(check2D(getClassNumPerBatch(), "class_num_per_batch")))
    return failure();
  if (failed(check2D(getScore(), "score")))
    return failure();

  // Batch dims must match across inputs.
  auto boxCoorType = dyn_cast<RankedTensorType>(getBoxCoor().getType());
  auto boxNumType = dyn_cast<RankedTensorType>(getBoxNumPerClass().getType());
  auto classNumType =
      dyn_cast<RankedTensorType>(getClassNumPerBatch().getType());
  auto scoreType = dyn_cast<RankedTensorType>(getScore().getType());
  if (boxCoorType && boxNumType && classNumType && scoreType) {
    int64_t batch = boxCoorType.getDimSize(0);
    auto sameBatch = [&](RankedTensorType t) {
      int64_t b = t.getDimSize(0);
      return ShapedType::isDynamic(batch) || ShapedType::isDynamic(b) ||
             b == batch;
    };
    if (!sameBatch(boxNumType) || !sameBatch(classNumType) ||
        !sameBatch(scoreType)) {
      return emitOpError(
          "all inputs must have the same batch size (first dimension)");
    }
  }

  // method must be HARD or GAUSSIAN
  auto method = getMethod();
  if (method != "HARD" && method != "GAUSSIAN") {
    return emitOpError("method must be 'HARD' or 'GAUSSIAN', got '")
           << method << "'";
  }

  return success();
}

void NmsOp::inferShapes() { return; }

//===----------------------------------------------------------------------===//
// AIPU Attribute Definitions.
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "aipugc/AipuAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aipugc/AipuOps.cpp.inc"
