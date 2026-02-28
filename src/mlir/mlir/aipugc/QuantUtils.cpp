//===- QuantUtils.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains AIPU numerical support functions and quantization
// attribute builders.
//
//===----------------------------------------------------------------------===//
#include "aipugc/QuantUtils.h"
#include "aipugc/Dialect.h"
#include "aipugc/utils.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/Support/Debug.h"
#include <sstream>

using namespace mlir;
using namespace mlir::aipugc::detail;
namespace mlir {
namespace aipugc {
namespace {
// Returns true on success, false when numBits is unsupported.
bool getDefaultStorageParams(unsigned numBits, bool isSigned, MLIRContext *ctx,
                             Type &storageType, int64_t &qmin, int64_t &qmax) {
  // Hard-coded type mapping from TFLite.
  if (numBits <= 8) {
    if (isSigned) {
      qmin = -128;
      qmax = 127;
      storageType = IntegerType::get(ctx, 8, IntegerType::Signed);
    } else {
      qmin = 0;
      qmax = 255;
      storageType = IntegerType::get(ctx, 8, IntegerType::Unsigned);
    }
  } else if (numBits <= 16) {
    storageType = IntegerType::get(ctx, 16);
    if (isSigned) {
      qmin = -32768;
      qmax = 32767;
      storageType = IntegerType::get(ctx, 16, IntegerType::Signed);
    } else {
      qmin = 0;
      qmax = 65535;
      storageType = IntegerType::get(ctx, 16, IntegerType::Unsigned);
    }
  } else if (numBits <= 32) {
    if (isSigned) {
      qmin = std::numeric_limits<int32_t>::min();
      qmax = std::numeric_limits<int32_t>::max();
      storageType = IntegerType::get(ctx, 32, IntegerType::Signed);
    } else {
      qmin = std::numeric_limits<uint32_t>::min();
      qmax = std::numeric_limits<uint32_t>::max();
      storageType = IntegerType::get(ctx, 32, IntegerType::Unsigned);
    }
  } else if (numBits <= 64) {
    if (isSigned) {
      qmin = std::numeric_limits<int64_t>::min();
      qmax = std::numeric_limits<int64_t>::max();
      storageType = IntegerType::get(ctx, 64, IntegerType::Signed);
    } else {
      qmin = 0;
      qmax = std::numeric_limits<uint64_t>::max();
      storageType = IntegerType::get(ctx, 64, IntegerType::Unsigned);
    }
  } else {
    return false;
  }

  return true;
}
} // namespace

//
//
// QuantizedType
unsigned QuantizedType::getFlags() const {
  return static_cast<ImplType *>(impl)->flags;
}

bool QuantizedType::classof(Type type) {
  return llvm::isa<AIPUDialect>(type.getDialect());
}

LogicalResult
QuantizedType::verify(function_ref<InFlightDiagnostic()> emitError,
                      unsigned flags, Type storageType, Type expressedType,
                      int64_t storageTypeMin, int64_t storageTypeMax) {
  // Verify that the storage type is integral.
  // This restriction may be lifted at some point in favor of using bf16
  // or f16 as exact representations on hardware where that is advantageous.
  auto intStorageType = llvm::dyn_cast<IntegerType>(storageType);
  if (!intStorageType)
    return emitError() << "storage type must be integral";
  unsigned integralWidth = intStorageType.getWidth();

  // Verify storage width.
  if (integralWidth == 0 || integralWidth > MaxStorageBits)
    return emitError() << "illegal storage type size: " << integralWidth;

  // Verify storageTypeMin and storageTypeMax.
  bool isSigned =
      (flags & QuantizationFlags::Signed) == QuantizationFlags::Signed;
  int64_t defaultIntegerMin =
      getDefaultMinimumForInteger(isSigned, integralWidth);
  int64_t defaultIntegerMax =
      getDefaultMaximumForInteger(isSigned, integralWidth);
  if (storageTypeMax - storageTypeMin <= 0 ||
      storageTypeMin < defaultIntegerMin ||
      storageTypeMax > defaultIntegerMax) {
    return emitError() << "illegal storage min and storage max: ("
                       << storageTypeMin << ":" << storageTypeMax << ")";
  }
  return success();
}

Type QuantizedType::getStorageType() const {
  return static_cast<ImplType *>(impl)->storageType;
}

int64_t QuantizedType::getStorageTypeMin() const {
  return static_cast<ImplType *>(impl)->storageTypeMin;
}

int64_t QuantizedType::getStorageTypeMax() const {
  return static_cast<ImplType *>(impl)->storageTypeMax;
}

unsigned QuantizedType::getStorageTypeIntegralWidth() const {
  // NOTE: If ever supporting non-integral storage types, some other scheme
  // for determining the width will be needed.
  return static_cast<ImplType *>(impl)->storageType.getIntOrFloatBitWidth();
}

Type QuantizedType::getExpressedType() const {
  return static_cast<ImplType *>(impl)->expressedType;
}
bool QuantizedType::isCompatibleExpressedType(Type candidateExpressedType) {
  if (llvm::isa<ShapedType>(candidateExpressedType)) {
    return llvm::cast<ShapedType>(candidateExpressedType).getElementType() ==
           getExpressedType();
  }
  return candidateExpressedType == getExpressedType();
}

QuantizedType
QuantizedType::getQuantizedElementType(Type primitiveOrContainerType) {
  if (llvm::isa<ShapedType>(primitiveOrContainerType)) {
    Type elementType =
        llvm::cast<ShapedType>(primitiveOrContainerType).getElementType();
    return llvm::dyn_cast<QuantizedType>(elementType);
  }
  return llvm::dyn_cast<QuantizedType>(primitiveOrContainerType);
}

Type QuantizedType::castFromStorageType(Type candidateType) {
  if (candidateType == getStorageType()) {
    // i.e. i32 -> quant<"uniform[i8:f32]{1.0}">
    return *this;
  }
  if (llvm::isa<RankedTensorType>(candidateType)) {
    // i.e. tensor<4xi8> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
    return RankedTensorType::get(
        llvm::cast<RankedTensorType>(candidateType).getShape(),
        getStorageType());
  }
  if (llvm::isa<UnrankedTensorType>(candidateType)) {
    // i.e. tensor<i8> -> tensor<!quant<"uniform[i8:f32]{1.0}">>
    return UnrankedTensorType::get(getStorageType());
  }
  if (llvm::isa<VectorType>(candidateType)) {
    // i.e. tensor<4xi8> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
    return VectorType::get(llvm::cast<VectorType>(candidateType).getShape(),
                           getStorageType());
  }

  return nullptr;
}

Type QuantizedType::castToStorageType(Type quantizedType) {
  if (llvm::isa<QuantizedType>(quantizedType)) {
    // i.e. quant<"uniform[i8:f32]{1.0}"> -> i8
    return llvm::cast<QuantizedType>(quantizedType).getStorageType();
  }
  if (llvm::isa<ShapedType>(quantizedType)) {
    // i.e. tensor<4xi8> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
    ShapedType sType = llvm::cast<ShapedType>(quantizedType);
    if (!llvm::isa<QuantizedType>(sType.getElementType())) {
      return nullptr;
    }
    Type storageType =
        llvm::cast<QuantizedType>(sType.getElementType()).getStorageType();
    if (llvm::isa<RankedTensorType>(quantizedType)) {
      return RankedTensorType::get(sType.getShape(), storageType);
    }
    if (llvm::isa<UnrankedTensorType>(quantizedType)) {
      return UnrankedTensorType::get(storageType);
    }
    if (llvm::isa<VectorType>(quantizedType)) {
      return VectorType::get(sType.getShape(), storageType);
    }
  }

  return nullptr;
}

Type QuantizedType::castFromExpressedType(Type candidateType) {
  if (candidateType == getExpressedType()) {
    // i.e. f32 -> quant<"uniform[i8:f32]{1.0}">
    return *this;
  }
  if (llvm::isa<ShapedType>(candidateType)) {
    ShapedType candidateShapedType = llvm::cast<ShapedType>(candidateType);
    if (candidateShapedType.getElementType() != getExpressedType()) {
      return nullptr;
    }

    if (llvm::isa<RankedTensorType>(candidateType)) {
      // i.e. tensor<4xf32> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
      return RankedTensorType::get(candidateShapedType.getShape(), *this);
    }
    if (llvm::isa<UnrankedTensorType>(candidateType)) {
      // i.e. tensor<xf32> -> tensor<x!quant<"uniform[i8:f32]{1.0}">>
      return UnrankedTensorType::get(*this);
    }
    if (llvm::isa<VectorType>(candidateType)) {
      // i.e. tensor<4xf32> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
      return VectorType::get(candidateShapedType.getShape(), *this);
    }
  }

  return nullptr;
}

Type QuantizedType::castToExpressedType(Type quantizedType) {
  if (llvm::isa<QuantizedType>(quantizedType)) {
    // i.e. quant<"uniform[i8:f32]{1.0}"> -> f32
    return llvm::cast<QuantizedType>(quantizedType).getExpressedType();
  }
  if (llvm::isa<ShapedType>(quantizedType)) {
    // i.e. tensor<4xi8> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
    ShapedType sType = llvm::cast<ShapedType>(quantizedType);
    if (!llvm::isa<QuantizedType>(sType.getElementType())) {
      return nullptr;
    }
    Type expressedType =
        llvm::cast<QuantizedType>(sType.getElementType()).getExpressedType();
    if (llvm::isa<RankedTensorType>(quantizedType)) {
      return RankedTensorType::get(sType.getShape(), expressedType);
    }
    if (llvm::isa<UnrankedTensorType>(quantizedType)) {
      return UnrankedTensorType::get(expressedType);
    }
    if (llvm::isa<VectorType>(quantizedType)) {
      return VectorType::get(sType.getShape(), expressedType);
    }
  }

  return nullptr;
}

Type QuantizedType::castExpressedToStorageType(Type candidateType) {
  Type expressedQuantizedType = castFromExpressedType(candidateType);
  if (!expressedQuantizedType) {
    return nullptr;
  }
  return QuantizedType::castToStorageType(expressedQuantizedType);
}

//
//
// UniformQuantizedType
UniformQuantizedType UniformQuantizedType::get(unsigned flags, Type storageType,
                                               Type expressedType, double scale,
                                               int64_t zeroPoint,
                                               int64_t storageTypeMin,
                                               int64_t storageTypeMax) {
  return Base::get(storageType.getContext(), flags, storageType, expressedType,
                   scale, zeroPoint, storageTypeMin, storageTypeMax);
}

UniformQuantizedType UniformQuantizedType::getChecked(
    function_ref<InFlightDiagnostic()> emitError, unsigned flags,
    Type storageType, Type expressedType, double scale, int64_t zeroPoint,
    int64_t storageTypeMin, int64_t storageTypeMax) {
  return Base::getChecked(emitError, storageType.getContext(), flags,
                          storageType, expressedType, scale, zeroPoint,
                          storageTypeMin, storageTypeMax);
}

LogicalResult UniformQuantizedType::verify(
    function_ref<InFlightDiagnostic()> emitError, unsigned flags,
    Type storageType, Type expressedType, double scale, int64_t zeroPoint,
    int64_t storageTypeMin, int64_t storageTypeMax) {
  if (failed(QuantizedType::verify(emitError, flags, storageType, expressedType,
                                   storageTypeMin, storageTypeMax))) {
    return failure();
  }

  // Uniform quantization requires fully expressed parameters, including
  // expressed type.
  if (!expressedType)
    return emitError() << "uniform quantization requires expressed type";

  // Verify that the expressed type is floating point.
  // If this restriction is ever eliminated, the parser/printer must be
  // extended.
  if (!llvm::isa<FloatType>(expressedType))
    return emitError() << "expressed type must be floating point";

  // Verify scale.
  if (scale <= 0.0 || std::isinf(scale) || std::isnan(scale))
    return emitError() << "illegal scale: " << scale;

  return success();
}

double UniformQuantizedType::getScale() const { return getImpl()->scale; }

int64_t UniformQuantizedType::getZeroPoint() const {
  return getImpl()->zeroPoint;
}
//
//
// UniformQuantizedPerAxisType
UniformQuantizedPerAxisType UniformQuantizedPerAxisType::get(
    unsigned flags, Type storageType, Type expressedType,
    ArrayRef<double> scales, ArrayRef<int64_t> zeroPoints,
    int32_t quantizedDimension, int64_t storageTypeMin,
    int64_t storageTypeMax) {
  DEBUG() << "UniformQuantizedPerAxisType get";
  return Base::get(storageType.getContext(), flags, storageType, expressedType,
                   scales, zeroPoints, quantizedDimension, storageTypeMin,
                   storageTypeMax);
}

UniformQuantizedPerAxisType UniformQuantizedPerAxisType::getChecked(
    function_ref<InFlightDiagnostic()> emitError, unsigned flags,
    Type storageType, Type expressedType, ArrayRef<double> scales,
    ArrayRef<int64_t> zeroPoints, int32_t quantizedDimension,
    int64_t storageTypeMin, int64_t storageTypeMax) {
  return Base::getChecked(emitError, storageType.getContext(), flags,
                          storageType, expressedType, scales, zeroPoints,
                          quantizedDimension, storageTypeMin, storageTypeMax);
}

LogicalResult UniformQuantizedPerAxisType::verify(
    function_ref<InFlightDiagnostic()> emitError, unsigned flags,
    Type storageType, Type expressedType, ArrayRef<double> scales,
    ArrayRef<int64_t> zeroPoints, int32_t quantizedDimension,
    int64_t storageTypeMin, int64_t storageTypeMax) {
  if (failed(QuantizedType::verify(emitError, flags, storageType, expressedType,
                                   storageTypeMin, storageTypeMax))) {
    return failure();
  }

  // Uniform quantization requires fully expressed parameters, including
  // expressed type.
  if (!expressedType)
    return emitError() << "uniform quantization requires expressed type";

  // Verify that the expressed type is floating point.
  // If this restriction is ever eliminated, the parser/printer must be
  // extended.
  if (!llvm::isa<FloatType>(expressedType))
    return emitError() << "expressed type must be floating point";

  // Ensure that the number of scales and zeroPoints match.
  if (scales.size() != zeroPoints.size())
    return emitError() << "illegal number of scales and zeroPoints: "
                       << scales.size() << ", " << zeroPoints.size();

  // Verify scale.
  for (double scale : scales) {
    if (scale <= 0.0 || std::isinf(scale) || std::isnan(scale))
      return emitError() << "illegal scale: " << scale;
  }

  return success();
}

ArrayRef<double> UniformQuantizedPerAxisType::getScales() const {
  return getImpl()->getScales();
}

ArrayRef<int64_t> UniformQuantizedPerAxisType::getZeroPoints() const {
  return getImpl()->getZeroPoints();
}

int32_t UniformQuantizedPerAxisType::getQuantizedDimension() const {
  return getImpl()->quantizedDimension;
}

//
//
// print function
namespace {
void printQuantParams(double scale, int64_t zeroPoint,
                      DialectAsmPrinter &printer) {
  std::stringstream ss;
  ss << std::fixed << std::setprecision(6) << scale;
  printer << ss.str();
  if (zeroPoint != 0) {
    printer << ":" << zeroPoint;
  }
}

void printStorageType(QuantizedType type, DialectAsmPrinter &printer) {
  // storage type
  unsigned storageWidth = type.getStorageTypeIntegralWidth();
  bool isSigned = type.isSigned();
  if (isSigned) {
    printer << "i" << storageWidth;
  } else {
    printer << "u" << storageWidth;
  }
}
} // namespace
/// Helper that prints a UniformQuantizedType.
void printUniformQuantizedType(UniformQuantizedType type,
                               DialectAsmPrinter &printer) {
  printer << "quant<";
  printStorageType(type, printer);
  printer << ":" << type.getExpressedType() << ", ";

  // scheme specific parameters
  printQuantParams(type.getScale(), type.getZeroPoint(), printer);
  printer << ">";
}

/// Helper that prints a UniformQuantizedPerAxisType.
void printUniformQuantizedPerAxisType(UniformQuantizedPerAxisType type,
                                      DialectAsmPrinter &printer) {
  printer << "quant<";
  printStorageType(type, printer);
  printer << ":" << type.getExpressedType();
  printer << ":";
  printer << type.getQuantizedDimension();
  printer << ", ";

  // scheme specific parameters
  ArrayRef<double> scales = type.getScales();
  ArrayRef<int64_t> zeroPoints = type.getZeroPoints();
  printer << "{";
  llvm::interleave(
      llvm::seq<size_t>(0, scales.size()), printer,
      [&](size_t index) {
        printQuantParams(scales[index], zeroPoints[index], printer);
      },
      ",");
  printer << "}>";
}

//
//
// Translation function
UniformQuantizedType QuantizationToType(mlir::OpBuilder &builder,
                                        aipubt::TensorPtr tensor) {
  mlir::Type expressedType = builder.getF32Type();
  double scale = tensor->quantization().scale();
  int64_t zeroPoint = tensor->quantization().offset();
  // TODO: May be no bits in aipu tensor, which needs to be calculated.
  int64_t storageTypeMin;
  int64_t storageTypeMax;
  mlir::Type storageType;
  bool isSigned = aipubt::Tensor::is_signed(tensor->type());
  unsigned numBits = aipubt::Tensor::dtype_bits(tensor->type());
  if (!getDefaultStorageParams(numBits, isSigned, expressedType.getContext(),
                              storageType, storageTypeMin, storageTypeMax)) {
    // Unsupported numBits (e.g. 0 or >64); fall back to non-quantized type.
    return UniformQuantizedType();
  }
  unsigned flags = isSigned ? QuantizationFlags::Signed : 0;
  // Storage Quantization Type
  return UniformQuantizedType::get(flags, storageType, expressedType, scale,
                                   zeroPoint, storageTypeMin, storageTypeMax);
}

UniformQuantizedPerAxisType
QuantizationToPerAxisType(mlir::OpBuilder &builder, aipubt::TensorPtr tensor) {
  mlir::Type expressedType = builder.getF32Type();
  auto q_scales = tensor->quantization().scales();
  std::vector<double> double_scales;
  for (size_t i = 0; i < q_scales.size(); i++) {
    float scale = q_scales[i];
    double_scales.push_back(scale < 0 ? 1e-5 : static_cast<double>(scale));
  }
  mlir::ArrayRef<double> scales(double_scales);

  auto q_offsets = tensor->quantization().offsets();
  std::vector<int64_t> int64_zeroPoints;
  for (size_t i = 0; i < q_offsets.size(); i++) {
    int64_t offset = q_offsets[i];
    int64_zeroPoints.push_back(static_cast<int64_t>(offset));
  }
  mlir::ArrayRef<int64_t> zeroPoints(int64_zeroPoints);
  int32_t quantizedDimension =
      tensor->quantization().quantization_dim(); // axis/channel
  int64_t storageTypeMin;
  int64_t storageTypeMax;
  mlir::Type storageType;
  bool isSigned = aipubt::Tensor::is_signed(tensor->type());
  unsigned numBits = aipubt::Tensor::dtype_bits(tensor->type());
  if (!getDefaultStorageParams(numBits, isSigned, expressedType.getContext(),
                              storageType, storageTypeMin, storageTypeMax)) {
    return UniformQuantizedPerAxisType();
  }

  unsigned flags = isSigned ? QuantizationFlags::Signed : 0;
  return UniformQuantizedPerAxisType::get(
      flags, storageType, expressedType, scales, zeroPoints, quantizedDimension,
      storageTypeMin, storageTypeMax);
}

static aipubt::Quantization UniformTypeToQuantization(mlir::Type type) {
  auto uniformType = llvm::dyn_cast<UniformQuantizedType>(type);
  aipubt::Quantization quant;
  quant.set_scale(uniformType.getScale());
  quant.set_offset(uniformType.getZeroPoint());
  quant.set_aipu_dtype(utils::GetDtype(uniformType.getStorageType()));
  return quant;
}

static aipubt::Quantization UniformPerAxisTypeToQuantization(mlir::Type type) {
  auto perAxisType = llvm::dyn_cast<UniformQuantizedPerAxisType>(type);
  std::vector<float> float_scales;
  for (auto scale : perAxisType.getScales()) {
    float_scales.push_back(static_cast<float>(scale));
  }
  std::vector<int> int_zeroPoints;
  for (auto zeroPoint : perAxisType.getZeroPoints()) {
    int_zeroPoints.push_back(static_cast<int>(zeroPoint));
  }
  aipubt::Quantization quant;
  quant.set_scales(float_scales);
  quant.set_offsets(int_zeroPoints);
  quant.set_aipu_dtype(utils::GetDtype(perAxisType.getStorageType()));

  return quant;
}

aipubt::Quantization GetQuantization(mlir::Type type) {
  if (llvm::dyn_cast<UniformQuantizedType>(type))
    return UniformTypeToQuantization(type);
  else if (llvm::dyn_cast<UniformQuantizedPerAxisType>(type))
    return UniformPerAxisTypeToQuantization(type);
  else
    return aipubt::Quantization();
}

} // namespace aipugc
} // namespace mlir
