//===-- QuantUtils.h - AIPU numerical support declarations ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Function declarations for AIPU numerical support functions and quantization
// attribute builders
//
//===----------------------------------------------------------------------===//

#ifndef AIPUGC_UTILS_QUANTUTILS_H
#define AIPUGC_UTILS_QUANTUTILS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/MathExtras.h"

#include <gcompiler_api.h>

namespace mlir {
namespace aipugc {
namespace detail {
struct QuantizedTypeStorage : public mlir::TypeStorage {
  QuantizedTypeStorage(unsigned flags, Type storageType, Type expressedType,
                       int64_t storageTypeMin, int64_t storageTypeMax)
      : flags(flags), storageType(storageType), expressedType(expressedType),
        storageTypeMin(storageTypeMin), storageTypeMax(storageTypeMax) {}

  /// Flags corresponding to the bitmapped enum QuantizationFlags::FlagValue.
  unsigned flags;

  // Integral type for the storage point representation.
  Type storageType;

  // Floating point type that the quantized type approximates.
  Type expressedType;

  // The minimum value storageType can take.
  int64_t storageTypeMin;

  // The maximum value storageType can take.
  int64_t storageTypeMax;
};

struct UniformQuantizedTypeStorage : public QuantizedTypeStorage {
  struct KeyTy {
    KeyTy(unsigned flags, Type storageType, Type expressedType, double scale,
          int64_t zeroPoint, int64_t storageTypeMin, int64_t storageTypeMax)
        : flags(flags), storageType(storageType), expressedType(expressedType),
          scale(scale), zeroPoint(zeroPoint), storageTypeMin(storageTypeMin),
          storageTypeMax(storageTypeMax) {}
    /// Flags corresponding to the bitmapped enum QuantizationFlags::FlagValue.
    unsigned flags;

    // Integral type for the storage point representation.
    Type storageType;

    // Floating point type that the quantized type approximates.
    Type expressedType;

    double scale;
    int64_t zeroPoint;
    int64_t storageTypeMin;
    int64_t storageTypeMax;

    // Check for equality of two structures that share KeyTy data members
    // (by name).
    template <typename T, typename U>
    static bool genericIsEqual(const T &lhs, const U &rhs) {
      return lhs.flags == rhs.flags && lhs.storageType == rhs.storageType &&
             lhs.expressedType == rhs.expressedType && lhs.scale == rhs.scale &&
             lhs.zeroPoint == rhs.zeroPoint &&
             lhs.storageTypeMin == rhs.storageTypeMin &&
             lhs.storageTypeMax == rhs.storageTypeMax;
    }

    bool operator==(const KeyTy &other) const {
      return genericIsEqual(*this, other);
    }

    unsigned getHashValue() const {
      int64_t scaleBits = llvm::bit_cast<int64_t>(scale);
      return llvm::hash_combine(flags, storageType, expressedType, scaleBits,
                                zeroPoint, storageTypeMin, storageTypeMax);
    }
  };

  UniformQuantizedTypeStorage(const KeyTy &key)
      : QuantizedTypeStorage(key.flags, key.storageType, key.expressedType,
                             key.storageTypeMin, key.storageTypeMax),
        scale(key.scale), zeroPoint(key.zeroPoint) {}

  bool operator==(const KeyTy &key) const {
    return KeyTy::genericIsEqual(*this, key);
  }

  /// Construction.
  static UniformQuantizedTypeStorage *construct(TypeStorageAllocator &allocator,
                                                const KeyTy &key) {
    return new (allocator.allocate<UniformQuantizedTypeStorage>())
        UniformQuantizedTypeStorage(key);
  }

  static unsigned hashKey(const KeyTy &key) { return key.getHashValue(); }

  double scale;
  int64_t zeroPoint;
};

struct UniformQuantizedPerAxisTypeStorage : public QuantizedTypeStorage {
  struct KeyTy {
    KeyTy(unsigned flags, Type storageType, Type expressedType,
          ArrayRef<double> scales, ArrayRef<int64_t> zeroPoints,
          int32_t quantizedDimension, int64_t storageTypeMin,
          int64_t storageTypeMax)
        : flags(flags), storageType(storageType), expressedType(expressedType),
          scales(scales), zeroPoints(zeroPoints),
          quantizedDimension(quantizedDimension),
          storageTypeMin(storageTypeMin), storageTypeMax(storageTypeMax) {}
    /// Flags corresponding to the bitmapped enum QuantizationFlags::FlagValue.
    unsigned flags;

    // Integral type for the storage point representation.
    Type storageType;

    // Floating point type that the quantized type approximates.
    Type expressedType;

    ArrayRef<double> scales;
    ArrayRef<int64_t> zeroPoints;
    int32_t quantizedDimension;
    int64_t storageTypeMin;
    int64_t storageTypeMax;

    ArrayRef<double> getScales() const { return scales; }

    ArrayRef<int64_t> getZeroPoints() const { return zeroPoints; }

    // Check for equality of two structures that share KeyTy data members
    // (by name).
    template <typename T, typename U>
    static bool genericIsEqual(const T &lhs, const U &rhs) {
      return lhs.flags == rhs.flags && lhs.storageType == rhs.storageType &&
             lhs.expressedType == rhs.expressedType &&
             lhs.getScales() == rhs.getScales() &&
             lhs.getZeroPoints() == rhs.getZeroPoints() &&
             lhs.quantizedDimension == rhs.quantizedDimension &&
             lhs.storageTypeMin == rhs.storageTypeMin &&
             lhs.storageTypeMax == rhs.storageTypeMax;
    }

    bool operator==(const KeyTy &other) const {
      return genericIsEqual(*this, other);
    }

    unsigned getHashValue() const {
      int64_t *scalesCast = llvm::bit_cast<int64_t *>(scales.data());
      ArrayRef<int64_t> scalesBits(scalesCast, scales.size());
      return llvm::hash_combine(
          flags, storageType, expressedType,
          llvm::hash_combine_range(scalesBits.begin(), scalesBits.end()),
          llvm::hash_combine_range(zeroPoints.begin(), zeroPoints.end()),
          storageTypeMin, storageTypeMax);
    }
  };

  // We pass scales and zeroPoints in directly rather than relying on KeyTy
  // because we have to create new reallocated versions in `construct` below.
  UniformQuantizedPerAxisTypeStorage(const KeyTy &key, ArrayRef<double> scales,
                                     ArrayRef<int64_t> zeroPoints)
      : QuantizedTypeStorage(key.flags, key.storageType, key.expressedType,
                             key.storageTypeMin, key.storageTypeMax),
        scaleElements(scales.data()), zeroPointElements(zeroPoints.data()),
        quantParamsSize(scales.size()),
        quantizedDimension(key.quantizedDimension) {}

  bool operator==(const KeyTy &key) const {
    return KeyTy::genericIsEqual(*this, key);
  }

  /// Construction.
  static UniformQuantizedPerAxisTypeStorage *
  construct(TypeStorageAllocator &allocator, const KeyTy &key) {
    ArrayRef<double> scales = allocator.copyInto(key.scales);
    ArrayRef<int64_t> zeroPoints = allocator.copyInto(key.zeroPoints);
    return new (allocator.allocate<UniformQuantizedPerAxisTypeStorage>())
        UniformQuantizedPerAxisTypeStorage(key, scales, zeroPoints);
  }

  static unsigned hashKey(const KeyTy &key) { return key.getHashValue(); }

  ArrayRef<double> getScales() const {
    return ArrayRef<double>(scaleElements, quantParamsSize);
  }

  ArrayRef<int64_t> getZeroPoints() const {
    return ArrayRef<int64_t>(zeroPointElements, quantParamsSize);
  }

  const double *scaleElements;
  const int64_t *zeroPointElements;
  unsigned quantParamsSize;
  int32_t quantizedDimension;
};
} // namespace detail

/// Base class for all quantized types known to this dialect.
/// All quantized types have:
///   - storageType: The (narrower) numeric type that is being used to
///     approximate some expressed type.
///   - expressedType: The type that is being approximated.
///
/// The base class provides generic support for manipulating the types based
/// on these fields.

/// Enumeration of bit-mapped flags related to quantized types.
namespace QuantizationFlags {
enum FlagValue {
  // Indicates that the storage type should be interpreted as a signed
  // integer. The default is to interpret it as an unsigned value.
  Signed = 1,
};
} // namespace QuantizationFlags

class QuantizedType : public Type {
public:
  using ImplType = detail::QuantizedTypeStorage;
  using Type::Type;
  /// The maximum number of bits supported for storage types.
  static constexpr unsigned MaxStorageBits = 32;

  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              unsigned flags, Type storageType,
                              Type expressedType, int64_t storageTypeMin,
                              int64_t storageTypeMax);

  /// Support method to enable LLVM-style type casting.
  static bool classof(Type type);

  /// Gets the minimum possible stored by a storageType. storageTypeMin must
  /// be greater than or equal to this value.
  static int64_t getDefaultMinimumForInteger(bool isSigned,
                                             unsigned integralWidth) {
    if (isSigned) {
      return llvm::minIntN(integralWidth);
    }
    return 0;
  }

  /// Gets the maximum possible stored by a storageType. storageTypeMax must
  /// be less than or equal to this value.
  static int64_t getDefaultMaximumForInteger(bool isSigned,
                                             unsigned integralWidth) {
    if (isSigned) {
      return llvm::maxIntN(integralWidth);
    }
    return llvm::maxUIntN(integralWidth);
  }

  /// Gets the original expressed type that this quantized type approximates.
  /// Note that this presumes that the quantized type was always derived from
  /// a floating point type, which in the broadest definition, is not true (i.e.
  /// it could be some form of integral, fixed type or affine type in its own
  /// right); however, at the high level, no examples of such usage are
  /// presently known and the restriction serves some useful purposes (such as
  /// always being able to reverse a transformation or measure error). In most
  /// cases, this will be f32.
  Type getExpressedType() const;

  /// Gets the flags associated with this type. Typically a more specific
  /// accessor is appropriate.
  unsigned getFlags() const;

  // Convenience helpers.
  /// Whether the storage type should be interpreted as a signed quantity
  /// (true) or an unsigned value (false).
  bool isSigned() const {
    return (getFlags() & QuantizationFlags::Signed) ==
           QuantizationFlags::Signed;
  }

  /// Gets the underlying type used for to store values. Note that this may
  /// be signed or unsigned. Use the isSigned() accessor to differentiate.
  Type getStorageType() const;

  /// The minimum value that storageType can take.
  int64_t getStorageTypeMin() const;

  /// The maximum value that storageType can take.
  int64_t getStorageTypeMax() const;

  /// Gets the integral bit width that the underlying storage type can exactly
  /// represent.For integral storage types,this will just be their width.
  unsigned getStorageTypeIntegralWidth() const;

  /// Returns whether the candidateExpressedType is a match for this
  /// QuantizedType. This will be true if the candidate type is either a
  /// primitive type or a container type whose element type equals this
  /// QuantizedType's expressed type.
  /// Examples of compatible candidateExpressedType:
  ///   !aipugc.quant<i8:f32, 1.0> =~ f32
  ///   !aipugc.quant<i8:f32, 1.0> =~ tensor<4xf32>
  bool isCompatibleExpressedType(Type candidateExpressedType);

  /// Returns the element type as a QuantizedType or nullptr if it is not
  /// a quantized type. If the type is primitive, returns that. If it is a
  /// container (vector/tensor), return the element type.
  /// Examples:
  ///   !aipugc.quant<i8:f32, 1.0> -> !aipugc.quant<i8:f32, 1.0>
  ///   tensor<4x!aipugc.quant<i8:f32, 1.0> -> aipugc.quant<i8:f32, 1.0>
  static QuantizedType getQuantizedElementType(Type primitiveOrContainerType);

  /// Casts from a type based on the storageType to a corresponding type based
  /// on this type (returns nullptr if the cast is not valid).
  /// Examples:
  ///   i8 -> !aipugc.quant<i8:f32, 1.0>
  ///   tensor<4xi8> -> tensor<4x!aipugc.quant<i8:f32, 1.0}>>
  ///   vector<4xi8> -> vector<4x!aipugc.quant<i8:f32, 1.0>>
  Type castFromStorageType(Type candidateType);

  /// Casts from a type based on a QuantizedType to a corresponding type based
  /// on the storageType (returns nullptr if the cast is not valid).
  /// This is the inverse of castFromStorageType().
  static Type castToStorageType(Type quantizedType);

  /// Casts from a type based on the expressedType to a corresponding type based
  /// on this type (returns nullptr if the cast is not valid).
  /// Examples:
  ///   f32 -> !aipugc.quant<i8:f32, 1.0>
  ///   tensor<4xf32> -> tensor<4x!aipugc.quant<i8:f32, 1.0>>
  ///   vector<4xf32> -> vector<4x!aipugc.quant<i8:f32, 1.0>>
  Type castFromExpressedType(Type candidateType);

  /// Casts from a type based on QuantizedType to a corresponding type based
  /// on the expressedType (returns nullptr if the cast is not valid).
  /// This is the inverse of castFromExpressedType.
  static Type castToExpressedType(Type quantizedType);

  /// Casts from a type based on the expressedType to the equivalent type
  /// based on storageType by way of this QuantizedType. Equivalent to:
  /// QuantizedType::castToStorageType(castFromExpressedType(candidateType))
  /// (but with validity checks).
  /// Example (for this = !aipugc.quant<i8:f32, 1.0>):
  ///   tensor<4xf32> -> tensor<4xi8>
  Type castExpressedToStorageType(Type candidateType);

private:
  /// Hide the following methods inherited from `Type`. It is almost certainly
  /// a bug to call them from a `QuantizedType` object. Users should call
  /// `getStorageType` or `getExpressedType` to get the underlying types
  /// they want to inspect.
  using Type::isBF16;
  using Type::isF16;
  using Type::isF32;
  using Type::isF64;
  using Type::isIndex;
  using Type::isInteger;
  using Type::isSignedInteger;
  using Type::isUnsignedInteger;
};
class UniformQuantizedType
    : public Type::TypeBase<UniformQuantizedType, QuantizedType,
                            detail::UniformQuantizedTypeStorage> {
public:
  using Base::Base;
  using Base::getChecked;

  static constexpr StringLiteral name = "aipugc.quant";

  /// Gets an instance of the type with all parameters specified but not
  /// checked.
  static UniformQuantizedType get(unsigned flags, Type storageType,
                                  Type expressedType, double scale,
                                  int64_t zeroPoint, int64_t storageTypeMin,
                                  int64_t storageTypeMax);

  /// Gets an instance of the type with all specified parameters checked.
  /// Returns a nullptr convertible type on failure.
  static UniformQuantizedType
  getChecked(function_ref<InFlightDiagnostic()> emitError, unsigned flags,
             Type storageType, Type expressedType, double scale,
             int64_t zeroPoint, int64_t storageTypeMin, int64_t storageTypeMax);

  /// Verifies construction invariants and issues errors/warnings.
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              unsigned flags, Type storageType,
                              Type expressedType, double scale,
                              int64_t zeroPoint, int64_t storageTypeMin,
                              int64_t storageTypeMax);

  /// Gets the scale term. The scale designates the difference between the real
  /// values corresponding to consecutive quantized values differing by 1.
  double getScale() const;

  /// Gets the storage value corresponding to the real value 0 in the affine
  /// equation.
  int64_t getZeroPoint() const;

  // Fixed point values are real numbers divided by a scale.
  // Currently, only signed storage types are treated as fixed point.
  // A fixed point value can be obtained from an affine value by subtracting
  // the zeroPoint.
  // In the future, this may be explicit versus implied by type andzeroPoint.
  bool isFixedPoint() const { return isSigned() && getZeroPoint() == 0; }
};

/// Represents per-axis (also known as per-channel quantization).
///
/// Syntax synopsis:
///   Per-axis, all parameters expressed:
///     !quant<uniform[StorageType:ExpressedType:QuantizedDim]{QuantParams}>
///   Per-axis, optional parameters omitted:
///     !quant<uniform[StorageType]{Scale}>
///
///   StorageType: 'i'|'u' NumBits
///   ExpressedType: 'f16', 'f32', 'bf16', 'f64'
///   QuantizedDim: An integer value
///   QuantParams: (Scale ':' ZeroPoint)+
///   Scale: A legal double value
///   ZeroPoint: An integer value
class UniformQuantizedPerAxisType
    : public Type::TypeBase<UniformQuantizedPerAxisType, QuantizedType,
                            detail::UniformQuantizedPerAxisTypeStorage> {
public:
  using Base::Base;
  using Base::getChecked;

  static constexpr StringLiteral name = "aipugc.quant_per_axis";

  /// Gets an instance of the type with all parameters specified but not
  /// checked.
  static UniformQuantizedPerAxisType
  get(unsigned flags, Type storageType, Type expressedType,
      ArrayRef<double> scales, ArrayRef<int64_t> zeroPoints,
      int32_t quantizedDimension, int64_t storageTypeMin,
      int64_t storageTypeMax);

  /// Gets an instance of the type with all specified parameters checked.
  /// Returns a nullptr convertible type on failure.
  static UniformQuantizedPerAxisType
  getChecked(function_ref<InFlightDiagnostic()> emitError, unsigned flags,
             Type storageType, Type expressedType, ArrayRef<double> scales,
             ArrayRef<int64_t> zeroPoints, int32_t quantizedDimension,
             int64_t storageTypeMin, int64_t storageTypeMax);

  /// Verifies construction invariants and issues errors/warnings.
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              unsigned flags, Type storageType,
                              Type expressedType, ArrayRef<double> scales,
                              ArrayRef<int64_t> zeroPoints,
                              int32_t quantizedDimension,
                              int64_t storageTypeMin, int64_t storageTypeMax);

  /// Gets the quantization scales. The scales designate the difference between
  /// the real values corresponding to consecutive quantized values differing
  /// by 1. The ith scale corresponds to the ith slice in the
  /// quantized_dimension.
  ArrayRef<double> getScales() const;

  /// Gets the storage values corresponding to the real value 0 in the affine
  /// equation. The ith zero point corresponds to the ith slice in the
  /// quantized_dimension.
  ArrayRef<int64_t> getZeroPoints() const;

  /// Specifies the dimension of the Tensor's shape that the scales and
  /// zero_points correspond to. For example, a tensor t, with dims=[4, 3, 2, 1]
  /// with quantization params:
  ///   scales=[1.0, 2.0, 3.0], zeroPoints=[1, 2, 3], quantizedDimension=1
  /// will be quantized across the second dimension of t.
  ///   t[:, 0, :, :] will have scale[0]=1.0, zero_point[0]=1
  ///   t[:, 1, :, :] will have scale[1]=2.0, zero_point[0]=2
  ///   t[:, 2, :, :] will have scale[2]=3.0, zero_point[0]=3
  int32_t getQuantizedDimension() const;

  /// Fixed point values are real numbers divided by a scale.
  /// Currently, only signed storage types are treated as fixed point.
  /// A fixed point value can be obtained from an affine value by subtracting
  /// the zeroPoint.
  /// In the future, this may be explicit versus implied by type andzeroPoint.
  bool isFixedPoint() const {
    if (!isSigned())
      return false;
    return !llvm::is_contained(getZeroPoints(), 0);
  }
};

void printUniformQuantizedType(UniformQuantizedType type,
                               DialectAsmPrinter &printer);

void printUniformQuantizedPerAxisType(UniformQuantizedPerAxisType type,
                                      DialectAsmPrinter &printer);

UniformQuantizedType QuantizationToType(mlir::OpBuilder &builder,
                                        aipubt::TensorPtr tensor);

UniformQuantizedPerAxisType QuantizationToPerAxisType(mlir::OpBuilder &builder,
                                                      aipubt::TensorPtr tensor);

aipubt::Quantization GetQuantization(mlir::Type type);

} // namespace aipugc
} // namespace mlir

#endif // MLIR_DIALECT_AIPU_UTILS_QUANTUTILS_H
