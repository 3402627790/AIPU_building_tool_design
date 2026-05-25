//===-- AipuTypesBase.td - AIPU type definitions -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the type definitions for the AIPU dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AIPU_TYPES_BASE
#define AIPU_TYPES_BASE

include "mlir/IR/OpBase.td"
include "aipugc/AipuOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def Aipu_UniformQuantizedType :
    DialectType<AIPUDialect, CPred<"::llvm::isa<UniformQuantizedType>($_self)">,
                "AIPU quantized type">;

def Aipu_UniformQuantizedPerAxisType :
    DialectType<AIPUDialect, CPred<"::llvm::isa<UniformQuantizedPerAxisType>($_self)">,
                "AIPU quantized per axis type">;

// 1. Base type predicates
def AIPUGCBool : Type<CPred<"$_self.isInteger(1)">, "AIPUGC boolean type">;
def AIPUGCInt4 : Type<CPred<"$_self.isInteger(4)">, "AIPUGC 4-bit integer">;
def AIPUGCInt12 : Type<CPred<"$_self.isInteger(12)">, "AIPUGC 12-bit integer">;

def Aipu_Int32 : I<32>;

// 2. Composite type system
def AIPUGCFloat : Type<Or<[
  F16.predicate,
  F32.predicate,
  F64.predicate,
  BF16.predicate
]>>;

def AIPUGCInt : Type<Or<[
  AIPUGCBool.predicate,
  I8.predicate,
  I16.predicate,
  I32.predicate,
  I64.predicate,
  AIPUGCInt4.predicate,
  AIPUGCInt12.predicate
]>>;

def AIPUGCUint : Type<Or<[
  UI8.predicate,
  UI16.predicate,
  UI32.predicate,
  UI64.predicate
]>>;

def AIPUGCAny : AnyTypeOf<[Type<Or<[
  AIPUGCFloat.predicate,
  AIPUGCInt.predicate,
  AIPUGCUint.predicate
  ]>>,
  Aipu_UniformQuantizedType,
  Aipu_UniformQuantizedPerAxisType]>;

def Aipu_FloatAttr : Attr<CPred<"::llvm::isa<::mlir::FloatAttr>($_self)">,
                          "arbitrary float attribute"> {
  let storageType = [{ ::mlir::FloatAttr }];
  let returnType = [{ ::mlir::APFloat }];
}
// 3. Tensor type definitions
//===----------------------------------------------------------------------===//
// Tensor types with constrained ranks.
//===----------------------------------------------------------------------===//

def AIPUGCAnyTensor : TensorOf<[AIPUGCAny]>;
def AIPUGCType : AnyTypeOf<[AIPUGCAnyTensor]>;

// Rank-0 (scalar) tensor
def Aipu_ScalarTensor : TensorRankOf<[AIPUGCAny], [0]>;

// We include unranked tensors as a supported type for all possible aipugc
// Tensors as unranked does not guarantee invalid. If unranked tensors exist
// they should be shape propagate used Aipu's shape inference pass and verified
// to not include any remaining unranked tensors.
def Aipu_UnrankedTensor : UnrankedTensorOf<[AIPUGCAny]>;

def Aipu_Tensor1D : AnyTypeOf<[Aipu_UnrankedTensor, 1DTensorOf<[AIPUGCAny]>], "1-d tensor", "::mlir::TensorType">;
def Aipu_Tensor2D : AnyTypeOf<[Aipu_UnrankedTensor, 2DTensorOf<[AIPUGCAny]>], "2-d tensor", "::mlir::TensorType">;
def Aipu_Tensor3D : AnyTypeOf<[Aipu_UnrankedTensor, 3DTensorOf<[AIPUGCAny]>], "3-d tensor", "::mlir::TensorType">;
def Aipu_Tensor4D : AnyTypeOf<[Aipu_UnrankedTensor, 4DTensorOf<[AIPUGCAny]>], "4-d tensor", "::mlir::TensorType">;
def Aipu_Tensor5D : AnyTypeOf<[Aipu_UnrankedTensor, TensorRankOf<[AIPUGCAny], [5]>], "5-d tensor", "::mlir::TensorType">;

// Ranked tensors up to given rank.
def Aipu_Tensor1Dto4D : AnyTypeOf<[
  Aipu_UnrankedTensor, TensorRankOf<[AIPUGCAny], [1,2,3,4]>]>;
def Aipu_Tensor1Dto6D : AnyTypeOf<[
  Aipu_UnrankedTensor, TensorRankOf<[AIPUGCAny], [1,2,3,4,5,6]>]>;

def Aipu_TensorUpto4D : AnyTypeOf<[
  Aipu_UnrankedTensor, TensorRankOf<[AIPUGCAny], [0,1,2,3,4]>]>;

def Aipu_Int32TensorUpto4D : AnyTypeOf<[
  Aipu_UnrankedTensor, TensorRankOf<[Aipu_Int32], [0,1,2,3,4]>]>;


//===----------------------------------------------------------------------===//
// Attribute predicates and classes.
//===----------------------------------------------------------------------===//
class DenseArrayMaxCtx<int n> : AttrConstraint<
    CPred<"::llvm::cast<::mlir::DenseArrayAttr>($_self).size() <= " # n>,
    "with at least " # n # " elements">;

def Aipu_Fp32ArrayAttr2 : ConfinedAttr<DenseF32ArrayAttr, [DenseArrayCount<2>]>;
def Aipu_Fp32ArrayAttr3 : ConfinedAttr<DenseF32ArrayAttr, [DenseArrayCount<3>]>;
def Aipu_Fp32ArrayAttr4 : ConfinedAttr<DenseF32ArrayAttr, [DenseArrayCount<4>]>;
def Aipu_Fp32ArrayAttr5 : ConfinedAttr<DenseF32ArrayAttr, [DenseArrayCount<5>]>;
def Aipu_Fp32ArrayAttr6 : ConfinedAttr<DenseF32ArrayAttr, [DenseArrayCount<6>]>;

def Aipu_IntArrayAttr2 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<2>]>;
def Aipu_IntArrayAttr3 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<3>]>;
def Aipu_IntArrayAttr4 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<4>]>;
def Aipu_IntArrayAttr5 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<5>]>;
def Aipu_IntArrayAttr6 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<6>]>;

def Aipu_IntArrayAttrUpto2 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayMaxCtx<2>]>;
def Aipu_IntArrayAttrUpto4 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayMaxCtx<4>]>;
def Aipu_IntArrayAttrUpto5 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayMaxCtx<5>]>;

#endif // AIPU_TYPES_BASE
