//===-- AIPU_OP_BASE.td - AIPU dialect op builders -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the common definitions for the AIPU dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AIPUGC_OP_BASE
#define AIPUGC_OP_BASE

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"


// Provide a definition of the 'aipugc' dialect in the ODS framework so that we
// can define our operations.
def AIPUDialect : Dialect {
  let name = "aipugc";
  let cppNamespace = "mlir::aipugc";
  let extraClassDeclaration = [{
  private:
    // Register the builtin Attributes.
    //void registerAttributes();
  public:
    mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
    void printType(mlir::Type ty, mlir::DialectAsmPrinter &p) const override;
  }];
}

//===----------------------------------------------------------------------===//
// AIPU Attributes.
//===----------------------------------------------------------------------===//

class Aipu_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<AIPUDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// AIPU Conv Operator Builder.
//===----------------------------------------------------------------------===//

// This builder is called on all convolution operators except for TransposeConv,
// which has specialized output shape semantics. The builder also defines the
// bitwidth of the output given the bit width of the input & weight content.
def Aipu_ConvOpBuilder : OpBuilder<
  (ins
       "::mlir::Type":$outputType,
       "::mlir::Value":$input,
       "::mlir::Value":$weight,
       "::mlir::Value":$bias,
       "::mlir::DenseI64ArrayAttr":$kernel,
       "::mlir::DenseI64ArrayAttr":$pad,
       "::mlir::DenseI64ArrayAttr":$stride,
       "::mlir::DenseI64ArrayAttr":$dilation,
       "::mlir::DictionaryAttr":$dictAttr),
  [{
    buildConv($_builder, $_state, outputType,
              input, weight, bias,
              kernel, pad, stride, dilation,
              dictAttr);
  }]>;

#endif // AIPUGC_OP_BASE