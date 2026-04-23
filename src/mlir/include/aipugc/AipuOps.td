//===- Ops.td - AIPUGC dialect operation definitions ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the AIPUGC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AIPUGC_OPS
#define AIPUGC_OPS

include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "aipugc/ShapeInferenceInterface.td"
include "aipugc/AipuTypesBase.td"


// Base class for aipugc dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class AIPUOp<string mnemonic, list<Trait> traits = []> :
    Op<AIPUDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// AIPUGC Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// InputOp
//===----------------------------------------------------------------------===//

// We define a aipugc operation by inheriting from our base 'AIPUOp' class above.
// Here we provide the mnemonic and a list of traits for the operation. The
// constant operation is marked as 'Pure' as it is a pure operation
// and may be removed if dead.
def InputOp : AIPUOp<"input", [Pure]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "input";
  let description = [{
    Input operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = aipugc.input : tensor<2x3xf64>
    ```
  }];

  let results = (outs AIPUGCAnyTensor:$output);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<InputOp>(...)`.

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: activation
//===----------------------------------------------------------------------===//
def ActivationOp : AIPUOp<"activation",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {

  let summary = "Activation operation";

  let description = [{
    Performs a set of permissible cast operations
    method support:RELU/RELU1/RELU6/SIGMOID/TANH/LEAKYRELU/
    THRESHOLDEDRELU/SOFTPLUS/ELU/GELU/HARDSWISH/
    CRELU/SELU/SHRINK/SILU/CELU/MISH/HARDSIGMOID/
    SWISH
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    StrAttr:$method,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let hasFolder = 0;
}

//===----------------------------------------------------------------------===//
// Operator: Cast
//===----------------------------------------------------------------------===//
def CastOp : AIPUOp<"cast",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "Element-wise data type casting with quantization support";

  let description = [{
    Casts elements from input tensor to specified data type with optional
    quantization parameters. Supports saturation clipping modes and scaling/shift
    transformations during type conversion.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    StrAttr:$to_dtype,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::StringAttr":$to_dtype,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

// We define a aipugc operation by inheriting from our base 'AIPUOp' class above.
// Here we provide the mnemonic and a list of traits for the operation. The
// constant operation is marked as 'Pure' as it is a pure operation
// and may be removed if dead.
def ConstantOp : AIPUOp<"constant", [Pure]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = aipugc.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                        : tensor<2x3xf64>
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs AIPUGCAnyTensor:$output);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

    // Build a constant with a given constant floating-point value.
    OpBuilder<(ins "double":$value)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: Abs
//===----------------------------------------------------------------------===//
def AbsOp : AIPUOp<"Abs",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Computes the absolute value of a tensor element-wise";

  let description = [{
    Given an input tensor, this operation returns a tensor containing the 
    absolute value of each element in the input tensor. The operation is 
    applied element-wise and preserves the input tensor's shape.

    Example:
      Input: [-1.0, 2.0, -3.5]
      Output: [1.0, 2.0, 3.5]
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: Negative
//===----------------------------------------------------------------------===//
def NegativeOp : AIPUOp<"Negative",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Computes the negation of a tensor element-wise";

  let description = [{
    Given an input tensor, this operation returns a tensor containing the
    negation of each element in the input tensor. The operation is
    applied element-wise and preserves the input tensor's shape.

    Example:
      Input: [-1.0, 2.0, -3.5]
      Output: [1.0, -2.0, 3.5]
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: Square
//===----------------------------------------------------------------------===//
def SquareOp : AIPUOp<"Square",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Computes the square of a tensor element-wise";

  let description = [{
    Given an input tensor, this operation returns a tensor containing the 
    square of each element in the input tensor. The operation is 
    applied element-wise and preserves the input tensor's shape.

    Example:
      Input: [-1.0, 2.0, -3.5]
      Output: [1.0, 4.0, 12.25]
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def AddOp : AIPUOp<"add", [
    Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
]> {
  let summary = "addition operation";
  let description = [{
    The "add" operation performs addition between two tensors.
    The shapes of the tensor operands are expected to match or be broadcastable.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$lhs,
    AIPUGCAnyTensor:$rhs,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs AIPUGCAnyTensor:$output);
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$lhs,
      "::mlir::Value":$rhs,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

def DivOp : AIPUOp<"div", [
    Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
]> {
  let summary = "division operation";
  let description = [{
    The "div" operation performs division between two tensors.
    The shapes of the tensor operands are expected to match or be broadcastable.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$lhs,
    AIPUGCAnyTensor:$rhs,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs AIPUGCAnyTensor:$output);
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$lhs,
      "::mlir::Value":$rhs,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

def SubOp : AIPUOp<"sub", [
    Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
]> {
  let summary = "subtraction operation";
  let description = [{
    The "sub" operation performs subtraction between two tensors.
    The shapes of the tensor operands are expected to match or be broadcastable.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$lhs,
    AIPUGCAnyTensor:$rhs,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs AIPUGCAnyTensor:$output);
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$lhs,
      "::mlir::Value":$rhs,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

def MulOp : AIPUOp<"mul", [
    Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
]> {
  let summary = "multiplication operation";
  let description = [{
    The "mul" operation performs multiplication between two tensors.
    The shapes of the tensor operands are expected to match or be broadcastable.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$lhs,
    AIPUGCAnyTensor:$rhs,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs AIPUGCAnyTensor:$output);
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$lhs,
      "::mlir::Value":$rhs,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];
}


//===----------------------------------------------------------------------===//
// Operator: eltwise
//===----------------------------------------------------------------------===//

def EltwiseOp : AIPUOp<"eltwise", [
    Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
]> {
  let summary = "Element-wise operation with strict shape matching";
  let description = [{
    Performs element-wise operations (e.g. ADD/MUL) between two tensors.
    Requires identical shapes and compatible quantization parameters.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$lhs,
    AIPUGCAnyTensor:$rhs,
    StrAttr:$method,
    StrAttr:$with_activation,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$lhs,
      "::mlir::Value":$rhs,
      "::mlir::StringAttr":$method,
      "::mlir::StringAttr":$with_activation,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];
}

//===----------------------------------------------------------------------===//
// Operator: depthwise_conv2d
//===----------------------------------------------------------------------===//
def DepthwiseConvOp : AIPUOp<"depthwise_conv2d",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Depthwise 2D Convolution operator";

  let description = [{
    Performs 2D convolutions separately over each channel of the given tensor
    input, using the weight tensor.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    4DTensorOf<[AIPUGCAny]>:$weight,
    AIPUGCAnyTensor:$bias,

    Aipu_IntArrayAttr2:$kernel,
    Aipu_IntArrayAttr4:$padding,
    Aipu_IntArrayAttr2:$strides,
    Aipu_IntArrayAttr2:$dilations,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [Aipu_ConvOpBuilder];
  let hasVerifier = 0;
}


//===----------------------------------------------------------------------===//
// Operator: FullyConnected
//===----------------------------------------------------------------------===//
def FullyConnectedOp : AIPUOp<"FullyConnected",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Fully Connected operator";

  let description = [{
    Performs a fully connected network.
  }];

  let arguments = (ins
    Aipu_Tensor2D:$input,
    2DTensorOf<[AIPUGCAny]>:$weight,
    Aipu_Tensor1D:$bias,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    Aipu_Tensor2D:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::Value":$weight,
      "::mlir::Value":$bias,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];

  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// Operator: BatchNorm
//===----------------------------------------------------------------------===//
def BatchNormOp : AIPUOp<"BatchNorm",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Batch Normalization operator";

  let description = [{
    Applies batch normalization over the input tensor using scale and bias.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    AIPUGCAnyTensor:$weight,
    AIPUGCAnyTensor:$bias,
    I32Attr:$axis,
    F32Attr:$epsilon,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::Value":$weight,
      "::mlir::Value":$bias,
      "::mlir::IntegerAttr":$axis,
      "::mlir::FloatAttr":$epsilon,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: InstanceNorm
//===----------------------------------------------------------------------===//
def InstanceNormOp : AIPUOp<"InstanceNorm",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Instance Normalization operator";

  let description = [{
    Applies instance normalization: y = weight * (x - mean) / sqrt(variance + epsilon) + bias,
    where mean and variance are computed per instance per channel (over spatial dimensions).
    Aligns with core `instance_normalization` (input, weight, bias, epsilon); constants are
    stored as "weights" / "biases" on the graph node.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    AIPUGCAnyTensor:$weight,
    AIPUGCAnyTensor:$bias,
    F32Attr:$epsilon,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::Value":$weight,
      "::mlir::Value":$bias,
      "::mlir::FloatAttr":$epsilon,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: GroupNorm
//===----------------------------------------------------------------------===//
def GroupNormOp : AIPUOp<"GroupNorm",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Group Normalization operator";

  let description = [{
    Applies group normalization along the given axis. Aligns with core
    `group_normalization` (input, weight, bias, axis, group, epsilon).
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    AIPUGCAnyTensor:$weight,
    AIPUGCAnyTensor:$bias,
    I32Attr:$axis,
    I32Attr:$group,
    F32Attr:$epsilon,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::Value":$weight,
      "::mlir::Value":$bias,
      "::mlir::IntegerAttr":$axis,
      "::mlir::IntegerAttr":$group,
      "::mlir::FloatAttr":$epsilon,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: LayerNorm
//===----------------------------------------------------------------------===//
def LayerNormOp : AIPUOp<"LayerNorm",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Layer Normalization operator";

  let description = [{
    Applies layer normalization over the input tensor along the specified axes.
    Operand layout matches BatchNorm-style: separate `weight` and `bias` tensors.
    Both are optional; when both are absent the core uses method
    WITHOUT_GAMMA_BETA.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    AIPUGCAnyTensor:$weight,
    AIPUGCAnyTensor:$bias,
    DenseI64ArrayAttr:$axis,
    F32Attr:$epsilon,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output,
    Variadic<AIPUGCAnyTensor>:$mean_invstddev
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::Value":$weight,
      "::mlir::Value":$bias,
      "::mlir::DenseI64ArrayAttr":$axis,
      "::mlir::FloatAttr":$epsilon,
      "::mlir::DictionaryAttr":$dictAttr,
      CArg<"::llvm::ArrayRef<::mlir::Type>", "{}">:$meanInvstddevTypes
    )>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: RMSNorm
//===----------------------------------------------------------------------===//
def RMSNormOp : AIPUOp<"RMSNorm",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Root Mean Square Normalization operator";

  let description = [{
    Applies RMS (Root Mean Square) normalization over the input tensor along the
    specified axes. The scale weight operand is optional. When absent, the core
    uses method WITHOUT_GAMMA_BETA. Unlike LayerNorm, RMSNorm normalizes
    by the root mean square of activations without subtracting the mean. Input
    rank must be in [2, 4]. Axis values are normalized to [0, rank). Optional
    parameters (e.g. quantization info) are carried in the generic dictionary
    attribute.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    AIPUGCAnyTensor:$weight,
    DenseI64ArrayAttr:$axis,
    F32Attr:$epsilon,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::Value":$weight,
      "::mlir::DenseI64ArrayAttr":$axis,
      "::mlir::FloatAttr":$epsilon,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];

  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def FuncOp : AIPUOp<"func", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
  let summary = "user defined function operation";
  let description = [{
    The "aipugc.func" operation represents a user defined function. These are
    callable SSA-region operations that contain aipugc computations.

    Example:

    ```mlir
    aipugc.func @main() {
      %0 = aipugc.constant dense<5.500000e+00> : tensor<f64>
      %1 = aipugc.reshape(%0 : tensor<f64>) to tensor<2x2xf64>
      aipugc.print %1 : tensor<2x2xf64>
      aipugc.return
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Returns the region on the function operation that is callable.
    Region *getCallableRegion() { return &getBody(); }
  }];
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// GenericCallOp
//===----------------------------------------------------------------------===//

def GenericCallOp : AIPUOp<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "generic call operation";
  let description = [{
    Generic calls represent calls to a user defined function that needs to
    be specialized for the shape of its arguments. The callee name is attached
    as a symbol reference via an attribute. The arguments list must match the
    arguments expected by the callee. For example:

    ```mlir
     %4 = toy.generic_call @my_func(%1, %3)
           : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    ```

    This is only valid if a function named "my_func" exists and takes two
    arguments.
  }];

  // The generic call operation takes a symbol reference attribute as the
  // callee, and inputs for the call.
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AIPUGCAnyTensor>:$inputs,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  // The generic call operation returns a single value of TensorType.
  let results = (outs AIPUGCAnyTensor:$output);

  // Specialize assembly printing and parsing using a declarative format.
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  // Add custom build methods for the generic call operation.
  let builders = [
    OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
  ];
}

//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

def PrintOp : AIPUOp<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input tensor to print.
  // We also allow a F64MemRef to enable interop during partial lowering.
  let arguments = (ins AnyTypeOf<[AIPUGCAnyTensor, F64MemRef]>:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// TileOp
//===----------------------------------------------------------------------===//
def TileOp : AIPUOp<"Tile",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Repeats a tensor along each dimension according to repeats";

  let description = [{
    Constructs a tensor by tiling the input tensor. The `repeats` attribute
    specifies the number of repetitions along each dimension.

    Example:
      Input shape: [2, 6, 16, 32], repeats = [2, 2, 5, 4]
      Output shape: [4, 12, 80, 128]
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    DenseI64ArrayAttr:$repeats,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::DenseI64ArrayAttr":$repeats,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

def ReshapeOp : AIPUOp<"reshape", [Pure]> {
  let summary = "tensor reshape operation";
  let description = [{
    Reshape operation is transforming its input tensor into a new tensor with
    the same number of elements but different shapes. For example:

    ```mlir
       %0 = aipugc.reshape (%arg1 : tensor<10xf64>) to tensor<5x2xf64>
    ```
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    DenseI64ArrayAttr:$shape,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  // We expect that the reshape operation returns a statically shaped tensor.
  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::DenseI64ArrayAttr":$shape,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];
  // Enable registering canonicalization patterns with this operation.
  let hasCanonicalizer = 0;
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : AIPUOp<"return", [Pure, HasParent<"FuncOp">,
                                 Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional tensor operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
      aipugc.func @foo() -> tensor<2xf64> {
        ...
        aipugc.return %0 : tensor<2xf64>
      }
    ```
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<AIPUGCAnyTensor>:$input);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
    let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>
  ];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: conv2d
//===----------------------------------------------------------------------===//
def Conv2DOp : AIPUOp<"conv2d",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "2D Convolution Operator";

  let description = [{
    Performs a 2D convolution over the given tensor input, using the weight
    tensor.
  }];

  let arguments = (ins
    Aipu_Tensor4D:$input,
    4DTensorOf<[AIPUGCAny]>:$weight,
    Aipu_Tensor1D:$bias,

    Aipu_IntArrayAttr2:$kernel,
    Aipu_IntArrayAttr4:$padding,
    Aipu_IntArrayAttr2:$strides,
    Aipu_IntArrayAttr2:$dilations,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    Aipu_Tensor4D:$output
  );

  let builders = [Aipu_ConvOpBuilder];
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// Operator: pool2d
//===----------------------------------------------------------------------===//
def Pool2DOp : AIPUOp<"pool2d",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "2D Pooling Operator (Max/Avg/L1)";

  let description = [{
    Performs 2D pooling (e.g., MaxPool, AvgPool, L1Pool) over the input tensor.
    The output shape is determined by kernel size, strides, padding, and ceil mode.
  }];

  let arguments = (ins
    Aipu_Tensor4D:$input,
    Aipu_IntArrayAttr2:$kernel,
    Aipu_IntArrayAttr2:$strides,
    Aipu_IntArrayAttr2:$dilations,
    Aipu_IntArrayAttr4:$padding,
    BoolAttr:$ceil_mode,
    BoolAttr:$count_include_pad,
    StrAttr:$method,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    Aipu_Tensor4D:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::DenseI64ArrayAttr":$kernel,
      "::mlir::DenseI64ArrayAttr":$strides,
      "::mlir::DenseI64ArrayAttr":$dilations,
      "::mlir::DenseI64ArrayAttr":$padding,
      "::mlir::BoolAttr":$ceil_mode,
      "::mlir::BoolAttr":$count_include_pad,
      "::mlir::StringAttr":$method,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// Operator: transpose
//===----------------------------------------------------------------------===//
def TransposeOp : AIPUOp<"transpose",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Tensor Transpose Operator";

  let description = [{
    Permutes the dimensions of a tensor according to the `perm` attribute.
    Example: For input shape [N, C, H, W] and perm=[0, 2, 3, 1], output becomes [N, H, W, C].
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    I64ArrayAttr:$perm,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` type($input) `->` type($output)
  }];

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::ArrayAttr":$perm,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];

  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// Operator: transpose_conv2d
//===----------------------------------------------------------------------===//
def TransposeConv2DOp : AIPUOp<"transpose_conv2d",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Transpose 2D Convolution operator.";

  let description = [{
    Performs a 2D transposed convolution over the given tensor input, using the
    weights tensor.
  }];

  let arguments = (ins
    Aipu_Tensor4D:$input,
    4DTensorOf<[AIPUGCAny]>:$weight,
    Aipu_Tensor1D:$bias,

    Aipu_IntArrayAttr2:$kernel,
    Aipu_IntArrayAttr4:$outpad,
    Aipu_IntArrayAttr2:$stride,
    Aipu_IntArrayAttr2:$dilation,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    Aipu_Tensor4D:$output
  );

  let builders = [Aipu_ConvOpBuilder];
  let hasVerifier = 0;
}



//===----------------------------------------------------------------------===//
// Operator: softmax
//===----------------------------------------------------------------------===//
def SoftmaxOp : AIPUOp<"softmax",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]>  {
  let summary = "Softmax op";

  let description = [{
    The operator computes the normalized exponential values for the given input: Softmax (input, axis) = Exp(input) / Reduce_Sum (Exp(input), axis = axis, keepdims = 1). The input does not need to explicitly be a 2D vector. The 'axis' attribute indicates the dimension along which softmax will be performed. The output tensor has the same shape and contains the softmax values of the corresponding input.

    Example:

    ```mlir
    %out = aipu.softmax(%in) : (tensor<21x3xf32>) -> tensor<21x3xf32>
    ```
  }];

  let arguments = (ins
   AIPUGCAnyTensor:$input,
   I32Attr:$axis,
   OptionalAttr<Aipu_FloatAttr>:$beta,
   OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
   AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::IntegerAttr":$axis,
      "::mlir::FloatAttr":$beta,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];

  let hasFolder = 0;
}

//===----------------------------------------------------------------------===//
// Operator: concat
//===----------------------------------------------------------------------===//
def ConcatOp : AIPUOp<"concat",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Concatenate tensors along a specified axis.";

  let description = [{
    Concatenates the given list of input tensors along the specified axis.
  }];

  let arguments = (ins
    Variadic<AIPUGCAnyTensor>:$input,
    I32Attr:$axis,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::ValueRange":$input,
      "::mlir::IntegerAttr":$axis,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];
}

//===----------------------------------------------------------------------===//
// Operator: split
//===----------------------------------------------------------------------===//
def SplitOp : AIPUOp<"split",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Splits a tensor into multiple tensors along the specified axis.";

  let description = [{
    Splits the input tensor into a list of output tensors along the given axis,
    according to the specified split sizes.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    DenseI64ArrayAttr:$splits,
    I32Attr:$axis,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    Variadic<AIPUGCAnyTensor>:$outputs
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::TypeRange":$outputTypes,
      "::mlir::Value":$input,
      "::mlir::DenseI64ArrayAttr":$splits,
      "::mlir::IntegerAttr":$axis,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];
}

//===----------------------------------------------------------------------===//
// Operator: ChannelShuffle
//===----------------------------------------------------------------------===//
def ChannelShuffleOp : AIPUOp<"ChannelShuffle",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Shuffles channels by splitting along axis and stacking back.";

  let description = [{
    Splits the input tensor along the last dimension (axis = -1) into `splits`
    parts (each with size dim(-1)/splits), then produces that many outputs.
    Used for channel shuffle: group and splits define the shuffle pattern.
    Only supports up to 4D input.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    I32Attr:$group,
    I32Attr:$splits,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    Variadic<AIPUGCAnyTensor>:$outputs
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::TypeRange":$outputTypes,
      "::mlir::Value":$input,
      "::mlir::IntegerAttr":$group,
      "::mlir::IntegerAttr":$splits,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: crop
//===----------------------------------------------------------------------===//
def CropOp : AIPUOp<"crop",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Crops a tensor along specified dimensions with given cropping values.";

  let description = [{
    Crops the input tensor by removing specified amounts from the beginning and end
    of each dimension. The `crops` attribute is a list of `[begin, end]` pairs,
    where each pair corresponds to a dimension of the input tensor.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    ArrayAttr:$crops,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
    "::mlir::Type":$outputType,
    "::mlir::Value":$input,
    "::mlir::ArrayAttr":$crops,
    "::mlir::DictionaryAttr":$dictAttr
    )>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: DepthToSpace
//===----------------------------------------------------------------------===//
def DepthToSpaceOp : AIPUOp<"DepthToSpace",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Rearranges data from depth into spatial dimensions with block partitioning.";

  let description = [{
    Rearranges blocks of depth data into spatial blocks. The operation takes
    an input tensor of shape [N, C, H, W] where C must be divisible by
    (block_size_x * block_size_y). Outputs a tensor of shape
    [N, C/(block_size_x*block_size_y), H*block_size_x, W*block_size_y].
    The `mode` attribute specifies the layout pattern (DCR or CRD).
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    I32Attr:$block_size_x,
    I32Attr:$block_size_y,
    StrAttr:$mode,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::IntegerAttr":$block_size_x,
      "::mlir::IntegerAttr":$block_size_y,
      "::mlir::StringAttr":$mode,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];
}

//===----------------------------------------------------------------------===//
// Operator: SpaceToDepth
//===----------------------------------------------------------------------===//
def SpaceToDepthOp : AIPUOp<"SpaceToDepth",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Rearranges spatial blocks into depth dimension.";

  let description = [{
    Rearranges spatial data blocks into the depth dimension. The operation takes
    an input tensor of shape [N, C, H, W] where H must be divisible by block_size_x
    and W must be divisible by block_size_y. Outputs a tensor of shape
    [N, C*(block_size_x*block_size_y), H/block_size_x, W/block_size_y].
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    I32Attr:$block_size_x,
    I32Attr:$block_size_y,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::IntegerAttr":$block_size_x,
      "::mlir::IntegerAttr":$block_size_y,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];
}
//===----------------------------------------------------------------------===//
// Operator: BatchToSpace
//===----------------------------------------------------------------------===//
def BatchToSpaceOp : AIPUOp<"BatchToSpace",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Rearranges data from batch into spatial dimensions with cropping.";

  let description = [{
    Rearranges data from the batch dimension into spatial dimensions. The
    operation takes an input tensor of shape [N, C, H, W] where N must be
    divisible by (block_size_y * block_size_x). Outputs a tensor of shape
    [N/(block_size_y*block_size_x), C, H*block_size_y - crop_top - crop_bottom,
     W*block_size_x - crop_left - crop_right].
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    I32Attr:$block_size_y,
    I32Attr:$block_size_x,
    I32Attr:$crop_top,
    I32Attr:$crop_bottom,
    I32Attr:$crop_left,
    I32Attr:$crop_right,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::IntegerAttr":$block_size_y,
      "::mlir::IntegerAttr":$block_size_x,
      "::mlir::IntegerAttr":$crop_top,
      "::mlir::IntegerAttr":$crop_bottom,
      "::mlir::IntegerAttr":$crop_left,
      "::mlir::IntegerAttr":$crop_right,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];

  let hasVerifier = 1;
}
//===----------------------------------------------------------------------===//
// Operator: SpaceToBatch
//===----------------------------------------------------------------------===//
def SpaceToBatchOp : AIPUOp<"SpaceToBatch",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Rearranges data from spatial dimensions into batch with padding.";

  let description = [{
    Rearranges data from spatial dimensions into the batch dimension. The
    operation takes an input tensor of shape [N, C, H, W]. After padding
    (pad_top, pad_bottom, pad_left, pad_right), the spatial dimensions are
    divided by block_size_y and block_size_x. Outputs a tensor of shape
    [N*(block_size_y*block_size_x), C, (H+pad_top+pad_bottom)/block_size_y,
     (W+pad_left+pad_right)/block_size_x].
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    I32Attr:$block_size_y,
    I32Attr:$block_size_x,
    I32Attr:$pad_top,
    I32Attr:$pad_bottom,
    I32Attr:$pad_left,
    I32Attr:$pad_right,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::IntegerAttr":$block_size_y,
      "::mlir::IntegerAttr":$block_size_x,
      "::mlir::IntegerAttr":$pad_top,
      "::mlir::IntegerAttr":$pad_bottom,
      "::mlir::IntegerAttr":$pad_left,
      "::mlir::IntegerAttr":$pad_right,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];

  let hasVerifier = 1;
}
//===----------------------------------------------------------------------===//
// Operator: Reduce
//===----------------------------------------------------------------------===//
def ReduceOp : AIPUOp<"Reduce",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Reduces a tensor along the specified axes using the given method.";

  let description = [{
    Reduces input tensor along specified axes using operations like MEAN, SUM, MAX, MIN.
    The output tensor's shape is the input shape with reduced dimensions removed.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    DenseI64ArrayAttr:$axis,
    StrAttr:$method,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::DenseI64ArrayAttr":$axis,
      "::mlir::StringAttr":$method,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];

  let hasVerifier = 1;
}
//===----------------------------------------------------------------------===//
// Operator: ArgMinMax
//===----------------------------------------------------------------------===//
def ArgMinMaxOp : AIPUOp<"ArgMinMax",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Computes indices of min or max elements along a given axis.";

  let description = [{
    ArgMinMaxOp computes the indices of the minimum or maximum values of the
    input tensor along a specified axis. The reduction is performed over a
    single axis, and the output tensor has the same rank as the input tensor
    with the reduced dimension size set to 1.

    The `method` attribute controls whether the operation computes argmin
    ("MIN") or argmax ("MAX"). Additional behaviour such as selecting the last
    index in case of ties can be encoded via extra attributes (for example
    `select_last_index`) stored in the optional dictionary attribute.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    I32Attr:$axis,
    StrAttr:$method,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::IntegerAttr":$axis,
      "::mlir::StringAttr":$method,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];

  let hasVerifier = 1;
}
//===----------------------------------------------------------------------===//
// Operator: Gather
//===----------------------------------------------------------------------===//
def GatherOp : AIPUOp<"Gather",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Gather slices from params tensor according to indices";

  let description = [{
    Gather slices from `params` tensor according to `indices` along specified axis.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    AIPUGCAnyTensor:$indices,
    I64Attr:$axis,
    I64Attr:$batch_dims,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::Value":$indices,
      "::mlir::IntegerAttr":$axis,
      "::mlir::IntegerAttr":$batch_dims,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];
}
//===----------------------------------------------------------------------===//
// Operator: Slice
//===----------------------------------------------------------------------===//
def SliceOp : AIPUOp<"Slice",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Extracts a tensor slice defined by begin/end indices and strides";

  let description = [{
    Extracts a slice from a tensor according to the following formula:
    `output[..., :, :, :] = input[begin[0]:end[0]:strides[0], ..., begin[N-1]:end[N-1]:strides[N-1]]`

    Requirements:
    - `begin`, `end`, `strides` arrays must have same length as input tensor rank
    - `strides` elements must be positive integers
    - `begin[i] <= end[i]` for all dimensions

    Example:
      Input shape: [3, 5, 8]
      begin = [1, 0, 2], end = [3, 2, 7], strides = [1, 1, 2]
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    DenseI64ArrayAttr:$begin,
    DenseI64ArrayAttr:$end,
    DenseI64ArrayAttr:$strides,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::DenseI64ArrayAttr":$begin,
      "::mlir::DenseI64ArrayAttr":$end,
      "::mlir::DenseI64ArrayAttr":$strides,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];
}
//===----------------------------------------------------------------------===//
// Operator: Pad
//===----------------------------------------------------------------------===//
def PadOp : AIPUOp<"Pad",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Pads a tensor with multi-dimensional padding configuration";

  let description = [{
    Pads input tensor with explicit per-dimension padding pairs. 
    The `pads` attribute is a list of [begin, end] pairs where each pair
    corresponds to a dimension of the input tensor.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    ArrayAttr:$pads,       // two dims struction: [[begin1,end1], [begin2,end2], ...]
    StrAttr:$mode,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::ArrayAttr":$pads,
      "::mlir::StringAttr":$mode,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];

  let hasVerifier = 1;
}
//===----------------------------------------------------------------------===//
// Operator: Resize
//===----------------------------------------------------------------------===//
def ResizeOp : AIPUOp<"Resize",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Resizes a tensor using ratio or output size";

  let description = [{
    Resizes the input tensor using the specified interpolation method.
    The resize configuration may be provided either by spatial scaling factors
    (`ratio_x` / `ratio_y`) or by the target output `size`, but not both at
    the same time.

    Parameters:
    - method: Interpolation method (BILINEAR/NEAREST)
    - ratio_x/y: Optional scaling factors for width/height
    - size: Optional output tensor size in NHWC layout
    - mode: Coordinate transformation mode. Defaults to HALF_PIXEL.
      Supported values:
        HALF_PIXEL
        ALIGN_CORNERS
        ASYMMETRIC
        PYTORCH_HALF_PIXEL
        TF_HALF_PIXEL_FOR_NN
    - nearest_mode (in dictAttr): Rounding mode when method=NEAREST.
      Defaults to ROUND_PREFER_FLOOR. Supported values:
        ROUND_PREFER_FLOOR
        ROUND_PREFER_CEIL
        FLOOR
        CEIL
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    F32Attr:$ratio_x,
    F32Attr:$ratio_y,
    StrAttr:$method,
    StrAttr:$mode,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::FloatAttr":$ratio_x,
      "::mlir::FloatAttr":$ratio_y,
      "::mlir::StringAttr":$method,
      "::mlir::StringAttr":$mode,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];

  let hasVerifier = 1;
}
//===----------------------------------------------------------------------===//
// Operator: GridSample
//===----------------------------------------------------------------------===//
def GridSampleOp : AIPUOp<"GridSample",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Samples input tensor using a normalized grid of coordinates";

  let description = [{
    Performs grid-based sampling of an input tensor using a normalized
    coordinate grid (similar to PyTorch's grid_sample). Supports different
    interpolation and padding modes and an align_corners option.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    AIPUGCAnyTensor:$grid,
    StrAttr:$mode,            // e.g. "bilinear", "nearest"
    StrAttr:$padding_mode,    // e.g. "zeros", "border", "reflection"
    BoolAttr:$align_corners,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::Value":$grid,
      "::mlir::StringAttr":$mode,
      "::mlir::StringAttr":$padding_mode,
      "::mlir::BoolAttr":$align_corners,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];
}
//===----------------------------------------------------------------------===//
// Operator: ReverseSequence
//===----------------------------------------------------------------------===//
def ReverseSequenceOp : AIPUOp<"ReverseSequence",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Reverses variable length sequences along specified dimensions";

  let description = [{
    Reverses the input tensor along the time dimension for each batch entry
    according to the sequence lengths provided. Requires explicit specification
    of batch_axis and time_axis parameters.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    AIPUGCAnyTensor:$seq_lens,
    I64Attr:$batch_axis,
    I64Attr:$time_axis,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::Value":$seq_lens,
      "::mlir::IntegerAttr":$batch_axis,
      "::mlir::IntegerAttr":$time_axis,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];
}
//===----------------------------------------------------------------------===//
// Operator: SquaredDifference
//===----------------------------------------------------------------------===//
def SquaredDifferenceOp : AIPUOp<"SquaredDifference",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Computes element-wise squared difference between two tensors";

  let description = [{
    Calculates element-wise squared difference between two input tensors:
    output = (x - y)^2.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$x,
    AIPUGCAnyTensor:$y,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$x,
      "::mlir::Value":$y,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];
}
//===----------------------------------------------------------------------===//
// Operator: Sqrt
//===----------------------------------------------------------------------===//
def SqrtOp : AIPUOp<"Sqrt",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Computes element-wise square root of the input tensor";

  let description = [{
    Calculates the square root for each element in the input tensor:
      output = sqrt(x).
    Supports optional dictionary attributes for quantization or optimization,
    but has no dedicated parameters.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];
}

//===----------------------------------------------------------------------===//
// Operator: Rsqrt
//===----------------------------------------------------------------------===//
def RsqrtOp : AIPUOp<"Rsqrt",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Computes reciprocal of square root element-wise, optionally using a lookup table (LUT)";

  let description = [{
    Calculates reciprocal square root of input tensor values: output = 1 / sqrt(x).
    Supports optional lookup table parameters for quantization or optimization.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];
}

//===----------------------------------------------------------------------===//
// Operator: Cosine
//===----------------------------------------------------------------------===//
def CosineOp : AIPUOp<"Cosine",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Computes element-wise cosine of the input tensor";

  let description = [{
    Calculates the cosine for each element in the input tensor.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];
}
//===----------------------------------------------------------------------===//
// Operator: Exp
//===----------------------------------------------------------------------===//
def ExpOp : AIPUOp<"Exp",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Computes element-wise exponential of the input tensor";

  let description = [{
    Calculates the exponential for each element in the input tensor:
      output = exp(x).
    Supports optional lookup table (LUT) parameters for int8 quantization,
    where lut_type, lut_offset, lut_size, lut_shape can be passed via dictAttr.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];
}

//===----------------------------------------------------------------------===//
// Operator: Log
//===----------------------------------------------------------------------===//
def LogOp : AIPUOp<"Log",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Computes element-wise natural logarithm of the input tensor";

  let description = [{
    Calculates the natural logarithm for each element in the input tensor:
      output = log(x).
    Supports optional lookup table (LUT) parameters for int8 quantization,
    where lut_type, lut_offset, lut_size, lut_shape can be passed via dictAttr.

    Note: Mathematically log(x) requires x > 0. The compiler cannot verify
    operand values at compile time (only types/shapes are available), so
    ensuring valid input range is the caller's responsibility.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];
}

//===----------------------------------------------------------------------===//
// Operator: Pow
//===----------------------------------------------------------------------===//
def PowOp : AIPUOp<"Pow",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Computes element-wise power: base^exponent";

  let description = [{
    Calculates element-wise power for each pair of elements:
      output = base^exponent.
    The shapes of base and exponent are expected to match or be broadcastable.
    Supports optional dictionary attributes for quantization (e.g. LUT).
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$base,
    AIPUGCAnyTensor:$exponent,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$base,
      "::mlir::Value":$exponent,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];
}

//===----------------------------------------------------------------------===//
// Operator: Quantize
//===----------------------------------------------------------------------===//
def QuantizeOp : AIPUOp<"Quantize",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Quantizes input tensor values using specified parameters";

  let description = [{
    Quantizes input tensor values using specified parameters.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];
}
//===----------------------------------------------------------------------===//
// Operator: BasicLSTM
//===----------------------------------------------------------------------===//
def BasicLSTMOp : AIPUOp<"BasicLSTM",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Basic LSTM Operator";

  let description = [{
    Performs LSTM operation using input, hidden state and cell state tensors,
    with configurable parameters via dictionary attributes.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    AIPUGCAnyTensor:$hidden_state,
    AIPUGCAnyTensor:$cell_state,
    StrAttr:$out_sequence,
    StrAttr:$activations,
    StrAttr:$direction,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::Value":$hidden_state,
      "::mlir::Value":$cell_state,
      "::mlir::StringAttr":$out_sequence,
      "::mlir::StringAttr":$activations,
      "::mlir::StringAttr":$direction,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];
}
//===----------------------------------------------------------------------===//
// Operator: MatMulOp
//===----------------------------------------------------------------------===//
def MatMulOp : AIPUOp<"MatMul",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Matrix multiplication operator";

  let description = [{
    Performs matrix multiplication of two input tensors.
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    AIPUGCAnyTensor:$weight,
    BoolAttr:$trans_a,
    BoolAttr:$trans_b,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$outputType,
      "::mlir::Value":$input,
      "::mlir::Value":$weight,
      "::mlir::BoolAttr":$trans_a,
      "::mlir::BoolAttr":$trans_b,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];
}

//===----------------------------------------------------------------------===//
// Operator: RegionOp
//===----------------------------------------------------------------------===//
def RegionOp : AIPUOp<"Region",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Region operator for YOLO object detection post-processing";

  let description = [{
    Performs region extraction for YOLO object detection models.
    Decodes raw YOLO outputs to structured detection results including
    bounding box coordinates, confidence scores, and class information.

    Input shape: [batch_size, s, s, anchor_num, num_classes + 5]
    Output: scores, coordinates, boxes_number, valid_classes, total_class
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$input,
    F32Attr:$obj_thresh,
    I32Attr:$max_box_num,
    I32Attr:$box_per_grid,
    I32Attr:$class_num,
    I32Attr:$grid_width,
    I32Attr:$grid_height,
    BoolAttr:$grid_compensate,
    DenseF64ArrayAttr:$anchors,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$scores,
    AIPUGCAnyTensor:$coordinates,
    AIPUGCAnyTensor:$boxes_number,
    AIPUGCAnyTensor:$valid_classes,
    AIPUGCAnyTensor:$total_class
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$scoresType,
      "::mlir::Type":$coordinatesType,
      "::mlir::Type":$boxesNumberType,
      "::mlir::Type":$validClassesType,
      "::mlir::Type":$totalClassType,
      "::mlir::Value":$input,
      "::mlir::FloatAttr":$obj_thresh,
      "::mlir::IntegerAttr":$max_box_num,
      "::mlir::IntegerAttr":$box_per_grid,
      "::mlir::IntegerAttr":$class_num,
      "::mlir::IntegerAttr":$grid_width,
      "::mlir::IntegerAttr":$grid_height,
      "::mlir::BoolAttr":$grid_compensate,
      "::mlir::DenseF64ArrayAttr":$anchors,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: NmsOp
//===----------------------------------------------------------------------===//
def NmsOp : AIPUOp<"NMS",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Non-Maximum Suppression operator";

  let description = [{
    Performs Non-Maximum Suppression (NMS) for object detection.
    Selects a subset of bounding boxes based on IoU threshold and score
    threshold to remove overlapping detections.

    Inputs:
      - box_coor: 3D tensor of box coordinates, shape [batch, N, 4]
      - box_num_per_class: 2D tensor, number of boxes per class per batch
      - class_num_per_batch: 2D tensor, number of classes per batch
      - score: 2D tensor of scores

    Outputs:
      - selected_coordinates: selected box coordinates
      - selected_box_number: selected box count per class
      - selected_score: selected scores
      - selected_indices: selected box indices
  }];

  let arguments = (ins
    AIPUGCAnyTensor:$box_coor,
    AIPUGCAnyTensor:$box_num_per_class,
    AIPUGCAnyTensor:$class_num_per_batch,
    AIPUGCAnyTensor:$score,
    F32Attr:$score_threshold,
    F32Attr:$iou_threshold,
    I32Attr:$max_output_size,
    F32Attr:$soft_nms_sigma,
    BoolAttr:$center_point_box,
    StrAttr:$method,
    OptionalAttr<DictionaryAttr>:$dictAttr
  );

  let results = (outs
    AIPUGCAnyTensor:$selected_coordinates,
    AIPUGCAnyTensor:$selected_box_number,
    AIPUGCAnyTensor:$selected_score,
    AIPUGCAnyTensor:$selected_indices
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$selectedCoordinatesType,
      "::mlir::Type":$selectedBoxNumberType,
      "::mlir::Type":$selectedScoreType,
      "::mlir::Type":$selectedIndicesType,
      "::mlir::Value":$box_coor,
      "::mlir::Value":$box_num_per_class,
      "::mlir::Value":$class_num_per_batch,
      "::mlir::Value":$score,
      "::mlir::FloatAttr":$score_threshold,
      "::mlir::FloatAttr":$iou_threshold,
      "::mlir::IntegerAttr":$max_output_size,
      "::mlir::FloatAttr":$soft_nms_sigma,
      "::mlir::BoolAttr":$center_point_box,
      "::mlir::StringAttr":$method,
      "::mlir::DictionaryAttr":$dictAttr
    )>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------


//===----------------------------------------------------------------------===//
#endif // AIPUGC_OPS
