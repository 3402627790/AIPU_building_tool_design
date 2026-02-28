#ifndef AIPUGC_MLIR_DIALECT_H_
#define AIPUGC_MLIR_DIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
// #include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/ODSSupport.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

#include "aipugc/ShapeInferenceInterface.h"

// #include "aipugc/OpsEnum.h.inc"
/// Include the auto-generated header file containing the declaration of the aipugc
/// dialect.
#include "aipugc/AipuOpsDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "aipugc/AipuAttributes.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// aipugc operations.
#define GET_OP_CLASSES
#include "aipugc/AipuOps.h.inc"

#endif // AIPUGC_MLIR_DIALECT_H_
