#ifndef AIPUTILE_OPS_H
#define AIPUTILE_OPS_H
#include "aiputile/AIPUTileDialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#define GET_OP_CLASSES
#include "aiputile/AIPUTileOps.h.inc"
namespace mlir::aiputile {
// Shared with the source conversion so unsupported layouts/types fail early.
LogicalResult verifyTensorTypes(Operation *op);
} // namespace mlir::aiputile
#endif
