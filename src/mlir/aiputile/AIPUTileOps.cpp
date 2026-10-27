#include "aiputile/AIPUTileOps.h"
#include "mlir/IR/OpImplementation.h"
#include <limits>

using namespace mlir;

LogicalResult aiputile::verifyTensorTypes(Operation *op) {
  if (op->getNumResults() != 1)
    return op->emitOpError("requires exactly one tensor result");
  auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!type || !type.hasStaticShape() || type.getEncoding() ||
      !type.getElementType().isF32())
    return op->emitOpError("requires static, unencoded, dense FP32 tensors");
  // Bound static element indices to the signed 32-bit device index range.
  int64_t elements = 1;
  for (int64_t dim : type.getShape()) {
    if (dim <= 0 || dim > std::numeric_limits<int32_t>::max() / elements)
      return op->emitOpError(
          "requires nonempty shapes with at most INT32_MAX elements");
    elements *= dim;
  }
  for (Type operandType : op->getOperandTypes())
    if (operandType != type)
      return op->emitOpError("requires identical input/output shapes and "
                             "types; no broadcast or encoding");
  return success();
}

LogicalResult aiputile::AddOp::verify() { return verifyTensorTypes(*this); }
LogicalResult aiputile::MulOp::verify() { return verifyTensorTypes(*this); }
LogicalResult aiputile::FusedAddMulOp::verify() {
  if (failed(verifyTensorTypes(*this)))
    return failure();
  if (getTileSize() <= 0 || getTileSize() > std::numeric_limits<int32_t>::max())
    return emitOpError("tile_size must be in [1, INT32_MAX]");
  return success();
}

#define GET_OP_CLASSES
#include "aiputile/AIPUTileOps.cpp.inc"
