#include "aiputile/AIPUTileOps.h"
#include "mlir/IR/OpImplementation.h"
#include <limits>

using namespace mlir;

namespace {
LogicalResult checkTensor(Operation *op, RankedTensorType type,
                          int64_t &elements) {
  if (!type || !type.hasStaticShape() || type.getEncoding() ||
      !type.getElementType().isF32())
    return op->emitOpError("requires static, unencoded, dense FP32 tensors");
  elements = 1;
  for (int64_t dim : type.getShape()) {
    if (dim <= 0 || dim > std::numeric_limits<int32_t>::max() / elements)
      return op->emitOpError(
          "requires nonempty shapes with at most INT32_MAX elements");
    elements *= dim;
  }
  return success();
}

LogicalResult checkBoundary(Operation *op, RankedTensorType tensor,
                            aiputile::TileType tile) {
  int64_t elements;
  if (failed(checkTensor(op, tensor, elements)))
    return failure();
  if (tile.getCapacity() != elements ||
      tile.getElementType() != tensor.getElementType())
    return op->emitOpError("tensor shape and element type must match the tile "
                           "capacity and element type");
  return success();
}
} // namespace

LogicalResult aiputile::verifySourceTensorTypes(Operation *op) {
  if (op->getNumResults() != 1)
    return op->emitOpError("requires exactly one tensor result");
  auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  int64_t elements;
  if (failed(checkTensor(op, type, elements)))
    return failure();
  for (Type operandType : op->getOperandTypes())
    if (operandType != type)
      return op->emitOpError("requires identical input/output shapes and "
                             "types; no broadcast or encoding");
  return success();
}

LogicalResult aiputile::verifyTileTypes(Operation *op) {
  if (op->getNumResults() != 1)
    return op->emitOpError("requires exactly one tile result");
  auto type = dyn_cast<TileType>(op->getResult(0).getType());
  if (!type)
    return op->emitOpError("requires tile values, not tensor values");
  for (Type operandType : op->getOperandTypes())
    if (operandType != type)
      return op->emitOpError(
          "requires identical tile capacities and element types");
  return success();
}

LogicalResult aiputile::FromTensorOp::verify() {
  return checkBoundary(*this, getInput().getType(), getOutput().getType());
}
LogicalResult aiputile::ToTensorOp::verify() {
  return checkBoundary(*this, getOutput().getType(), getInput().getType());
}
LogicalResult aiputile::AddOp::verify() { return verifyTileTypes(*this); }
LogicalResult aiputile::MulOp::verify() { return verifyTileTypes(*this); }

LogicalResult aiputile::TileLoopOp::verify() {
  if (getTileSize() <= 0 || getTileSize() > std::numeric_limits<int32_t>::max())
    return emitOpError("tile_size must be in [1, INT32_MAX]");
  if (getInputs().empty())
    return emitOpError("requires at least one tile input");
  auto outputType = getOutput().getType();
  for (auto input : getInputs())
    if (input.getType() != outputType)
      return emitOpError("requires identical full input/output tile domains");
  if (!llvm::hasSingleElement(getBody()))
    return emitOpError("requires exactly one nonempty body block");
  Block &block = getBody().front();
  if (block.getNumArguments() != getInputs().size() + 2)
    return emitOpError("body requires one tile argument per input followed by "
                       "offset and valid_length indices");
  auto localType = TileType::get(
      getContext(), std::min<int64_t>(getTileSize(), outputType.getCapacity()),
      outputType.getElementType());
  for (auto arg : block.getArguments().take_front(getInputs().size()))
    if (arg.getType() != localType)
      return emitOpError("body tile arguments must have capacity "
                         "min(tile_size, input capacity)");
  for (auto arg : block.getArguments().take_back(2))
    if (!arg.getType().isIndex())
      return emitOpError(
          "body offset and valid_length arguments must have index type");
  if (block.empty() || !isa<YieldOp>(block.back()))
    return emitOpError("body must terminate with aiputile.yield");
  for (Operation &op : block) {
    if (!isa<AddOp, MulOp, YieldOp>(op))
      return op.emitOpError(
          "only tile-local Add/Mul and yield are supported in tile_loop");
    for (Type type : op.getOperandTypes())
      if (type != localType)
        return op.emitOpError("must use the current tile partition type");
    for (Type type : op.getResultTypes())
      if (type != localType)
        return op.emitOpError("must produce the current tile partition type");
  }
  return success();
}

LogicalResult aiputile::YieldOp::verify() {
  auto loop = (*this)->getParentOfType<TileLoopOp>();
  if (!loop || (*this)->getBlock() != &loop.getBody().front())
    return emitOpError("must directly terminate a tile_loop body");
  auto output = loop.getOutput().getType();
  if (loop.getTileSize() <= 0 ||
      loop.getTileSize() > std::numeric_limits<int32_t>::max())
    return loop.emitOpError("tile_size must be in [1, INT32_MAX]");
  auto expected = TileType::get(
      getContext(), std::min<int64_t>(loop.getTileSize(), output.getCapacity()),
      output.getElementType());
  if (getValue().getType() != expected)
    return emitOpError("yielded tile must match the current partition capacity "
                       "and element type");
  return success();
}

#define GET_OP_CLASSES
#include "aiputile/AIPUTileOps.cpp.inc"
