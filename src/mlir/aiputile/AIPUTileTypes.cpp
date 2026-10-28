#include "aiputile/AIPUTileTypes.h"
#include "aiputile/AIPUTileDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include <limits>

using namespace mlir;

LogicalResult
aiputile::TileType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                           int64_t capacity, Type elementType) {
  if (capacity <= 0 || capacity > std::numeric_limits<int32_t>::max())
    return emitError() << "tile capacity must be in [1, INT32_MAX]";
  if (!elementType.isF32())
    return emitError() << "tile element type must be FP32";
  return success();
}

Type aiputile::TileType::parse(AsmParser &parser) {
  SmallVector<int64_t> dimensions;
  Type elementType;
  auto loc = parser.getCurrentLocation();
  if (parser.parseLess() ||
      parser.parseDimensionList(dimensions, /*allowDynamic=*/false) ||
      parser.parseType(elementType) || parser.parseGreater())
    return {};
  if (dimensions.size() != 1) {
    parser.emitError(loc, "tile requires exactly one static capacity");
    return {};
  }
  return getChecked([&] { return parser.emitError(loc); }, parser.getContext(),
                    dimensions.front(), elementType);
}

void aiputile::TileType::print(AsmPrinter &printer) const {
  printer << '<' << getCapacity() << 'x' << getElementType() << '>';
}

#define GET_TYPEDEF_CLASSES
#include "aiputile/AIPUTileTypes.cpp.inc"

void aiputile::AIPUTileDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "aiputile/AIPUTileTypes.cpp.inc"
      >();
}
