#include "aiputile/AIPUTileDialect.h"
#include "aiputile/AIPUTileOps.h"

#include "aiputile/AIPUTileOpsDialect.cpp.inc"

void mlir::aiputile::AIPUTileDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aiputile/AIPUTileOps.cpp.inc"
      >();
}
