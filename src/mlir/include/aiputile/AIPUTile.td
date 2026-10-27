#ifndef AIPUTILE_DIALECT
#define AIPUTILE_DIALECT
include "mlir/IR/OpBase.td"

def AIPUTile_Dialect : Dialect {
  let name = "aiputile";
  let cppNamespace = "::mlir::aiputile";
  let summary = "Logical tiles for statically shaped elementwise kernels";
}
#endif
