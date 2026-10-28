#ifndef AIPUTILE_DIALECT
#define AIPUTILE_DIALECT
include "mlir/IR/OpBase.td"

def AIPUTile_Dialect : Dialect {
  let name = "aiputile";
  let cppNamespace = "::mlir::aiputile";
  let summary = "Logical tiles for statically shaped elementwise kernels";
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
  private:
    void registerTypes();
  }];
}
#endif
