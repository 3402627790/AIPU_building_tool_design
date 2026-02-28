#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

int main() {
  MLIRContext context;
  
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<affine::AffineDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<func::FuncDialect>();

//   const char *mlirCode = R"(
//   module {
//     func.func @square(%input: tensor<10x10xf32>, %output: tensor<10x10xf32>) -> tensor<10x10xf32> {
//       %x0 = linalg.square ins(%input : tensor<10x10xf32>) outs(%output : tensor<10x10xf32>) -> tensor<10x10xf32>
//       return %x0 : tensor<10x10xf32>
//     }
//   }
// )";
  const char *mlirCode = R"(
module {
  func.func @Add(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) -> memref<2x2xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32>
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 2 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<2x2xf32>
        %1 = affine.load %arg1[%arg2, %arg3] : memref<2x2xf32>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %alloc[%arg2, %arg3] : memref<2x2xf32>
      }
    }
    return %alloc : memref<2x2xf32>
  }
}
)";

  OwningOpRef<mlir::ModuleOp> module = parseSourceString<mlir::ModuleOp>(mlirCode, &context);
  
  if (!module) {
    llvm::errs() << "Failed to parse MLIR code\n";
    return 1;
  }

  llvm::outs() << "Parsed module:\n";
  module->dump();

  return 0;
}