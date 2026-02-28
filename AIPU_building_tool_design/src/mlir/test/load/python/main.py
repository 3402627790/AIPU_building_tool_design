from mlir import ir
from mlir import dialects as ops
from mlir.dialects import arith, affine, memref, func

context = ir.Context()

mlir_code = """
module {
  func.func @Add(%arg0: memref<2x2xf32, strided<[?, ?], offset: ?>>, %arg1: memref<2x2xf32, strided<[?, ?], offset: ?>>) -> memref<2x2xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32>
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 2 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<2x2xf32, strided<[?, ?], offset: ?>>
        %1 = affine.load %arg1[%arg2, %arg3] : memref<2x2xf32, strided<[?, ?], offset: ?>>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %alloc[%arg2, %arg3] : memref<2x2xf32>
      }
    }
    return %alloc : memref<2x2xf32>
  }
}
"""

module = ir.Module.parse(mlir_code, context)

if module is None:
    print("Failed to parse MLIR code")
    exit(1)

print("Parsed module:")
print(module)
