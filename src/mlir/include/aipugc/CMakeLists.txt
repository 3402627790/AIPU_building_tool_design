# Most dialects should use add_mlir_dialect().  See examples/standalone.
add_mlir_dialect(AipuOps aipugc)

set(LLVM_TARGET_DEFINITIONS AipuOps.td)
mlir_tablegen(AipuAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(AipuAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRAipuAttributesIncGen)

# set(LLVM_TARGET_DEFINITIONS Passes.td)
# mlir_tablegen(Passes.h.inc -gen-pass-decls -name aipugc)
# add_public_tablegen_target(AIPUGCPassIncGen)

# Most dialects should use add_mlir_interfaces().
set(LLVM_TARGET_DEFINITIONS ShapeInferenceInterface.td)
mlir_tablegen(ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(AIPUGCeInferenceInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS GraphOptimization.td)
mlir_tablegen(GraphOptimization.cpp.inc -gen-rewriters)
add_public_tablegen_target(GraphOptimizationIncGen)
