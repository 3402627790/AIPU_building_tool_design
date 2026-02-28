get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(CMAKE_CXX_FLAGS "-g -O0 -std=c++17")

set(LLVM_TARGET_DEFINITIONS ${AIPUGC_ROOT_DIR}/mlir/aipugc/GraphOptimization.td)
mlir_tablegen(GraphOptimization.cpp.inc -gen-rewriters)
add_public_tablegen_target(GraphOptimizationIncGen)


FILE(GLOB_RECURSE MLIRGEN_SRCS ${AIPUGC_ROOT_DIR}/mlir/aipugc/mlirgen/**.cpp )
add_llvm_library(AIPUMlir
  ${AIPUGC_ROOT_DIR}/mlir/aipugc/Dialect.cpp
  ${AIPUGC_ROOT_DIR}/mlir/aipugc/ShapeInferencePass.cpp
  ${AIPUGC_ROOT_DIR}/mlir/aipugc/Passes.cpp
  ${AIPUGC_ROOT_DIR}/mlir/aipugc/GraphOptimization.cpp
  ${AIPUGC_ROOT_DIR}/mlir/aipugc/MLIRGen.cpp
  ${AIPUGC_ROOT_DIR}/mlir/aipugc/MLIRGenRegister.cpp
  ${MLIRGEN_SRCS}

  DEPENDS
  AIPUGCeInferenceInterfaceIncGen
  GraphOptimizationIncGen
  )



llvm_update_compile_flags(AIPUMlir)
