get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(CMAKE_CXX_FLAGS "-g -O0 -std=c++17 -fPIC")


add_llvm_library(libexternalPass MODULE
  EliminateRedundantTransposePass.cpp

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRPass
  AIPUMLIR
  
)
add_dependencies(libexternalPass libAIPUMLIR)

target_link_directories(libexternalPass PUBLIC ${AIPUGC_ROOT_BIN_DIR})
