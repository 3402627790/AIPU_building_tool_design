set(CMAKE_CXX_FLAGS "-g -O0 -std=c++17 -fPIC")
set(NO_RTTI "-fno-rtti")
add_definitions(${NO_RTTI})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

# A plain SHARED library loaded by libaipubuildtool through dlopen at runtime.
# It must not be linked into other targets; it only re-exports a stable C ABI.
add_library(AIPULLIRToolC SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/LLIRToolCApi.cpp
)

# The implementation lives in libAIPUMLIR.so (a MODULE library), so link it by
# bare name plus a link directory, exactly the way libexternalPass does.
add_dependencies(AIPULLIRToolC libAIPUMLIR)
target_link_directories(AIPULLIRToolC PUBLIC ${AIPUGC_ROOT_BIN_DIR})

# Mirror the aipugc-llir link line: libAIPUMLIR.so deliberately leaves
# LLVMInitializeZhouyi* undefined, and the LLVM Zhouyi backend static archives
# pull in LLVM core libraries. Reusing the same closure guarantees the dlopen-ed
# library is self-contained. (aipubuildtool itself is omitted here: it is
# already present in the host process and reached transitively via libAIPUMLIR.)
target_link_libraries(AIPULLIRToolC
  PUBLIC
    AIPUMLIR
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    MLIRIR
    MLIRSupport
    MLIRParser
    MLIRDialect
    MLIRLLVMDialect
    MLIRTargetLLVMIRImport
    LLVMCore
    LLVMIRReader
    LLVMBitReader
    LLVMBitstreamReader
    LLVMAsmParser
    LLVMCodeGen
    LLVMTarget
    LLVMMC
    LLVMMCParser
    LLVMBinaryFormat
    LLVMZhouyiCodeGen
    LLVMZhouyiDesc
    LLVMZhouyiInfo
    LLVMZhouyiAsmParser
    LLVMSupport
)
