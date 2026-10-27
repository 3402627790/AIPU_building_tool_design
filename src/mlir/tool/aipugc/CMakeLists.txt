set(CMAKE_CXX_FLAGS "-g -O0 -std=c++17 -fPIC")
set(NO_RTTI "-fno-rtti")
add_definitions(${NO_RTTI})

set(GB_ROOT_DIR ${CMAKE_SOURCE_DIR}/../../)
set(AIPUBuildLIB m aipubuildtool)
#set(LLVMPublicPath /project/ai/scratch01/llvm/21.1.0//build/lib/)

include_directories("${GB_ROOT_DIR}/include")
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(
  aipugc
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
)

add_executable(
  aipugc-llir
  ${CMAKE_CURRENT_SOURCE_DIR}/llir_main.cpp
)

# Search priority, avoid search /project/ai/scratch01/AIPUBuilder/lib/libAIPUMLIR.so
target_link_directories(aipugc PUBLIC ${AIPUGC_ROOT_BIN_DIR})
target_link_directories(aipugc-llir PUBLIC ${AIPUGC_ROOT_BIN_DIR})
#target_link_directories(aipugc PUBLIC ${LLVMPublicPath})
target_link_directories(aipugc PUBLIC ${AIPUBuildPath}/lib)
target_link_directories(aipugc-llir PUBLIC ${AIPUBuildPath}/lib)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_dependencies(aipugc libAIPUMLIR)
add_dependencies(aipugc-llir libAIPUMLIR)

target_link_libraries(aipugc
  PUBLIC
    AIPUTile
    ${AIPUBuildLIB}
    AIPUMLIR

    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}

    MLIRIR
    MLIRSupport
    MLIRParser
    MLIRPass
    MLIRAnalysis
    MLIRTransforms
    MLIRTransformUtils

    MLIRSideEffectInterfaces
    MLIRFunctionInterfaces
    MLIRCallInterfaces
    MLIRCastInterfaces

    MLIRBuiltinToLLVMIRTranslation
    MLIRCAPIIR
    MLIRCAPITransforms
    MLIRDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRTosaDialect
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRVectorDialect
    MLIRMathDialect
    MLIRIndexDialect
    MLIRLLVMDialect
    MLIRTransformDialect

    MLIRTosaToLinalg
    MLIRTosaToArith
    MLIRAffineToStandard
    MLIRSCFToControlFlow
    MLIRSCFToGPU
    MLIRArithToLLVM
    MLIRMemRefToLLVM
    MLIRFuncToLLVM
    MLIRLinalgToStandard
    MLIRIndexToLLVM
    MLIRMathToLLVM
    MLIRVectorToSCF
    MLIRReconcileUnrealizedCasts

    MLIRBufferizationTransforms
    MLIRMemRefTransforms
    MLIRLinalgTransforms

    MLIRSPIRVDialect
    MLIRGPUToSPIRV
    MLIRSPIRVTransforms
    MLIRSPIRVConversion
    MLIRArithToSPIRV
    MLIRFuncToSPIRV
    MLIRMemRefToSPIRV
    MLIRSPIRVUtils
    MLIRSPIRVSerialization

    MLIRGPUToNVVMTransforms
    MLIRNVGPUToNVVM
    MLIRNVVMToLLVM
    MLIRNVVMToLLVMIRTranslation

    MLIRLLVMToLLVMIRTranslation
    MLIRExecutionEngine
    MLIRTargetLLVMIRExport
    LLVMExecutionEngine
    LLVMRuntimeDyld
    LLVMMCJIT
    LLVMOrcJIT
    LLVMSupport

    LLVMX86CodeGen
    LLVMX86Desc
    LLVMX86Info
    LLVMX86AsmParser
    LLVMX86Disassembler
    LLVMSPIRVCodeGen
    LLVMSPIRVDesc
    LLVMSPIRVInfo
    LLVMAArch64CodeGen
    LLVMAArch64Desc
    LLVMAArch64Info
    LLVMAArch64AsmParser
    LLVMAArch64Disassembler
    LLVMAArch64Utils
    LLVMAMDGPUCodeGen
    LLVMAMDGPUDesc
    LLVMAMDGPUInfo
    LLVMAMDGPUAsmParser
    LLVMAMDGPUDisassembler
    LLVMAMDGPUUtils
    LLVMNVPTXCodeGen
    LLVMNVPTXDesc
    LLVMNVPTXInfo
    LLVMAIPUCodeGen
    LLVMAIPUDesc
    LLVMAIPUInfo
    LLVMAIPUAsmParser
    LLVMAIPUDisassembler
    LLVMZhouyiCodeGen
    LLVMZhouyiDesc
    LLVMZhouyiInfo
    LLVMZhouyiAsmParser
    LLVMZhouyiDisassembler
)

target_link_libraries(aipugc-llir
  PUBLIC
    ${AIPUBuildLIB}
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
