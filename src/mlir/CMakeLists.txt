cmake_minimum_required(VERSION 3.29)
project("AIPUMLIR")


set(AIPUBuildPath /project/ai/scratch01/AIPUBuilder)

message(${CMAKE_CURRENT_BINARY_DIR})
set(AIPUGC_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(AIPUGC_ROOT_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
cmake_policy(SET CMP0003 NEW)

set(INTERNAL_LLVM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/llvm_mlir_sdk")
set(MLIR_DIR "${INTERNAL_LLVM_DIR}/lib/cmake/mlir" CACHE PATH "Path to internal MLIR config" FORCE)
set(LLVM_DIR "${INTERNAL_LLVM_DIR}/lib/cmake/llvm" CACHE PATH "Path to internal LLVM config" FORCE)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

find_library(AIPUBT_LIB
  NAMES aipubuildtool
  PATHS ${AIPUBuildPath}/lib
  REQUIRED
)
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/mlir/)
include_directories(${AIPUBuildPath}/include/)
add_definitions(${LLVM_DEFINITIONS})
set(BUILD_SHARED_LIBS ${LLVM_ENABLE_SHARED_LIBS} CACHE BOOL "" FORCE)
message(STATUS "BUILD_SHARED_LIBS       : " ${BUILD_SHARED_LIBS})
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,${AIPUBuildPath}/lib")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,${AIPUBuildPath}/lib")
set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(CMAKE_CXX_FLAGS "-g -O0 -std=c++17 -fPIC")

FILE(GLOB_RECURSE MLIR_SRCS ${AIPUGC_ROOT_DIR}/mlir/**.cpp )
FILE(GLOB_RECURSE TARGET_SRCS ${AIPUGC_ROOT_DIR}/Target/**.cpp )
add_llvm_library(libAIPUMLIR MODULE
  ${MLIR_SRCS}
  ${TARGET_SRCS}

  DEPENDS
  MLIRAipuOpsIncGen
  MLIRAipuAttributesIncGen
  AIPUGCeInferenceInterfaceIncGen
  GraphOptimizationIncGen

  LINK_LIBS PUBLIC
    ${dialect_libs}
    ${conversion_libs}
    MLIRArithTransforms
    MLIRGPUTransforms
    MLIRLinalgTransforms
    MLIRMemRefTransforms
    MLIRSPIRVTransforms
    MLIRVectorTransforms
    ${AIPUBT_LIB}
  )

add_subdirectory(include/aipugc)
add_subdirectory(externalPass)
add_subdirectory(tool/aipugc)
add_custom_target(libexternalPassLink ALL)
add_custom_command(
  TARGET   libexternalPassLink
  COMMAND rm -rf  ${AIPUGC_ROOT_BIN_DIR}/tool/aipugc/Plugin
  COMMAND  mkdir -p ${AIPUGC_ROOT_BIN_DIR}/tool/aipugc/Plugin
  COMMAND  ln -s ${AIPUGC_ROOT_BIN_DIR}/externalPass/libexternalPass.so ${AIPUGC_ROOT_BIN_DIR}/tool/aipugc/Plugin;
)