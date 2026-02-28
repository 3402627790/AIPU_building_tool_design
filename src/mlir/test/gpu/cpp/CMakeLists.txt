cmake_minimum_required(VERSION 3.15)
project(run_add LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(
  -Wall
  -Wextra
  -Wpedantic
  -Wno-unused-parameter
  -fPIC
  -fno-rtti
)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)

add_executable(run_add main.cpp)

target_include_directories(run_add
  SYSTEM PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

set(LLVM_BUILD_PATH "/project/ai/scratch01/llvm/21.1.0/build")
target_link_directories(run_add PUBLIC ${LLVM_BUILD_PATH}/lib)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(run_add
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRGPUDialect
  MLIRLLVMDialect
  MLIRTransforms
  MLIRPass
  LLVMSupport
  LLVMCore
  cuda
  CUDA::cudart
)
