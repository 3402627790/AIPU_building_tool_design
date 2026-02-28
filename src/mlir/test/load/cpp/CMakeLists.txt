cmake_minimum_required(VERSION 3.15)
project(MLIRParseExample)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-g -O0 -std=c++17 -fPIC")
set(NO_RTTI "-fno-rtti")
add_definitions(${NO_RTTI})

set(LLVM_PATH "/project/ai/scratch01/qidson01/code/llvm-project")
set(LLVM_BUILD_PATH "${LLVM_PATH}/build")

#set(LLVM_BUILD_PATH "/project/ai/zhouyi_compass/AIPU_TVM/packages/opt/llvm-project-a66376b")

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

add_executable(main main.cpp)

target_link_directories(main PUBLIC ${LLVM_BUILD_PATH}/lib)

target_link_libraries(main
  # MLIR Core
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRDialect
  
  # MLIR Dialects
  MLIRAffineDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRFuncDialect
  
  MLIRAffineToStandard
  MLIRArithToLLVM
  MLIRMemRefToLLVM
  MLIRFuncToLLVM
  
  LLVMSupport
  LLVMCore
)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(main PRIVATE -Wall -Wno-unused-parameter)
endif()