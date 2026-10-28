# Build the same Tile library independently when the proprietary AIPU SDK is
# unavailable. This does not replace or stub the existing compiler/backend.
cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(AIPUTile LANGUAGES C CXX)
  find_package(MLIR REQUIRED CONFIG)
  if(NOT LLVM_VERSION_MAJOR EQUAL 19)
    message(FATAL_ERROR "AIPUTile requires LLVM/MLIR 19")
  endif()
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
  add_definitions(${LLVM_DEFINITIONS})
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../include" "${CMAKE_CURRENT_BINARY_DIR}/include")
  add_subdirectory("../include/aiputile" "include/aiputile")
  set(AIPUTILE_STANDALONE ON)
endif()

add_library(AIPUTile STATIC AIPUTileDialect.cpp AIPUTileTypes.cpp AIPUTileOps.cpp Passes.cpp)
add_dependencies(AIPUTile AIPUTileIncGen)
target_compile_options(AIPUTile PRIVATE -fno-rtti -fno-exceptions -Wall -Werror)
target_link_libraries(AIPUTile PUBLIC
  MLIRIR MLIRFuncDialect MLIRArithDialect MLIRSCFDialect MLIRMemRefDialect
  MLIRControlFlowDialect MLIRLLVMDialect MLIRPass MLIRTransforms
  MLIRSCFToControlFlow MLIRArithToLLVM MLIRMemRefToLLVM MLIRFuncToLLVM
  MLIRControlFlowToLLVM MLIRReconcileUnrealizedCasts)

add_executable(aiputile-opt aiputile-opt.cpp)
target_compile_options(aiputile-opt PRIVATE -fno-rtti -fno-exceptions -Wall -Werror)
target_link_libraries(aiputile-opt PRIVATE AIPUTile MLIROptLib)

if(AIPUTILE_STANDALONE)
  enable_testing()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME aiputile
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../tests/aiputile/run_tests.py"
      --opt "$<TARGET_FILE:aiputile-opt>" --llvm-bin "${LLVM_TOOLS_BINARY_DIR}")
endif()
