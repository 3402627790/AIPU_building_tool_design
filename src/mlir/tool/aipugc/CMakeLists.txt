
set(CMAKE_CXX_FLAGS "-g -O0 -std=c++17")
set(NO_RTTI "-fno-rtti")
add_definitions(${NO_RTTI})

set(GB_ROOT_DIR ${CMAKE_SOURCE_DIR}/../../)
set(AIPUBuildLIB m aipubuildtool)
set(LLVMPublicPath /project/ai/scratch01/llvm/llvm-project-llvmorg-19.1.6/build/lib/)
#set(LLVMPublicPath /project/ai/zhouyi_compass/alezhe01/project/MLIR/llvm-project-llvmorg-19.1.6/build/lib)

include_directories("${GB_ROOT_DIR}/include")
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
include_directories(${CMAKE_SOURCE_DIR}/include)
add_executable(aipugc ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_directories(aipugc PUBLIC ${AIPUBuildPath}/lib)
target_link_directories(aipugc PUBLIC ${LLVMPublicPath})
target_link_directories(aipugc PUBLIC ${AIPUGC_ROOT_BIN_DIR}/mlir/aipugc)

add_dependencies(aipugc AIPUMlir)
target_link_libraries(aipugc
  PUBLIC
    -Wl,--push-state,--whole-archive  ${AIPUGC_ROOT_BIN_DIR}/mlir/aipugc/libAIPUMlir.a -Wl,--pop-state
    MLIRSupport
    MLIRAnalysis
    MLIRFunctionInterfaces
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRTransforms
    MLIRTransformDialect
    ${AIPUBuildLIB}
    AIPUMlir
)
