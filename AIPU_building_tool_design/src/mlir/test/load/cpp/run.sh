#!/bin/tcsh

setenv LLVM_INSTALL_PATH /project/ai/scratch01/llvm/llvm-project-llvmorg-19.1.6
setenv MLIR_INSTALL_PATH ${LLVM_INSTALL_PATH}/build

setenv LD_LIBRARY_PATH ${MLIR_INSTALL_PATH}/lib:${LD_LIBRARY_PATH}

set BUILD_DIR = "build"

if (! -d "${BUILD_DIR}") then
    mkdir -p "${BUILD_DIR}"
endif

cd "${BUILD_DIR}"

if (! -f "Makefile") then
    cmake ..
endif

make -j32

cd ..
"./${BUILD_DIR}/main"
