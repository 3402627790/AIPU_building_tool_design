#!/bin/tcsh

# Note: tcsh does not have a robust equivalent to BASH_SOURCE to get the script's directory.
# This assumes you source the script from the project root directory.
set PROJECT_ROOT="`pwd`"

setenv PATH "${PROJECT_ROOT}/3rdparty/llvm_mlir_sdk/bin:${PATH}"

if ( $?LD_LIBRARY_PATH ) then
    setenv LD_LIBRARY_PATH "${PROJECT_ROOT}/3rdparty/llvm_mlir_sdk/lib:${LD_LIBRARY_PATH}"
else
    setenv LD_LIBRARY_PATH "${PROJECT_ROOT}/3rdparty/llvm_mlir_sdk/lib"
endif

if ( $?PYTHONPATH ) then
    setenv PYTHONPATH "${PROJECT_ROOT}/3rdparty/llvm_mlir_sdk/python_packages/mlir_core:${PYTHONPATH}"
else
    setenv PYTHONPATH "${PROJECT_ROOT}/3rdparty/llvm_mlir_sdk/python_packages/mlir_core"
endif
