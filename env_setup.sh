# #!/bin/bash

# PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# export PATH="${PROJECT_ROOT}/3rdparty/llvm_mlir_sdk/bin:${PATH}"
# export LD_LIBRARY_PATH="${PROJECT_ROOT}/3rdparty/llvm_mlir_sdk/lib:${LD_LIBRARY_PATH}"
# export PYTHONPATH="${PROJECT_ROOT}/3rdparty/llvm_mlir_sdk/python_packages/mlir_core:${PYTHONPATH}"



# for tcshrc

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


