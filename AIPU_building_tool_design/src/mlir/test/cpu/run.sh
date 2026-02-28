#!/bin/tcsh

set LLVM_PATH = "/project/ai/scratch01/qidson01/code/llvm-project"
set AIPUGC_PATH = "/project/ai/scratch01/qidson01/code/AIPU_building_tool_design/src/mlir"
set AIPUGC_BUILD = "$AIPUGC_PATH/build"
set LL_FILE = "$AIPUGC_BUILD/add.ll"
set OBJ_FILE = "add.o"
set MAIN_EXEC = "matrix_add"

set LLVM_BUILD = "$LLVM_PATH/build"

llc -filetype=obj $LL_FILE -o $OBJ_FILE

g++ main.cpp $OBJ_FILE -o $MAIN_EXEC

./$MAIN_EXEC

endif
