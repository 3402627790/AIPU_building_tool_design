project(AIPUBuildTool)
cmake_minimum_required(VERSION 3.14)
option(STATIC_STDLIB Off)
option(CODE_COVERAGE "CHECK CODE COVERAGE" Off)


if(CMAKE_BUILD_TYPE MATCHES "Release")
set(RELEASE on)
else()
set(RELEASE off)
endif()

option(USE_OMP "using openmp" On)
option(WITH_TEST "build with test" OFF)
option(WITH_MLIR "build MLIR module" OFF)

if(STATIC_STDLIB)
set(C_CXX_FLAGS " -static-libgcc -static-libstdc++  ${C_CXX_FLAGS}" )
endif()


if(RELEASE)
message("Building Release version")
set(C_CXX_FLAGS " -O3 -g  ${C_CXX_FLAGS}" )
set(CMAKE_EXE_LINKER_FLAGS "-s")
add_definitions(-DRELEASE_VERSION)
else()
message("Building Debug version")
set(C_CXX_FLAGS " -g   -O0  ${C_CXX_FLAGS}" )
endif()

if(CODE_COVERAGE)
message("code coverage option on")
set(C_CXX_FLAGS " -fprofile-arcs -ftest-coverage   ${C_CXX_FLAGS}")
else()
message("code coverage option off")
endif()


if(BUILDTOOL_BUILD_NUMBER)
add_definitions(-DBUILDTOOL_BUILD_NUMBER=${BUILDTOOL_BUILD_NUMBER})
endif()

set(GIT_COMMIT_ID "")
execute_process(COMMAND git rev-parse HEAD OUTPUT_VARIABLE GIT_COMMIT_ID)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  ${C_CXX_FLAGS}  -std=c11 -Wall -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}   ${C_CXX_FLAGS}  -std=c++17  -Wall -Werror  -fno-exceptions -fPIC -Wsign-promo  -Wno-null-dereference -fvisibility-inlines-hidden -Werror=int-to-pointer-cast -Werror=return-type   -Wno-enum-compare  -Wno-expansion-to-defined  -Wno-multichar -fno-strict-aliasing")

set(ROOT_DIR ${PROJECT_SOURCE_DIR})

set(CLUSTER_INSTALL_DIR "/project/ai/scratch01/AIPUBuilder/" CACHE STRING "set the cluster shared dir")
set(LAYERLIB_PATH ${CLUSTER_INSTALL_DIR}/lib/ CACHE STRING "set the LAYERLIB_PATH")
set(PLUGIN_PATH ${CLUSTER_INSTALL_DIR}/bin/plugin/ CACHE STRING "set the PLUGIN_PATH")
set(DRIVER_PATH ${CLUSTER_INSTALL_DIR}/driver/ CACHE STRING "set the DRIVER_PATH")
set(DRIVER_API ${CLUSTER_INSTALL_DIR}/driver_api/ CACHE STRING "set the DRIVER_API")
set(SIM_PATH ${CLUSTER_INSTALL_DIR}/simulator/ CACHE STRING "set the SIM_PATH for simulator")
set(LLVM_PATH ${CLUSTER_INSTALL_DIR}/llvm/ CACHE STRING "set the LLVM_PATH for llvm compiler")
set(X5SIMU_PATH ${CLUSTER_INSTALL_DIR}/x5_simu/ CACHE STRING "set the X5SIMU_PATH for x5 simulator")
set(BT_PLUGIN_PATH "${CMAKE_BINARY_DIR}/plugin/")
set(CMAKE_PREFIX_PATH  "${PROJECT_SOURCE_DIR}/cmake/")
list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/cmake/pybind11")
set(LLVM_TOOLCHAIN_PATH ${LLVM_PATH}/bin CACHE STRING "set the LLVM_TOOLCHAIN_PATH for llvm compiler")


# set(pybind11_DIR pybind11)



if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(OS_LINUX 1)
    EXECUTE_PROCESS( COMMAND uname -o COMMAND tr -d '\n' OUTPUT_VARIABLE OPERATING_SYSTEM)
      if(${OPERATING_SYSTEM} MATCHES "Android")
        set(OS_ANDROID 1)
      endif(${OPERATING_SYSTEM} MATCHES "Android")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Android")
    set(OS_ANDROID 1)
    set(OS_LINUX 1)
else()
    message(FATAL_ERROR "Operating system not supported")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
  message("Detected x86_64 architecture")
  set(CPU_x86_64 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(arm|aarch64)")
  message("Detected ARM architecture")
  set(CPU_ARM64 1)
else()
  message("Unknown architecture")
endif()

if(CPU_ARM64)
if(OS_ANDROID)
  message("disable OMP for aarch64")
  set(USE_OMP Off)
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas -Wno-nonnull" )
endif()

message("using layerlib path as ${LAYERLIB_PATH}")
message("using plugin path as ${PLUGIN_PATH}")
message("using aipu llvm toolchain as ${LLVM_PATH}")
set(PYTHON_EXE "/arm/tools/python/python/3.10.12/rhe7-x86_64/bin/python3"  CACHE STRING "set the python interpreter for build")
# set(PYTHON_EXE "/arm/tools/python/python/3.11.0/rhe7-x86_64/bin/python3"  CACHE STRING "set the python interpreter for build")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
set(REL_LIB_PATH_FLAG "-Wl,-rpath '-Wl,\$ORIGIN'  -Wl,-z,origin")
endif()

if(NOT DEFINED CMAKE_STRIP)
find_program(CMAKE_STRIP strip)
message("using strip tool ${CMAKE_STRIP}")
endif()


add_subdirectory(src)
add_subdirectory(tools)
add_subdirectory(plugin)
add_subdirectory(lib)
add_subdirectory(python)
add_subdirectory(modules)
add_subdirectory(doc)
if(WITH_TEST)
    add_subdirectory(test)
endif()
