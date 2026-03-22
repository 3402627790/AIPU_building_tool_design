
project(AIPUBuildToolTest)

include_directories("${ROOT_DIR}/src")
include_directories("${ROOT_DIR}/src/quantization/quant_ops")
include_directories("${ROOT_DIR}/include")

set(AIPUBuildLIB ${AIPUBuildLIB} m aipubuildtool)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}   ${C_CXX_FLAGS} -g -O0")

FILE(GLOB_RECURSE  TEST_SRCS
${PROJECT_SOURCE_DIR}/builder/*.cpp
${PROJECT_SOURCE_DIR}/core/**.cpp
${PROJECT_SOURCE_DIR}/dtypes/*.cpp
${PROJECT_SOURCE_DIR}/qtlib/*.cpp
${PROJECT_SOURCE_DIR}/runtime/*.cpp
${PROJECT_SOURCE_DIR}/utils/*.cpp
${PROJECT_SOURCE_DIR}/simplifier/*.cpp
)


add_executable(testcase  ${TEST_SRCS} ${PROJECT_SOURCE_DIR}/main.cpp  )
target_link_libraries(testcase ${AIPUBuildLIB})