
project(VMTest)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  ${C_CXX_FLAGS}  -g -std=c11 -Wall -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}   ${C_CXX_FLAGS} -g  -O3 -std=c++17 -Wall -Werror -fPIC -Wsign-promo  -Wno-null-dereference -fvisibility-inlines-hidden -Werror=int-to-pointer-cast -Werror=pointer-to-int-cast -Werror=return-type   -Wno-enum-compare  -Wno-expansion-to-defined  -Wno-multichar -fno-strict-aliasing")

set(AIPUBUILDERROOT "${PROJECT_SOURCE_DIR}/../..")
include_directories("${AIPUBUILDERROOT}/3rdparty")
include_directories("${AIPUBUILDERROOT}/include")
include_directories("${AIPUBUILDERROOT}/src")

FILE(GLOB_RECURSE  VM_SRC ${AIPUBUILDERROOT}/src/vm/**.cpp )
aux_source_directory("." APP_SRC)
add_executable(vmtest ${VM_SRC} ${APP_SRC})
