
project(AIPUBuildTool)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -fPIC ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wdeprecated-declarations  -fno-strict-aliasing  -faligned-new")

include_directories("${ROOT_DIR}/src")
include_directories("${ROOT_DIR}/include")
include_directories("${ROOT_DIR}/3rdparty")

find_package(Threads REQUIRED)
set(AIPUBuildLIB ${AIPUBuildLIB} m dl Threads::Threads)

if(USE_OMP)
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
if(NOT DEFINED LINKER_EXE)
  set(LINKER_EXE_TARGET "")
  if(DEFINED ENV{LD})
    set(LINKER_EXE $ENV{LD})
  elseif (OS_ANDROID)
    set(LINKER_EXE  "${ANDROID_TOOLCHAIN_ROOT}/bin/ld.lld" )
    set(LINKER_EXE_TARGET "-maarch64linux") #TODO support more target
  elseif(NOT CMAKE_LINKER)
    set(LINKER_EXE ${CMAKE_LINKER})
  endif()
endif()

if(NOT DEFINED LINKER_EXE OR LINKER_EXE STREQUAL "")
  message("Please set linker!!! use ld by default")
  set(LINKER_EXE "ld")
endif()
message("Using linker ${LINKER_EXE}")
# aux_source_directory("${PROJECT_SOURCE_DIR}/core" CORE_SRCS)
aux_source_directory("${PROJECT_SOURCE_DIR}/utils" UTILS_SRCS)
FILE(GLOB_RECURSE  CORE_SRCS ${PROJECT_SOURCE_DIR}/core/**.cpp )
FILE(GLOB_RECURSE  BUILDER_SRCS ${PROJECT_SOURCE_DIR}/builder/**.cpp )
FILE(GLOB_RECURSE  CHECKER_SRCS ${PROJECT_SOURCE_DIR}/checker/**.cpp )
FILE(GLOB_RECURSE  DTYPES_SRCS ${PROJECT_SOURCE_DIR}/dtypes/**.cpp )
FILE(GLOB_RECURSE  QUANTIZATION_TRANSFORM_SRCS ${PROJECT_SOURCE_DIR}/quantization_transform/**.cpp )
FILE(GLOB_RECURSE  OPT_SRCS ${PROJECT_SOURCE_DIR}/transform/**.cpp )
FILE(GLOB_RECURSE  GSIM_SRCS ${PROJECT_SOURCE_DIR}/gsim/**.cpp )
FILE(GLOB_RECURSE  ANALYS_SRCS ${PROJECT_SOURCE_DIR}/statistic/**.cpp )
FILE(GLOB_RECURSE  ENGINE_SRCS ${PROJECT_SOURCE_DIR}/engine/**.cpp )
FILE(GLOB_RECURSE  TONGUE_SRCS ${PROJECT_SOURCE_DIR}/tongue/**.cpp )

if(OS_ANDROID)
  message("vm module not support android build")
else() 
  if(CPU_x86_64)
    FILE(GLOB_RECURSE  VM_SRCS ${PROJECT_SOURCE_DIR}/vm/**.cpp )
  else() 
    message("vm module only support x86-64 arch build")
  endif()
endif()

if(GIT_COMMIT_ID)
    string(STRIP ${GIT_COMMIT_ID} GIT_COMMIT_ID)
    if(NOT GIT_COMMIT_ID STREQUAL "")
        add_definitions(-DGIT_COMMIT_ID="${GIT_COMMIT_ID}")
        add_definitions(-DINTERNAL_MAGIC_NUMBER=${GIT_COMMIT_ID})
    endif()
endif()
add_definitions(-DXBYAK_NO_EXCEPTION)

set(SRCS ${SRCS} ${CORE_SRCS} ${UTILS_SRCS} ${BUILDER_SRCS}
        ${CHECKER_SRCS} ${DTYPES_SRCS} ${QUANTIZATION_TRANSFORM_SRCS}
        ${OPT_SRCS} ${GSIM_SRCS}
        ${ANALYS_SRCS} ${ENGINE_SRCS} ${TONGUE_SRCS} ${VM_SRCS}
    )

add_custom_target(PREPROCESS)

set(EmbedObjs "")
macro(embedfiles  FILENAME)
    message(STATUS "Objcopy'ing ${FILENAME}")
    get_filename_component(FILENAME_ONLY ${FILENAME} NAME)
    get_filename_component(FILEPATH_ONLY ${FILENAME} PATH)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME_ONLY}.o
      DEPENDS ${FILENAME}
      COMMAND cd  ${FILEPATH_ONLY} &&  ${LINKER_EXE} -r ${LINKER_EXE_TARGET} -z noexecstack --format=binary -o  ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME_ONLY}.o  ${FILENAME_ONLY}
      COMMAND cd  ${FILEPATH_ONLY} &&  objdump  -t  ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME_ONLY}.o |sed -n  -e 's/.*_binary_/extern char _binary_/p' |sed -n -e 's/_start/_start\;/p' -e 's/_end/_end\;/p'  > _BIN_RESOURCE_${FILENAME_ONLY}__.h
    )
    add_custom_target(EMBEDFILES_${FILENAME_ONLY}
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME_ONLY}.o)
    set(EmbedObjs  ${EmbedObjs} "${CMAKE_CURRENT_BINARY_DIR}/${FILENAME_ONLY}.o")
  add_dependencies(PREPROCESS EMBEDFILES_${FILENAME_ONLY})
endmacro()



add_library(M_ALLOBJECTS OBJECT ${SRCS})
embedfiles(${PROJECT_SOURCE_DIR}/statistic/cost_model_x3p.txt)
embedfiles(${PROJECT_SOURCE_DIR}/statistic/cost_model_x3p.cbin)
# add_library(aipubuildtool_static STATIC $<TARGET_OBJECTS:M_ALLOBJECTS>)
# SET_TARGET_PROPERTIES(aipubuildtool_static PROPERTIES OUTPUT_NAME "aipubuildtool")
add_library(aipubuildtool        SHARED $<TARGET_OBJECTS:M_ALLOBJECTS> ${EmbedObjs} )
add_dependencies(aipubuildtool PREPROCESS)
    



add_custom_command(
        OUTPUT ${PROJECT_SOURCE_DIR}/builder/profiler/x2_intrusive_impl.h
        COMMAND python3 ${ROOT_DIR}/scripts/raw_string2c_string.py ${PROJECT_SOURCE_DIR}/builder/profiler/x2_intrusive_impl.c x2_instrusive_impl
        DEPENDS ${ROOT_DIR}/src/builder/profiler/x2_intrusive_impl.c
    )

add_custom_target(GEN_X2_PROFILER_CL
  DEPENDS ${PROJECT_SOURCE_DIR}/builder/profiler/x2_intrusive_impl.h
)
add_dependencies(aipubuildtool GEN_X2_PROFILER_CL)

if(RELEASE AND CMAKE_STRIP)
add_custom_command(
        TARGET   aipubuildtool POST_BUILD
        COMMAND ${CMAKE_STRIP}  "$<TARGET_FILE:aipubuildtool>"
        DEPENDS aipubuildtool
    )
endif()

add_link_options($<$<CONFIG:RELEASE>:-s>)
target_link_libraries(aipubuildtool ${AIPUBuildLIB})

install(TARGETS aipubuildtool
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION include
        )
install(DIRECTORY "${ROOT_DIR}/include/" TYPE INCLUDE)


if(WITH_MLIR)
  message("Building mlir ...")
  add_subdirectory(mlir)
endif()