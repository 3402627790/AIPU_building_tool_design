
project(AIPUBuildToolCAPI)
set(REL_LIB_PATH_FLAG "")


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -fexceptions -fPIC  ${REL_LIB_PATH_FLAG}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -fPIC ${REL_LIB_PATH_FLAG}  -faligned-new")

include_directories("${ROOT_DIR}/src")
include_directories("${ROOT_DIR}/include")
include_directories("${ROOT_DIR}/tools")
include_directories("${ROOT_DIR}/3rdparty")
include_directories("${PROJECT_SOURCE_DIR}/include")

message("using python interpreter ${PYTHON_EXE}")
FILE(GLOB_RECURSE  PYTHON_SRCS ${PROJECT_SOURCE_DIR}/CGBuilder/**.py )

# for #include <python.h>
execute_process (
    COMMAND ${PYTHON_EXE} -c "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE PYTHON_INCLUDES
    RESULT_VARIABLE CMD_RESULT
)
if(NOT CMD_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to execute Python script")
endif()
include_directories(${PYTHON_INCLUDES})


include_directories("${DRIVER_API}")
link_directories(${DRIVER_PATH})
find_package(Threads REQUIRED)
FILE(GLOB_RECURSE  SRCS ${PROJECT_SOURCE_DIR}/**.cpp )


MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()

set(CPYMODULE "" )
SUBDIRLIST(ALL_MODULES ${PROJECT_SOURCE_DIR})
FOREACH(cmodule ${ALL_MODULES})
    # skip vm build
    if( NOT CPU_x86_64  AND cmodule MATCHES "^(vm)|^(grmsfloat)$")
      continue()
    endif()

    set(MODULE_SRCS "")
    set(PYCORELIB "_core")

    if (cmodule MATCHES "^(core)$")
      set(PYCORELIB "")
    endif()



    
    aux_source_directory(${cmodule} MODULE_SRCS)
    file(GLOB ALIB_FILES "${cmodule}/*.a")
    get_filename_component(cmname ${cmodule} NAME)
    set(modulename "_${cmname}")

    add_library(${modulename}  SHARED  ${MODULE_SRCS})
    list(APPEND CPYMODULE ${modulename})
    SET_TARGET_PROPERTIES(${modulename} PROPERTIES PREFIX   "")
    target_link_libraries(${modulename} aipubuildtool  ${ALIB_FILES} ${PYCORELIB})
    if(RELEASE AND CMAKE_STRIP)
      add_custom_command(
              TARGET   ${modulename}  POST_BUILD
              COMMAND ${CMAKE_STRIP}  "$<TARGET_FILE:${modulename}>"
              DEPENDS ${modulename} 
          )
    endif()
    add_custom_command(
        TARGET   ${modulename}
        COMMAND  rm -rf ${PYTHON_DES}/${modulename}.so
        COMMAND  ln -s ${CMAKE_CURRENT_BINARY_DIR}/${modulename}.so ${PYTHON_DES}/
    )
ENDFOREACH()


target_link_libraries(_driver aipudrv _core)
target_link_libraries(_tools aipubttools)
target_link_libraries(_builder aipubttools)



add_custom_command(
  TARGET   ${modulename}
  COMMAND  rm -rf ${PYTHON_DES}/*.a 
  COMMAND  cp -sv "${LAYERLIB_PATH}/*.a" "${PYTHON_DES}/" || echo "no *.a files"
  COMMAND  rm -rf ${PYTHON_DES}/*.o
  COMMAND  cp -sv "${CMAKE_CURRENT_BINARY_DIR}/../../lib/*.o" "${PYTHON_DES}/" || echo "no *.o files"
  COMMAND  rm -rf ${PYTHON_DES}/libBT*.a
  COMMAND  cp -sv "${CMAKE_CURRENT_BINARY_DIR}/../../lib/opt/*.a" "${PYTHON_DES}/" || echo "no *.a files"
  COMMAND  rm -rf ${PYTHON_DES}/libaipubuildtool.so &&  ln -s ${CMAKE_CURRENT_BINARY_DIR}/../../src/libaipubuildtool.so ${PYTHON_DES}/
  COMMAND  rm -rf ${PYTHON_DES}/libaipubttools.so &&  ln -s ${CMAKE_CURRENT_BINARY_DIR}/../../tools/libaipubttools.so ${PYTHON_DES}/
  COMMAND  rm -rf ${PYTHON_DES}/libaiputoolchain.so &&  ln -s ${CMAKE_CURRENT_BINARY_DIR}/../../modules/toolchain/libaiputoolchain.so ${PYTHON_DES}/
  COMMAND  mkdir -p  "${PYTHON_DES}/plugin" && rm -rf "${PYTHON_DES}/plugin/*.so" && touch "${PYTHON_DES}/plugin/__init__.py"
  COMMAND  cp -sfv "${PLUGIN_PATH}/*.so"    "${PYTHON_DES}/plugin" || echo "no PLUGIN_PATH *.so files"
  COMMAND  cp -sfv "${DRIVER_PATH}/*.so"    "${PYTHON_DES}/"       || echo "no DRIVER_PATH *.so files"
  COMMAND  cp -sfv "${BT_PLUGIN_PATH}/*.so" "${PYTHON_DES}/plugin" || echo "no BT_PLUGIN_PATH *.so files"
  # cp llvm toolchain
  COMMAND  mkdir -p  "${PYTHON_DES}/bin" 
  COMMAND  rm -rf "${PYTHON_DES}/bin/aipuo*"
  COMMAND cp -sfv "${LLVM_TOOLCHAIN_PATH}/aipuoas" "${PYTHON_DES}/bin/" || echo "no LLVM_TOOLCHAIN_PATH/aipuoas file"
  COMMAND cp -sfv "${LLVM_TOOLCHAIN_PATH}/aipuold" "${PYTHON_DES}/bin/" || echo "no LLVM_TOOLCHAIN_PATH/aipuold file"
  COMMAND cp -sfv "${LLVM_TOOLCHAIN_PATH}/aipuocc" "${PYTHON_DES}/bin/" || cp -sfv "${LLVM_TOOLCHAIN_PATH}/clang" "${PYTHON_DES}/bin/aipuocc" || echo "no LLVM_TOOLCHAIN_PATH/aipuocc or clang files"
  COMMAND  rm -rf "${PYTHON_DES}/include/**.h"
  COMMAND cp -sfr "${LLVM_TOOLCHAIN_PATH}/../include" "${PYTHON_DES}/" || echo "no toolchain include file"
  COMMAND cp -sf "${LLVM_TOOLCHAIN_PATH}/../lib/clang/*/include/aipu*.h" "${PYTHON_DES}/include" || echo "no toolchain aipu*.h include file"
  COMMAND  rm -rf "${PYTHON_DES}/share/**.bc"
  COMMAND cp -sfr "${LLVM_TOOLCHAIN_PATH}/../share" "${PYTHON_DES}/" || echo "no toolchain bc file"
  COMMAND  mkdir -p  "${PYTHON_DES}/lib" 
  COMMAND  rm -rf "${PYTHON_DES}/lib/**.a"
  COMMAND cp -sfvr "${LLVM_TOOLCHAIN_PATH}/../lib/libcob.a" "${PYTHON_DES}/lib" || echo "no cob lib file"

)

add_dependencies(${modulename} ds_builtin aipubuildtool)