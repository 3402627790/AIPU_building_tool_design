
project(AIPUBuildToolLayerLibLink)

add_custom_target(LINKLAYERLIB ALL)
add_custom_target(LINKPlugin ALL)
set(PLUGIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/../tools/plugin/")
set(X2_CL_COMPILER "/project/ai/scratch01/toolchain/llvm/binary/bin/aipuocc")
set(OPT_LIB "${CMAKE_CURRENT_BINARY_DIR}/opt/")
set(INTERNAL_LIB "aipubt_internal_lib.o")
set(HOST_LIB "cpshost")

FILE(GLOB_RECURSE  X2_SRCS ${PROJECT_SOURCE_DIR}/x2/**.cl)
FILE(GLOB_RECURSE  X3_SRCS ${PROJECT_SOURCE_DIR}/opt/x3/**.cl)
FILE(GLOB_RECURSE  HOST_SRCS ${PROJECT_SOURCE_DIR}/host/**.cpp)

MACRO(COMPILECL result sources ARFILE)
  set(CUSTOM_OBJ_FILES "")
  foreach(FILE ${sources})
    get_filename_component(OUTPUT_OBJ ${FILE} NAME_WE)
    add_custom_command(
      COMMAND mkdir -p ${OPT_LIB} 
      OUTPUT ${OPT_LIB}/${OUTPUT_OBJ}.o
      COMMAND ${X2_CL_COMPILER} -c ${FILE} -target aipu  -O2  -o  "${OPT_LIB}/${OUTPUT_OBJ}.o"
      DEPENDS ${FILE}
      COMMENT "Compiling ${FILE} with custom compiler"
    )
    message("compiling src ${OPT_LIB}/${OUTPUT_OBJ}.o")

    list(APPEND CUSTOM_OBJ_FILES ${OPT_LIB}/${OUTPUT_OBJ}.o)
  endforeach()
  set(LIBFILE "${OPT_LIB}/libBT${ARFILE}.a")
  add_custom_command(
      OUTPUT ${LIBFILE}
      COMMAND mkdir -p ${OPT_LIB} 
      COMMAND ar rcs  ${LIBFILE} ${CUSTOM_OBJ_FILES}
      DEPENDS ${CUSTOM_OBJ_FILES}
      COMMENT "archiving lib ${ARFILE} ${LIBFILE}"
    )
  set(${result} ${LIBFILE})
ENDMACRO()

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${INTERNAL_LIB}"
    COMMAND ${X2_CL_COMPILER} ${X2_SRCS} -target aipu -c -O2  -o "${CMAKE_CURRENT_BINARY_DIR}/${INTERNAL_LIB}"
    DEPENDS ${X2_SRCS}
)

COMPILECL(X3_OBJS "${X3_SRCS}" "optx3")

message(${X3_OBJS})

add_custom_target(GEN_X2_INTERNAL_LIB
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${INTERNAL_LIB}" ${X3_OBJS} 
)
add_dependencies(aipubuildtool GEN_X2_INTERNAL_LIB)

add_library(${HOST_LIB} SHARED ${HOST_SRCS})

target_include_directories(${HOST_LIB} PUBLIC "${ROOT_DIR}/include")

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/${INTERNAL_LIB}" ${X3_OBJS}
    DESTINATION lib
)

install(
    TARGETS ${HOST_LIB}
    DESTINATION lib
)

add_custom_command(
  TARGET   LINKLAYERLIB
  COMMAND  rm -rf ${CMAKE_CURRENT_BINARY_DIR}/*.a
  COMMAND  cp -s ${LAYERLIB_PATH}/*.a ${CMAKE_CURRENT_BINARY_DIR} || echo "cannot link LAYERLIB_PATH"	
  COMMAND  rm -rf ${CMAKE_CURRENT_BINARY_DIR}/libBT*.a
  COMMAND  cp -s ${OPT_LIB}/*.a ${CMAKE_CURRENT_BINARY_DIR} || echo "cannot link LAYERLIB_PATH"	
  COMMAND  rm -rf ${CMAKE_CURRENT_BINARY_DIR}/libcob.a
  COMMAND  cp -s ${LLVM_TOOLCHAIN_PATH}/../lib/libcob.a ${CMAKE_CURRENT_BINARY_DIR} || echo "cannot link libcob.a"	
  DEPENDS aipubuildtool
)
add_dependencies(LINKLAYERLIB GEN_X2_INTERNAL_LIB)

add_custom_command(	
  TARGET   LINKPlugin 
  COMMAND  mkdir -p ${PLUGIN_DIR} && rm -rf ${PLUGIN_DIR}/*.so 
  COMMAND  cp -s ${PLUGIN_PATH}/*.so ${PLUGIN_DIR} || echo "cannot link PLUGIN_PATH"
  COMMAND  cp -sf ${CMAKE_CURRENT_BINARY_DIR}/../plugin/*.so ${PLUGIN_DIR} || echo "cannot link BT PLUGIN_PATH"
)